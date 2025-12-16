import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import os
from PIL import Image
import open_clip
from sklearn.preprocessing import LabelEncoder
import csv
import torch.serialization
from tqdm import tqdm
from model_utils import SimpleMLP  # Assuming SimpleMLP is defined in model_utils


# --- Global Configuration and Setup ---
# device is defined in __main__

# --- 1. Model Loading (Remains the same) ---

def load_model_and_encoder(model_path, device):
    """Loads the trained model state and label encoder."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at: {model_path}")

    # Fix for WeightsUnpickler error: Allowlist the LabelEncoder
    try:
        # Attempt to import the specific type needed for allowlisting
        from sklearn.preprocessing._label import LabelEncoder as SklearnLabelEncoder
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([SklearnLabelEncoder])
    except Exception as e:
        # This warning is harmless if the default safe load works
        print(f"Warning: Could not allowlist LabelEncoder. Error: {e}")

    # Load the checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print("Initial safe load failed. Attempting load with weights_only=False...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_classes = checkpoint['num_classes']
    label_encoder = checkpoint['label_encoder']

    model = SimpleMLP(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, label_encoder


# --- 2. Multi-Scale Tiled Feature Extractor ---

class TiledBioclipFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:imageomics/bioclip-2')
        self.model.eval()
        self.preprocess = preprocess_train
        self.model.to(self.device)
        print('Bioclip-2 Model Loaded...')

    def extract_tile_features(self, image: Image.Image, grid_size: int) -> list:
        """Extracts features from all tiles of a given grid size."""
        w, h = image.size
        # Handle the case where grid_size leads to non-integer tile dimensions
        if w % grid_size != 0 or h % grid_size != 0:
            # For simplicity, we'll use integer division for tile size
            # and potentially lose a few pixels at the edges, or just
            # use the common method of resizing to a power of 2 first,
            # but keeping the original logic for direct tile cropping.
            pass

        tile_w = w // grid_size
        tile_h = h // grid_size

        feature_list = []

        for i in range(grid_size):
            for j in range(grid_size):
                left = j * tile_w
                upper = i * tile_h
                right = (j + 1) * tile_w
                lower = (i + 1) * tile_h

                tile = image.crop((left, upper, right, lower))

                image_input = self.preprocess(tile).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    # L2-normalize the features
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    feature_list.append(image_features.squeeze(0).cpu())

        return feature_list

    def extract_multi_scale_features(self, image_path: str, grid_sizes: list) -> list:
        """Extracts features across multiple grid sizes."""
        try:
            image = Image.open(image_path).convert("RGB")
            all_features = []
            for grid_size in grid_sizes:
                all_features.extend(self.extract_tile_features(image, grid_size))
            return all_features

        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            return []
        except Exception as e:
            print(f"Error processing image {image_path}: {e}. Skipping.")
            return []


# --- 3. Multi-Scale Prediction Logic ---

def predict_multi_label_multi_scale(model, feature_extractor, image_paths, grid_sizes, device):
    """
    Predicts multi-label species for a list of images using a multi-scale tiling approach.
    Returns a list of dictionaries with quadrat_id and ALL raw tile probabilities.
    """
    all_raw_predictions = []

    # Run a single pass to get all probability scores for all tiles and all scales
    for image_path in tqdm(image_paths, desc="Extracting features and running prediction"):
        quadrat_id = os.path.splitext(os.path.basename(image_path))[0]

        # This returns features from 1x1 (1), 2x2 (4), and 4x4 (16) tiles -> total 21 features
        feature_list = feature_extractor.extract_multi_scale_features(image_path, grid_sizes)

        if not feature_list:
            continue

        # Combine all features into a single batch tensor
        batch_features = torch.stack(feature_list).to(device)  # Shape: (Total_Tiles, 768)

        # 2. Batch Prediction
        with torch.no_grad():
            output = model(batch_features)  # Output shape: (Total_Tiles, Num_Classes)

            # Convert logits to probabilities using Softmax
            probabilities = nn.functional.softmax(output, dim=1)  # Shape: (Total_Tiles, Num_Classes)

            all_raw_predictions.append({
                "quadrat_id": quadrat_id,
                "probabilities": probabilities.cpu()  # Store probabilities on CPU for later processing
            })

    return all_raw_predictions


# --- 4. Threshold Tuning and Output Generation ---

def tune_and_save_predictions(all_raw_predictions, label_encoder, threshold_range, output_dir, file_prefix):
    """
    Applies multiple thresholds to the raw predictions and saves a CSV for each.
    """
    print("\n--- Starting Threshold Tuning and CSV Generation ---")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for threshold in threshold_range:
        predictions = []
        print(f"Processing predictions for THRESHOLD: {threshold:.1f}")

        for p_data in tqdm(all_raw_predictions, desc="Applying threshold"):
            quadrat_id = p_data['quadrat_id']
            probabilities = p_data['probabilities']  # Shape: (Total_Tiles, Num_Classes)

            # 1. Apply Thresholding for Multi-Label on ALL tiles simultaneously
            # Find all (tile_index, class_index) where P > THRESHOLD
            predicted_indices = (probabilities > threshold).nonzero(as_tuple=False)

            all_predicted_class_indices = []

            if predicted_indices.numel() > 0:
                # Get the class indices that exceeded the threshold across all tiles
                class_indices = predicted_indices[:, 1].cpu().numpy()
                all_predicted_class_indices.extend(class_indices)

            # Fallback: If no tile exceeds the threshold, use the top 1 prediction from the tile
            # with the highest confidence overall (common strategy).
            if not all_predicted_class_indices:
                # Max probability across all tiles and all classes
                max_prob, max_index_flat = torch.max(probabilities.flatten(), 0)
                # Find the corresponding class index
                max_class_index = max_index_flat % probabilities.size(1)
                all_predicted_class_indices.append(max_class_index.item())

            # 2. Consolidate and Inverse Transform
            # Get unique class indices and convert back to original species IDs
            unique_indices = np.unique(all_predicted_class_indices)
            predicted_species_ids = label_encoder.inverse_transform(unique_indices)

            predictions.append({
                "quadrat_id": quadrat_id,
                "species_ids": list(predicted_species_ids)
            })

        # Save the result for the current threshold
        output_file = os.path.join(output_dir, f"{file_prefix}_Threshold_{threshold:.1f}.csv")
        format_and_save_csv(predictions, output_file)

    print("\n--- Threshold Tuning Complete ---")


def format_and_save_csv(predictions, output_file):
    """
    Formats the predictions into the required double-bracket CSV format.
    """
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

        writer.writerow(["quadrat_id", "species_ids"])

        for p in predictions:
            id_list_str = ", ".join(map(str, p['species_ids']))
            formatted_ids = f"[{id_list_str}]"
            writer.writerow([p['quadrat_id'], formatted_ids])

    print(f"Saved submission file: {output_file}")


if __name__ == "__main__":
    MODEL_PATH = "/local/scratch1/siam/saved_models/plant_clef/bioclip2_mlp_classifier.pth"
    # IMPORTANT: The TEST_IMAGE_DIR variable points to a CSV, not a directory of images.
    # The image path base is derived from this path structure.
    TEST_CSV_PATH = '/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/PlantCLEF2025_test.csv'
    TEST_IMAGE_BASE_DIR = '/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/'

    # Multi-scale Grid Parameters
    GRID_SIZES = [1, 2, 4]  # 1x1 (full image), 2x2, 4x4
    THRESHOLD_RANGE = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    OUTPUT_BASE_DIR = "../results/results_multi_scale"
    OUTPUT_FILE_PREFIX = "multi_scale"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model and label encoder
    model, label_encoder = load_model_and_encoder(MODEL_PATH, device)
    print(f"Model loaded successfully with {label_encoder.classes_.size} classes.")

    # Initialize the Tiled feature extractor
    feature_extractor = TiledBioclipFeatureExtractor()

    # Get list of all test images
    test_data = pd.read_csv(TEST_CSV_PATH, sep=';', dtype={'partner': str})

    all_test_files = []
    for f_name in list(test_data['quadrat_id']):
        # Assuming the image file is named quadrat_id.jpg
        image_path = os.path.join(TEST_IMAGE_BASE_DIR, f_name) + '.jpg'
        all_test_files.append(image_path)

    image_paths = all_test_files
    print(f'Total Test Files : {len(image_paths)}')

    # 1. Run multi-scale prediction and collect ALL raw probabilities
    all_raw_predictions = predict_multi_label_multi_scale(
        model,
        feature_extractor,
        image_paths,
        GRID_SIZES,
        device
    )

    # 2. Apply threshold tuning and save output files
    tune_and_save_predictions(
        all_raw_predictions,
        label_encoder,
        THRESHOLD_RANGE,
        OUTPUT_BASE_DIR,
        OUTPUT_FILE_PREFIX
    )