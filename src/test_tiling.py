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
import math
from model_utils import SimpleMLP
from tqdm import tqdm


def load_model_and_encoder(model_path):
    """Loads the trained model state and label encoder."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at: {model_path}")

    # Fix for WeightsUnpickler error: Allowlist the LabelEncoder
    try:
        from sklearn.preprocessing._label import LabelEncoder as SklearnLabelEncoder
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([SklearnLabelEncoder])
    except Exception as e:
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


# --- 3. Tiled Feature Extractor (Incorporating your BiCLIP logic) ---

class TiledBioclipFeatureExtractor:
    def __init__(self):
        self.model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip-2')
        self.model.eval()
        self.preprocess = preprocess_train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print('Bioclip-2 Model Loaded...')

    def extract_tile_features(self, image: Image.Image, grid_size: int) -> list:
        w, h = image.size
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
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    feature_list.append(image_features.squeeze(0).cpu())

        return feature_list

    def extract_features_tiled(self, image_path: str, grid_size: int) -> list:
        try:
            image = Image.open(image_path).convert("RGB")
            return self.extract_tile_features(image, grid_size)

        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            return []
        except Exception as e:
            print(f"Error processing image {image_path}: {e}. Skipping.")
            return []


def predict_multi_label_tiled(model, label_encoder, feature_extractor, image_paths):
    predictions = []

    for image_path in tqdm(image_paths):
        quadrat_id = os.path.splitext(os.path.basename(image_path))[0]

        feature_list = feature_extractor.extract_features_tiled(image_path, GRID_SIZE)

        if not feature_list:
            continue

        # Combine all features into a single batch tensor
        batch_features = torch.stack(feature_list).to(device)  # Shape: (N*N, 768)

        # 2. Batch Prediction
        with torch.no_grad():
            output = model(batch_features)  # Output shape: (N*N, Num_Classes)

            # Convert logits to probabilities using Softmax
            probabilities = nn.functional.softmax(output, dim=1)  # Shape: (N*N, Num_Classes)

            # 3. Apply Thresholding for Multi-Label on ALL tiles simultaneously
            # Find all (tile_index, class_index) where P > THRESHOLD
            predicted_indices = (probabilities > PREDICTION_THRESHOLD).nonzero(as_tuple=False)

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

            # 4. Consolidate and Inverse Transform
            # Get unique class indices and convert back to original species IDs
            unique_indices = np.unique(all_predicted_class_indices)
            predicted_species_ids = label_encoder.inverse_transform(unique_indices)

            predictions.append({
                "quadrat_id": quadrat_id,
                "species_ids": list(predicted_species_ids)
            })

    return predictions


# --- 5. Output Formatting and CSV Generation (Same as before) ---

def format_and_save_csv(predictions, output_file):
    """
    Formats the predictions into the required double-bracket CSV format.
    """
    print(f"\nFormatting and saving results to {output_file}...")

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

        writer.writerow(["quadrat_id", "species_ids"])

        for p in predictions:
            id_list_str = ", ".join(map(str, p['species_ids']))
            formatted_ids = f"[{id_list_str}]"
            writer.writerow([p['quadrat_id'], formatted_ids])

    print("Submission file successfully created.")



if __name__ == "__main__":
    MODEL_PATH = "/local/scratch1/siam/saved_models/plant_clef/bioclip2_mlp_classifier.pth"
    TEST_IMAGE_DIR = '/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/PlantCLEF2025_test.csv'

    # Grid Parameters
    GRID_SIZE = 4  # N=3 -> 3x3 grid, resulting in 9 tiles per image
    PREDICTION_THRESHOLD = 0.5  # Same critical hyperparameter for multi-label prediction
    OUTPUT_CSV_FILE = f"../results/tiled_grid_{GRID_SIZE}_Thresold_{PREDICTION_THRESHOLD}.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the trained model and label encoder
    model, label_encoder = load_model_and_encoder(MODEL_PATH)
    print(f"Model loaded successfully with {label_encoder.classes_.size} classes.")

    # Initialize the Tiled feature extractor
    feature_extractor = TiledBioclipFeatureExtractor()

    # Get list of all test images
    test_data = pd.read_csv(TEST_IMAGE_DIR, sep=';', dtype={'partner': str})
    test_image_path = '/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/'

    all_test_files = []
    for f_name in list(test_data['quadrat_id']):
        image_path = os.path.join(test_image_path, f_name)
        image_path = image_path + '.jpg'
        all_test_files.append(image_path)

    print(f'Total Test Files : {len(all_test_files)}')
    image_paths = all_test_files

    print(f"Found {len(image_paths)} test images to process.")

    # Run prediction using the tiled approach
    predictions = predict_multi_label_tiled(model, label_encoder, feature_extractor, image_paths)

    # Generate output file
    format_and_save_csv(predictions, OUTPUT_CSV_FILE)
