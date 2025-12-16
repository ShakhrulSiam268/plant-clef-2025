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
from model_utils import SimpleMLP  # Assuming SimpleMLP is correctly defined in model_utils
from tqdm import tqdm
from collections import Counter


def load_model_and_encoder(model_path):
    """Loads the trained model state and label encoder."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at: {model_path}")

    # Fix for WeightsUnpickler error: Allowlist the LabelEncoder
    try:
        from sklearn.preprocessing._label import LabelEncoder as SklearnLabelEncoder
        # Check if the function exists before calling
        if hasattr(torch.serialization, 'add_safe_globals'):
            torch.serialization.add_safe_globals([SklearnLabelEncoder])
    except Exception as e:
        print(f"Warning: Could not allowlist LabelEncoder. Error: {e}")

    # Load the checkpoint
    try:
        # Assuming 'device' is defined globally or passed, using 'cpu' for safe loading
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    except Exception as e:
        print("Initial safe load failed. Attempting load with weights_only=False...")
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)

    # Move relevant items to the correct device
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_classes = checkpoint['num_classes']
    label_encoder = checkpoint['label_encoder']

    # device is expected to be defined in the main block before this function is called
    global device
    model = SimpleMLP(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, label_encoder


# --- 3. Tiled Feature Extractor (Incorporating your BiCLIP logic) ---

class TiledBioclipFeatureExtractor:
    def __init__(self):
        self.model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:imageomics/bioclip-2')
        self.model.eval()
        self.preprocess = preprocess_train
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print('Bioclip-2 Model Loaded...')
        # Store model's expected input size (e.g., 224 for CLIP models)
        self.input_size = self.preprocess.transforms[0].size

    def extract_tile_features(self, image: Image.Image, grid_size: int) -> list:
        w, h = image.size
        tile_w = w // grid_size
        tile_h = h // grid_size

        feature_list = []

        #

        for i in range(grid_size):
            for j in range(grid_size):
                left = j * tile_w
                upper = i * tile_h
                right = (j + 1) * tile_w
                lower = (i + 1) * tile_h

                # Crop the tile
                tile = image.crop((left, upper, right, lower))

                # BiCLIP/OpenCLIP preprocessing
                image_input = self.preprocess(tile).unsqueeze(0).to(self.device)

                # Extract features
                with torch.no_grad():
                    image_features = self.model.encode_image(image_input)
                    # L2-normalize the features
                    image_features /= image_features.norm(dim=-1, keepdim=True)
                    feature_list.append(image_features.squeeze(0).cpu())

        return feature_list

    def extract_features_tiled(self, image_path: str, grid_size: int) -> list:
        try:
            image = Image.open(image_path).convert("RGB")
            # Ensure the image dimensions are divisible by grid_size for clean tiling
            w, h = image.size
            if w % grid_size != 0 or h % grid_size != 0:
                # Resize to be divisible, while maintaining aspect ratio as much as possible,
                # or simply use the model's expected input size (common practice).
                # For simplicity and correctness with the existing logic, we proceed,
                # but for production, padding/resizing for clean division is better.
                pass

            return self.extract_tile_features(image, grid_size)

        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            return []
        except Exception as e:
            print(f"Error processing image {image_path}: {e}. Skipping.")
            return []


def predict_multi_label_tiled_topk_voting(model, label_encoder, feature_extractor, image_paths, grid_size, k_list):
    """
    Performs multi-label prediction using a top-K voting scheme across N*N tiles.
    """
    all_k_predictions = {k: [] for k in k_list}

    # Pre-calculate features and outputs for all images once
    image_outputs = {}
    print("\n--- 1. Generating features and model outputs for all test images ---")
    for image_path in tqdm(image_paths):
        quadrat_id = os.path.splitext(os.path.basename(image_path))[0]

        feature_list = feature_extractor.extract_features_tiled(image_path, grid_size)

        if not feature_list:
            # Store empty entry to maintain prediction count consistency
            image_outputs[quadrat_id] = None
            continue

        # Combine all features into a single batch tensor
        batch_features = torch.stack(feature_list).to(device)  # Shape: (N*N, 768)

        # Batch Prediction
        with torch.no_grad():
            output = model(batch_features)  # Output shape: (N*N, Num_Classes)
            probabilities = nn.functional.softmax(output, dim=1)  # Shape: (N*N, Num_Classes)
            image_outputs[quadrat_id] = probabilities.cpu()

    print("\n--- 2. Applying Top-K Voting for K in", k_list, "---")

    for k in k_list:
        print(f"Processing K={k}...")
        current_k_predictions = []

        for image_path in tqdm(image_paths):
            quadrat_id = os.path.splitext(os.path.basename(image_path))[0]
            probabilities = image_outputs.get(quadrat_id)

            if probabilities is None:
                continue

            # --- Top-K Prediction per Tile (First part of your request) ---
            # Get the top K class indices for *each* tile.
            # probabilities shape: (N*N, Num_Classes)
            # topk_values, topk_indices shape: (N*N, K)
            topk_values, topk_indices = torch.topk(probabilities, k=k, dim=1)

            # topk_indices contains the K predicted class indices for each of the N*N tiles.
            # Flatten to get a list of all predicted class indices across all tiles.
            # total_votes will be a flat list of size (N*N * K)
            all_predicted_class_indices = topk_indices.flatten().numpy()

            # --- Calculate Frequency and Take Top-K (Second part of your request) ---
            # Calculate frequency of all predicted class indices (the votes)
            vote_counts = Counter(all_predicted_class_indices)

            # Get the top K classes by vote count (frequency)
            # The result is a list of (class_index, count) tuples.
            topk_voted_classes_with_counts = vote_counts.most_common(k)

            # Extract just the class indices
            final_predicted_indices = [item[0] for item in topk_voted_classes_with_counts]

            # 4. Inverse Transform
            if final_predicted_indices:
                # Ensure unique, though `most_common(k)` should ensure uniqueness up to k items
                unique_indices = np.unique(final_predicted_indices)
                predicted_species_ids = label_encoder.inverse_transform(unique_indices)
            else:
                # Fallback, though this should rarely happen if N*N*K > 0
                predicted_species_ids = []

            current_k_predictions.append({
                "quadrat_id": quadrat_id,
                "species_ids": list(predicted_species_ids)
            })

        all_k_predictions[k] = current_k_predictions

    return all_k_predictions


# --- 5. Output Formatting and CSV Generation (Same as before) ---

def format_and_save_csv(predictions, output_file):
    """
    Formats the predictions into the required double-bracket CSV format.
    """
    print(f"\nFormatting and saving results to {output_file}...")

    with open(output_file, 'w', newline='') as csvfile:
        # Use csv.writer with a quoting option to handle the list format
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

        writer.writerow(["quadrat_id", "species_ids"])

        for p in predictions:
            id_list_str = ", ".join(map(str, p['species_ids']))
            # Ensure the output is formatted as `"[1, 5, 10]"`
            formatted_ids = f"[{id_list_str}]"
            writer.writerow([p['quadrat_id'], formatted_ids])

    print(f"Submission file successfully created: {output_file}")


if __name__ == "__main__":
    MODEL_PATH = "/local/scratch1/siam/saved_models/plant_clef/bioclip2_mlp_classifier.pth"
    TEST_IMAGE_DIR = '/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/PlantCLEF2025_test.csv'
    OUTPUT_DIR = "../results/"  # Output directory for all K files

    # Grid Parameters
    GRID_SIZE = 3  # N*N grid (e.g., N=2 -> 4 tiles)
    K_LIST = [5, 10, 20]  # Top-K values for voting

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load the trained model and label encoder
    model, label_encoder = load_model_and_encoder(MODEL_PATH)
    print(f"Model loaded successfully with {label_encoder.classes_.size} classes.")

    # 2. Initialize the Tiled feature extractor
    feature_extractor = TiledBioclipFeatureExtractor()

    # 3. Get list of all test images
    test_data = pd.read_csv(TEST_IMAGE_DIR, sep=';', dtype={'partner': str})
    test_image_path_base = '/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/'

    all_test_files = []
    for f_name in list(test_data['quadrat_id']):
        # Assuming the image file name is the quadrat_id + '.jpg'
        image_path = os.path.join(test_image_path_base, f_name)
        image_path = image_path + '.jpg'
        all_test_files.append(image_path)

    print(f'Total Test Files : {len(all_test_files)}')
    image_paths = all_test_files

    print(f"Found {len(image_paths)} test images to process.")

    # 4. Run prediction using the tiled top-K voting approach for all K
    # The function now handles the loop for K_LIST internally
    all_k_predictions = predict_multi_label_tiled_topk_voting(
        model,
        label_encoder,
        feature_extractor,
        image_paths,
        GRID_SIZE,
        K_LIST
    )

    # 5. Generate output files for each K
    for k, predictions in all_k_predictions.items():
        OUTPUT_CSV_FILE = os.path.join(OUTPUT_DIR, f"tiled_grid_{GRID_SIZE}_TopK_Voting_K{k}.csv")
        format_and_save_csv(predictions, OUTPUT_CSV_FILE)

    print("\nAll prediction files created successfully.")