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
from model_utils import BioclipFeatureExtractor, SimpleMLP

# --- Configuration ---
MODEL_PATH = "/local/scratch1/siam/saved_models/plant_clef/bioclip2_mlp_classifier.pth"
TEST_IMAGE_DIR = '/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/PlantCLEF2025_test.csv'
OUTPUT_CSV_FILE = "../results/test_entire_image_results.csv"

# Hyperparameter for multi-label prediction
# This threshold is critical and will determine how many labels you predict.
# You will likely need to tune this value on a small validation set.
PREDICTION_THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --- 2. Load Model and Encoder ---

def load_model_and_encoder(model_path):
    """Loads the trained model state and label encoder."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at: {model_path}")

    # --- FIX for WeightsUnpickler error ---
    # 1. Allowlist the LabelEncoder class as safe to load
    # This must be done BEFORE torch.load()
    try:
        from sklearn.preprocessing._label import LabelEncoder as SklearnLabelEncoder

        # Check if the function exists (it might not in older PyTorch versions)
        if hasattr(torch.serialization, 'add_safe_globals'):
            # Add the specific class/function to the allowlist
            torch.serialization.add_safe_globals([SklearnLabelEncoder])
            print("Successfully allowlisted sklearn LabelEncoder for safe loading.")
    except Exception as e:
        print(f"Warning: Could not allowlist LabelEncoder. Error: {e}")
    # -----------------------------------

    # 2. Attempt to load the checkpoint
    try:
        # Default load attempt with weights_only=True
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        # If it still fails, the second safest option is to disable weights_only=False
        # Only do this if the file is from a trusted source (your own training script)
        print("Initial safe load failed. Attempting load with weights_only=False...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Extract components from the checkpoint
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_classes = checkpoint['num_classes']
    label_encoder = checkpoint['label_encoder']  # This is the object that caused the issue

    model = SimpleMLP(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, label_encoder

# --- 4. Prediction Function ---

def predict_multi_label(model, label_encoder, feature_extractor, image_paths):
    """
    Processes images, gets features, predicts probabilities, and applies threshold.
    """
    predictions = []

    for image_path in tqdm(image_paths):
        quadrat_id = os.path.splitext(os.path.basename(image_path))[0]

        # 1. Feature Extraction
        feature_vector = feature_extractor.extract_features(image_path)

        # Convert feature to tensor and move to device
        # feature_tensor = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0).to(device)
        feature_tensor = feature_vector.unsqueeze(0).to(device)

        # 2. Prediction
        with torch.no_grad():
            output = model(feature_tensor)

            # The MLP was trained with CrossEntropyLoss, so the output is logits.
            # Convert logits to probabilities using Softmax.
            probabilities = nn.functional.softmax(output, dim=1).squeeze(0)

            # 3. Apply Thresholding for Multi-Label
            # Find all classes where P > THRESHOLD
            predicted_indices = (probabilities > PREDICTION_THRESHOLD).nonzero(as_tuple=True)[0]

            if len(predicted_indices) == 0:
                # If no probability exceeds the threshold, take the top 1 prediction
                # (This is a common strategy to ensure at least one prediction is made)
                top_index = torch.argmax(probabilities).item()
                predicted_indices = [top_index]
            else:
                predicted_indices = predicted_indices.cpu().numpy()

            # 4. Inverse Transform to original species IDs
            # Use label_encoder to convert numerical indices back to original species_ids
            predicted_species_ids = label_encoder.inverse_transform(predicted_indices)

            predictions.append({
                "quadrat_id": quadrat_id,
                "species_ids": list(predicted_species_ids)
            })

    return predictions


# --- 5. Output Formatting and CSV Generation ---

def format_and_save_csv(predictions, output_file):
    """
    Formats the predictions into the required double-bracket CSV format.
    """
    print(f"\nFormatting and saving results to {output_file}...")

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

        # Write header
        writer.writerow(["quadrat_id", "species_ids"])

        for p in predictions:
            # Format species_ids column: "[123, 456, 789]"
            # Note: We must ensure the list is enclosed in double square brackets [[]]

            # Convert IDs to string and join with ", "
            id_list_str = ", ".join(map(str, p['species_ids']))

            # Enclose the string list in double square brackets
            formatted_ids = f"[{id_list_str}]"

            # Write row: quadrat_id (quoted), formatted_ids (quoted)
            writer.writerow([p['quadrat_id'], formatted_ids])

    print("Submission file successfully created.")


# --- Main Execution ---

if __name__ == "__main__":
    try:
        # Load the trained model and label encoder
        model, label_encoder = load_model_and_encoder(MODEL_PATH)
        print(f"Model loaded successfully with {label_encoder.classes_.size} classes.")

        # Initialize your feature extractor
        feature_extractor = BioclipFeatureExtractor()
        print('Bioclip Loaded...')

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

        # Run prediction
        predictions = predict_multi_label(model, label_encoder, feature_extractor, image_paths)

        # Generate output file
        format_and_save_csv(predictions, OUTPUT_CSV_FILE)

    except FileNotFoundError as e:
        print(f"Execution Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")