import os
import json
import csv
import math
from typing import List, Tuple, Optional

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.serialization
import open_clip
from tqdm import tqdm

from model_utils import SimpleMLP


# ------------------------
# Model / Encoder loading
# ------------------------

def load_model_and_encoder(model_path, device):
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
    except Exception:
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


# ------------------------
# BBox JSON utilities
# ------------------------

def load_bboxes_from_json(json_path: str) -> List[Tuple[float, float, float, float]]:
    """
    Reads a JSON file that contains a list of objects with "bbox_xyxy": [x1,y1,x2,y2].
    Returns list of (x1,y1,x2,y2) floats.
    """
    if not os.path.exists(json_path):
        return []

    try:
        with open(json_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read/parse json {json_path}: {e}")
        return []

    bboxes = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_xyxy", None)
            if bbox is None or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, bbox)
                bboxes.append((x1, y1, x2, y2))
            except Exception:
                continue
    return bboxes


def clamp_and_validate_bbox(
    bbox_xyxy: Tuple[float, float, float, float],
    image_w: int,
    image_h: int,
    min_size: int = 2
) -> Optional[Tuple[int, int, int, int]]:
    """
    Converts bbox to int pixel coords, clamps to image bounds, and validates area.
    Returns (left, upper, right, lower) or None if invalid/tiny.
    """
    x1, y1, x2, y2 = bbox_xyxy

    # Ensure x1<x2, y1<y2
    left = min(x1, x2)
    right = max(x1, x2)
    upper = min(y1, y2)
    lower = max(y1, y2)

    # Floor/ceil to include full region
    left = int(math.floor(left))
    upper = int(math.floor(upper))
    right = int(math.ceil(right))
    lower = int(math.ceil(lower))

    # Clamp to image bounds
    left = max(0, min(left, image_w - 1))
    upper = max(0, min(upper, image_h - 1))
    right = max(1, min(right, image_w))
    lower = max(1, min(lower, image_h))

    # Validate size
    if (right - left) < min_size or (lower - upper) < min_size:
        return None

    return (left, upper, right, lower)


# ------------------------
# Feature extractor (bbox crops)
# ------------------------

class BBoxBioclipFeatureExtractor:
    def __init__(self, device: torch.device):
        self.model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            'hf-hub:imageomics/bioclip-2'
        )
        self.model.eval()
        self.preprocess = preprocess_train
        self.device = device
        self.model.to(self.device)
        print('Bioclip-2 Model Loaded...')

    def extract_bbox_features(
        self,
        image: Image.Image,
        bboxes_xyxy: List[Tuple[float, float, float, float]],
        min_box_size: int = 2
    ) -> List[torch.Tensor]:
        """
        Crops each bbox from the image, preprocesses, encodes with Bioclip2.
        Returns list of 1D feature tensors (on CPU).
        """
        w, h = image.size
        feats = []

        for bbox in bboxes_xyxy:
            crop_box = clamp_and_validate_bbox(bbox, w, h, min_size=min_box_size)
            if crop_box is None:
                continue

            crop = image.crop(crop_box)
            image_input = self.preprocess(crop).unsqueeze(0).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                feats.append(image_features.squeeze(0).cpu())

        return feats

    def extract_features_from_paths(
        self,
        image_path: str,
        json_path: str,
        min_box_size: int = 2
    ) -> List[torch.Tensor]:
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            return []
        except Exception as e:
            print(f"Error opening image {image_path}: {e}. Skipping.")
            return []

        bboxes = load_bboxes_from_json(json_path)
        if not bboxes:
            # No bboxes found (missing json or empty) -> return empty so caller can fallback/skip
            return []

        return self.extract_bbox_features(image, bboxes, min_box_size=min_box_size)


# ------------------------
# Prediction (multi-label over crops)
# ------------------------

def predict_multi_label_bboxes(
    model,
    label_encoder,
    feature_extractor: BBoxBioclipFeatureExtractor,
    image_paths: List[str],
    bbox_json_dir: str,
    prediction_threshold: float,
    min_box_size: int = 2,
    skip_if_no_json: bool = True
):
    """
    For each image:
      - reads <bbox_json_dir>/<quadrat_id>.json
      - crops all bboxes
      - encodes each crop -> feature
      - runs MLP on batch of crop features
      - multi-label threshold across all crops (same logic you used for tiles)
    """
    predictions = []

    for image_path in tqdm(image_paths):
        quadrat_id = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(bbox_json_dir, f"{quadrat_id}.json")

        feature_list = feature_extractor.extract_features_from_paths(
            image_path=image_path,
            json_path=json_path,
            min_box_size=min_box_size
        )

        if not feature_list:
            if not os.path.exists(json_path) and skip_if_no_json:
                # no bbox file -> skip silently (or print if you want)
                continue
            # If JSON exists but produced no valid crops, skip
            continue

        batch_features = torch.stack(feature_list).to(feature_extractor.device)  # (K, 768)

        with torch.no_grad():
            output = model(batch_features)  # (K, Num_Classes)
            probabilities = nn.functional.softmax(output, dim=1)  # (K, Num_Classes)

            predicted_indices = (probabilities > prediction_threshold).nonzero(as_tuple=False)
            all_predicted_class_indices = []

            if predicted_indices.numel() > 0:
                class_indices = predicted_indices[:, 1].cpu().numpy()
                all_predicted_class_indices.extend(class_indices)

            # Fallback: take the single most confident prediction across all crops/classes
            if not all_predicted_class_indices:
                max_prob, max_index_flat = torch.max(probabilities.flatten(), 0)
                max_class_index = max_index_flat % probabilities.size(1)
                all_predicted_class_indices.append(max_class_index.item())

            unique_indices = np.unique(all_predicted_class_indices)
            predicted_species_ids = label_encoder.inverse_transform(unique_indices)

            predictions.append({
                "quadrat_id": quadrat_id,
                "species_ids": list(predicted_species_ids)
            })

    return predictions


# ------------------------
# CSV output (unchanged)
# ------------------------

def format_and_save_csv(predictions, output_file):
    print(f"\nFormatting and saving results to {output_file}...")

    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(["quadrat_id", "species_ids"])

        for p in predictions:
            id_list_str = ", ".join(map(str, p['species_ids']))
            formatted_ids = f"[{id_list_str}]"
            writer.writerow([p['quadrat_id'], formatted_ids])

    print("Submission file successfully created.")


# ------------------------
# Main
# ------------------------

if __name__ == "__main__":
    MODEL_PATH = "/local/scratch1/siam/saved_models/plant_clef/bioclip2_mlp_classifier.pth"

    TEST_CSV = "/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/PlantCLEF2025_test.csv"
    TEST_IMAGE_DIR = "/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/"

    # NEW: directory containing JSONs named <quadrat_id>.json
    BBOX_JSON_DIR = "/local/scratch1/siam/dataset/plant_clef/test/sam_instances"  # <-- change me

    PREDICTION_THRESHOLD = 0.5
    MIN_BOX_SIZE = 2

    OUTPUT_CSV_FILE = f"../results/SAM_bbox_crops_Thresold_{PREDICTION_THRESHOLD}.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, label_encoder = load_model_and_encoder(MODEL_PATH, device=device)
    print(f"Model loaded successfully with {label_encoder.classes_.size} classes.")

    feature_extractor = BBoxBioclipFeatureExtractor(device=device)

    test_data = pd.read_csv(TEST_CSV, sep=';', dtype={'partner': str})

    image_paths = []
    for f_name in list(test_data['quadrat_id']):
        image_paths.append(os.path.join(TEST_IMAGE_DIR, f"{f_name}.jpg"))

    print(f"Found {len(image_paths)} test images to process.")

    predictions = predict_multi_label_bboxes(
        model=model,
        label_encoder=label_encoder,
        feature_extractor=feature_extractor,
        image_paths=image_paths,
        bbox_json_dir=BBOX_JSON_DIR,
        prediction_threshold=PREDICTION_THRESHOLD,
        min_box_size=MIN_BOX_SIZE,
        skip_if_no_json=True
    )

    format_and_save_csv(predictions, OUTPUT_CSV_FILE)
