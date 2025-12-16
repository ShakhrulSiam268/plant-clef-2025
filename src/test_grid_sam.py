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
from model_utils import SimpleMLP, SimpleMLP_2


# ------------------------
# 1) Model / Encoder loading
# ------------------------

def load_model_and_encoder(model_path, device):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Saved model not found at: {model_path}")

    # Allowlist sklearn LabelEncoder for torch.load safety mode
    try:
        from sklearn.preprocessing._label import LabelEncoder as SklearnLabelEncoder
        if hasattr(torch.serialization, "add_safe_globals"):
            torch.serialization.add_safe_globals([SklearnLabelEncoder])
    except Exception as e:
        print(f"Warning: Could not allowlist LabelEncoder. Error: {e}")

    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception:
        print("Initial safe load failed. Attempting load with weights_only=False...")
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    input_size = checkpoint["input_size"]
    hidden_size = checkpoint["hidden_size"]
    num_classes = checkpoint["num_classes"]
    label_encoder = checkpoint["label_encoder"]

    model = SimpleMLP_2(input_size, hidden_size, num_classes).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, label_encoder


# ------------------------
# 2) BBox JSON utilities
# ------------------------

def load_bboxes_from_json(json_path: str) -> List[Tuple[float, float, float, float]]:
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
    x1, y1, x2, y2 = bbox_xyxy

    left = min(x1, x2)
    right = max(x1, x2)
    upper = min(y1, y2)
    lower = max(y1, y2)

    left = int(math.floor(left))
    upper = int(math.floor(upper))
    right = int(math.ceil(right))
    lower = int(math.ceil(lower))

    left = max(0, min(left, image_w - 1))
    upper = max(0, min(upper, image_h - 1))
    right = max(1, min(right, image_w))
    lower = max(1, min(lower, image_h))

    if (right - left) < min_size or (lower - upper) < min_size:
        return None
    return (left, upper, right, lower)


# ------------------------
# 3) Unified feature extractor
#    - bbox crops (from JSON)
#    - multi-scale grid tiles
# ------------------------

class UnifiedBioclipFeatureExtractor:
    def __init__(self, device: torch.device):
        self.device = device
        self.model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            "hf-hub:imageomics/bioclip-2"
        )
        self.model.eval()
        self.preprocess = preprocess_train
        self.model.to(self.device)
        print("Bioclip-2 Model Loaded...")

    def _encode_crop(self, crop: Image.Image) -> torch.Tensor:
        x = self.preprocess(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(x)
            feat /= feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).cpu()

    # ---- BBOX crops ----
    def extract_bbox_features(
        self,
        image: Image.Image,
        json_path: str,
        min_box_size: int = 2
    ) -> List[torch.Tensor]:
        bboxes = load_bboxes_from_json(json_path)
        if not bboxes:
            return []

        w, h = image.size
        feats = []
        for bbox in bboxes:
            crop_box = clamp_and_validate_bbox(bbox, w, h, min_size=min_box_size)
            if crop_box is None:
                continue
            crop = image.crop(crop_box)
            feats.append(self._encode_crop(crop))
        return feats

    # ---- Multi-scale grid tiles ----
    def extract_tile_features(self, image: Image.Image, grid_size: int) -> List[torch.Tensor]:
        w, h = image.size
        tile_w = w // grid_size
        tile_h = h // grid_size

        feats = []
        for i in range(grid_size):
            for j in range(grid_size):
                left = j * tile_w
                upper = i * tile_h
                # include remainder pixels in last row/col
                right = (j + 1) * tile_w if j < grid_size - 1 else w
                lower = (i + 1) * tile_h if i < grid_size - 1 else h
                tile = image.crop((left, upper, right, lower))
                feats.append(self._encode_crop(tile))
        return feats

    def extract_multi_scale_features(self, image: Image.Image, grid_sizes: List[int]) -> List[torch.Tensor]:
        all_feats = []
        for gs in grid_sizes:
            all_feats.extend(self.extract_tile_features(image, gs))
        return all_feats

    # ---- Combined in the exact order you asked:
    #      bbox crops first, then grid tiles (multi-scale)
    def extract_features_bbox_then_grid(
        self,
        image_path: str,
        json_path: str,
        grid_sizes: List[int],
        min_box_size: int = 2
    ) -> List[torch.Tensor]:
        try:
            image = Image.open(image_path).convert("RGB")
            image_copy_1 = image.copy()
            image_copy_2 = image.copy()
        except FileNotFoundError:
            print(f"Warning: Image file not found at {image_path}. Skipping.")
            return []
        except Exception as e:
            print(f"Error opening image {image_path}: {e}. Skipping.")
            return []

        feats = []
        # 1) bbox crops (JSON)
        feats.extend(self.extract_bbox_features(image_copy_1, json_path, min_box_size=min_box_size))
        # 2) multi-scale grid tiles
        feats.extend(self.extract_multi_scale_features(image_copy_2, grid_sizes))
        return feats


# ------------------------
# 4) Run once: store raw probabilities per image
# ------------------------

def collect_raw_predictions_combined(
    model,
    feature_extractor: UnifiedBioclipFeatureExtractor,
    image_paths: List[str],
    bbox_json_dir: str,
    grid_sizes: List[int],
    device: torch.device,
    min_box_size: int = 2,
):
    all_raw_predictions = []

    for image_path in tqdm(image_paths, desc="Extracting bbox+grid features & predicting"):
        quadrat_id = os.path.splitext(os.path.basename(image_path))[0]
        json_path = os.path.join(bbox_json_dir, f"{quadrat_id}.json")

        # If you want to skip images that have no JSON, uncomment this:
        # if not os.path.exists(json_path):
        #     continue

        feature_list = feature_extractor.extract_features_bbox_then_grid(
            image_path=image_path,
            json_path=json_path,
            grid_sizes=grid_sizes,
            min_box_size=min_box_size
        )

        if not feature_list:
            continue

        batch_features = torch.stack(feature_list).to(device)  # (K, 768)

        with torch.no_grad():
            logits = model(batch_features)                  # (K, C)
            probs = nn.functional.softmax(logits, dim=1)    # (K, C)

        all_raw_predictions.append({
            "quadrat_id": quadrat_id,
            "probabilities": probs.cpu()
        })

    return all_raw_predictions


# ------------------------
# 5) Threshold sweep + CSV writing (same format as code-1)
# ------------------------

def format_and_save_csv(predictions, output_file):
    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
        writer.writerow(["quadrat_id", "species_ids"])
        for p in predictions:
            id_list_str = ", ".join(map(str, p["species_ids"]))
            writer.writerow([p["quadrat_id"], f"[{id_list_str}]"])
    print(f"Saved submission file: {output_file}")


def tune_and_save_predictions(all_raw_predictions, label_encoder, threshold_range, output_dir, file_prefix):
    print("\n--- Starting Threshold Tuning and CSV Generation ---")

    os.makedirs(output_dir, exist_ok=True)

    for threshold in threshold_range:
        predictions = []
        print(f"Processing predictions for THRESHOLD: {threshold:.1f}")

        for p_data in tqdm(all_raw_predictions, desc=f"Applying threshold {threshold:.1f}"):
            quadrat_id = p_data["quadrat_id"]
            probabilities = p_data["probabilities"]  # (K, C)

            predicted_indices = (probabilities > threshold).nonzero(as_tuple=False)

            all_predicted_class_indices = []
            if predicted_indices.numel() > 0:
                all_predicted_class_indices.extend(predicted_indices[:, 1].cpu().numpy().tolist())

            # fallback: best single class across all crops
            if not all_predicted_class_indices:
                _, max_index_flat = torch.max(probabilities.flatten(), 0)
                max_class_index = (max_index_flat % probabilities.size(1)).item()
                all_predicted_class_indices.append(max_class_index)

            unique_indices = np.unique(all_predicted_class_indices)
            predicted_species_ids = label_encoder.inverse_transform(unique_indices)

            predictions.append({
                "quadrat_id": quadrat_id,
                "species_ids": list(predicted_species_ids)
            })

        output_file = os.path.join(output_dir, f"{file_prefix}_Threshold_{threshold:.1f}.csv")
        format_and_save_csv(predictions, output_file)

    print("\n--- Threshold Tuning Complete ---")


# ------------------------
# Main
# ------------------------

if __name__ == "__main__":
    MODEL_PATH = "/local/scratch1/siam/saved_models/plant_clef/bioclip2_mlp_classifier_big_2048.pth"

    TEST_CSV_PATH = "/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/PlantCLEF2025_test.csv"
    TEST_IMAGE_BASE_DIR = "/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/"

    # JSONs named <quadrat_id>.json
    BBOX_JSON_DIR = "/local/scratch1/siam/dataset/plant_clef/test/sam_instances"

    # multi-scale tiling (same as code-1)
    GRID_SIZES = [1, 2, 4, 8]
    THRESHOLD_RANGE = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    MIN_BOX_SIZE = 2

    OUTPUT_BASE_DIR = "../results/results_bbox_plus_multi_scale"
    OUTPUT_FILE_PREFIX = "bbox_plus_multi_scale"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model, label_encoder = load_model_and_encoder(MODEL_PATH, device)
    print(f"Model loaded successfully with {label_encoder.classes_.size} classes.")

    feature_extractor = UnifiedBioclipFeatureExtractor(device=device)

    test_data = pd.read_csv(TEST_CSV_PATH, sep=";", dtype={"partner": str})
    image_paths = [os.path.join(TEST_IMAGE_BASE_DIR, f"{qid}.jpg") for qid in list(test_data["quadrat_id"])]

    print(f"Total Test Files: {len(image_paths)}")

    # 1) Collect raw probabilities once (bbox crops + multi-scale tiles)
    all_raw_predictions = collect_raw_predictions_combined(
        model=model,
        feature_extractor=feature_extractor,
        image_paths=image_paths,
        bbox_json_dir=BBOX_JSON_DIR,
        grid_sizes=GRID_SIZES,
        device=device,
        min_box_size=MIN_BOX_SIZE,
    )

    # 2) Sweep thresholds and write CSVs
    tune_and_save_predictions(
        all_raw_predictions=all_raw_predictions,
        label_encoder=label_encoder,
        threshold_range=THRESHOLD_RANGE,
        output_dir=OUTPUT_BASE_DIR,
        file_prefix=OUTPUT_FILE_PREFIX
    )
