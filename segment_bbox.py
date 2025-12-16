import os
import cv2
import numpy as np
import pandas as pd
import torch
import json
from tqdm import tqdm
from groundingdino.util.inference import load_model, load_image, predict
from segment_anything import sam_model_registry, SamPredictor

# ---------------- CONFIG ----------------
TEST_IMAGE_DIR = '/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/PlantCLEF2025_test.csv'

# Checkpoints you download manually (place next to this script)
DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
DINO_CKPT   = "/home/siam.5/MyResearch/cv_project/plant-clef-2025/GroundingDINO/weights/groundingdino_swint_ogc.pth"

SAM_TYPE = "vit_h"
SAM_CKPT = "/local/scratch1/siam/pretrain_models/SAM/sam_vit_h_4b8939.pth"

TEXT_PROMPT = "segment all plants"
BOX_THRESHOLD  = 0.03
TEXT_THRESHOLD = 0.1

# Reduce duplicates
BOX_NMS_IOU = 0.50
MASK_DEDUP_IOU = 0.85

# Filter tiny masks
MIN_MASK_AREA = 200  # pixels

OUT_DIR = "/local/scratch1/siam/dataset/plant_clef/test/sam_instances"
# --------------------------------------


def get_all_test_files():
    test_data = pd.read_csv(TEST_IMAGE_DIR, sep=';', dtype={'partner': str})
    test_image_path = '/local/scratch1/siam/dataset/plant_clef/test/data/PlantCLEF/PlantCLEF2025/DataOut/test/package/images/'

    all_test_files = []
    for f_name in list(test_data['quadrat_id']):
        image_path = os.path.join(test_image_path, f_name)
        image_path = image_path + '.jpg'
        all_test_files.append(image_path)

    print(f'Total Test Files : {len(all_test_files)}')
    return all_test_files


def ensure_file(path: str, hint: str = ""):
    if not os.path.exists(path):
        msg = f"Missing file: {path}"
        if hint:
            msg += f"\nHint: {hint}"
        raise FileNotFoundError(msg)


def clamp_box_xyxy(box: np.ndarray, w: int, h: int) -> np.ndarray:
    x1, y1, x2, y2 = box.tolist()
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return np.array([x1, y1, x2, y2], dtype=np.float32)


def box_iou_xyxy(boxes: np.ndarray, ref_box: np.ndarray) -> np.ndarray:
    # boxes: [N,4], ref_box: [4]
    x1 = np.maximum(boxes[:, 0], ref_box[0])
    y1 = np.maximum(boxes[:, 1], ref_box[1])
    x2 = np.minimum(boxes[:, 2], ref_box[2])
    y2 = np.minimum(boxes[:, 3], ref_box[3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_boxes = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    area_ref = max(0, ref_box[2] - ref_box[0]) * max(0, ref_box[3] - ref_box[1])
    union = area_boxes + area_ref - inter + 1e-6
    return inter / union


def nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thr: float) -> list:
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        rest = idxs[1:]
        ious = box_iou_xyxy(boxes[rest], boxes[i])
        idxs = rest[ious < iou_thr]
    return keep


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum() + 1e-6
    return float(inter / union)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("DEVICE:", device)

    # Check files
    ensure_file(DINO_CONFIG, "Fix DINO_CONFIG path.")
    ensure_file(DINO_CKPT, "Download groundingdino_swint_ogc.pth and set DINO_CKPT.")
    ensure_file(SAM_CKPT, "Download SAM checkpoint and set SAM_CKPT.")

    # ---- Load models ----
    dino = load_model(DINO_CONFIG, DINO_CKPT, device=device)

    sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_CKPT).to(device)
    predictor = SamPredictor(sam)

    os.makedirs(OUT_DIR, exist_ok=True)

    all_test_files = get_all_test_files()
    for IMAGE_PATH in tqdm(all_test_files):
        # ---- Load ORIGINAL image (for SAM + saving visuals) ----
        img_bgr = cv2.imread(IMAGE_PATH)
        if img_bgr is None:
            raise FileNotFoundError(f"cv2.imread failed for {IMAGE_PATH}")
        orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # HWC uint8
        H, W = orig_rgb.shape[:2]

        # ---- Load DINO tensor image (for detection only) ----
        _, image_tensor = load_image(IMAGE_PATH)  # torch.Tensor CHW float in [0,1]

        predictor.set_image(orig_rgb)  # SAM uses original RGB
        # ---- Detect boxes with GroundingDINO (torch tensor) ----
        boxes, logits, phrases = predict(
            model=dino,
            image=image_tensor,  # torch.Tensor
            caption=TEXT_PROMPT,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD
        )

        if boxes is None or len(boxes) == 0:
            print("No detections. Try lowering thresholds or changing TEXT_PROMPT.")
            return

        # ---- Convert normalized cxcywh -> pixel xyxy using ORIGINAL W,H ----
        boxes = boxes.detach().cpu().numpy()
        scores = logits.detach().cpu().numpy().astype(np.float32).reshape(-1)

        boxes_xyxy = []
        for (cx, cy, bw, bh) in boxes.tolist():
            x1 = (cx - bw / 2.0) * W
            y1 = (cy - bh / 2.0) * H
            x2 = (cx + bw / 2.0) * W
            y2 = (cy + bh / 2.0) * H
            boxes_xyxy.append([x1, y1, x2, y2])
        boxes_xyxy = np.array(boxes_xyxy, dtype=np.float32)

        # ---- NMS on boxes ----
        keep = nms_xyxy(boxes_xyxy, scores, BOX_NMS_IOU)
        boxes_xyxy = boxes_xyxy[keep]
        scores = scores[keep]

        # ---- Segment each box with SAM ----
        instance_masks = []
        instance_boxes = []
        instance_scores = []

        for box, det_score in zip(boxes_xyxy, scores):
            box = clamp_box_xyxy(box, W, H)

            masks, mask_scores, _ = predictor.predict(
                box=box,
                multimask_output=True
            )

            best = int(np.argmax(mask_scores))
            m = masks[best].astype(bool)

            if m.sum() < MIN_MASK_AREA:
                continue

            # dedup masks
            dup = False
            for ex in instance_masks:
                if mask_iou(m, ex) > MASK_DEDUP_IOU:
                    dup = True
                    break
            if dup:
                continue

            instance_masks.append(m)
            instance_boxes.append(box)
            instance_scores.append(float(det_score))

        # print(f"Kept {len(instance_masks)} instances after NMS+dedup.")

        # ---- Save individual masks + json ----
        results = []
        for i, (m, box, det_score) in enumerate(zip(instance_masks, instance_boxes, instance_scores)):
            # mask_path = os.path.join(OUT_DIR, f"mask_{i:03d}.png")
            # cv2.imwrite(mask_path, (m.astype(np.uint8) * 255))

            x1, y1, x2, y2 = box.tolist()
            results.append({
                "id": i,
                "det_score": det_score,
                "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "mask_area_px": float(m.sum()),
            })

        save_file_name = IMAGE_PATH.split('/')[-1].replace('.jpg', '.json')
        with open(os.path.join(OUT_DIR, save_file_name), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()