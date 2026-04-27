"""Convert LoveDA semantic masks to YOLO instance-seg labels with optional class remapping.

Usage:
    python examples/sam3_loveda/prepare_loveda_instance_labels.py \
        --images /data/LoveDA/Train/images_png \
        --masks /data/LoveDA/Train/masks_png \
        --out /data/LoveDA-Instance \
        --split train
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# LoveDA default semantic id mapping (background is 0 and will be dropped)
DEFAULT_CLASS_MAP = {
    1: 0,  # building
    2: 1,  # road
    3: 2,  # water
    4: 3,  # barren
    5: 4,  # forest (focus class = 5 in semantic mask)
    6: 5,  # agriculture
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare LoveDA YOLO instance labels for SAM3 fine-tuning.")
    parser.add_argument("--images", type=Path, required=True, help="Source image directory (tif/png/jpg).")
    parser.add_argument("--masks", type=Path, required=True, help="Semantic mask directory (single channel).")
    parser.add_argument("--out", type=Path, required=True, help="Output root directory.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--min-area", type=int, default=30, help="Minimum connected-component pixel area to keep.")
    parser.add_argument("--epsilon", type=float, default=1.5, help="Polygon simplification epsilon in pixels.")
    return parser.parse_args()


def find_mask_file(mask_dir: Path, stem: str) -> Path | None:
    for ext in (".png", ".tif", ".tiff", ".jpg"):
        p = mask_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def load_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3 and img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def mask_to_yolo_segments(mask: np.ndarray, min_area: int, epsilon: float) -> list[tuple[int, list[float]]]:
    h, w = mask.shape
    segments: list[tuple[int, list[float]]] = []

    for src_id, yolo_id in DEFAULT_CLASS_MAP.items():
        binary = (mask == src_id).astype(np.uint8)
        if not binary.any():
            continue

        n, cc, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for comp_id in range(1, n):
            area = int(stats[comp_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            comp_mask = (cc == comp_id).astype(np.uint8)
            contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if len(contour) < 3:
                    continue
                approx = cv2.approxPolyDP(contour, epsilon=epsilon, closed=True)
                if len(approx) < 3:
                    continue
                points = approx.reshape(-1, 2).astype(np.float32)
                points[:, 0] = np.clip(points[:, 0] / w, 0.0, 1.0)
                points[:, 1] = np.clip(points[:, 1] / h, 0.0, 1.0)
                segments.append((yolo_id, points.reshape(-1).tolist()))

    return segments


def main() -> None:
    args = parse_args()
    image_out = args.out / "images" / args.split
    label_out = args.out / "labels" / args.split
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    image_files = [p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}]
    image_files.sort()

    converted, skipped = 0, 0
    for img_path in image_files:
        mask_path = find_mask_file(args.masks, img_path.stem)
        if mask_path is None:
            skipped += 1
            continue

        img = load_image(img_path)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            skipped += 1
            continue
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # save image to PNG for stable downstream loading
        dst_img_path = image_out / f"{img_path.stem}.png"
        cv2.imwrite(str(dst_img_path), img)

        segments = mask_to_yolo_segments(mask, min_area=args.min_area, epsilon=args.epsilon)
        label_path = label_out / f"{img_path.stem}.txt"
        with label_path.open("w", encoding="utf-8") as f:
            for cls_id, coords in segments:
                f.write(f"{cls_id} " + " ".join(f"{v:.6f}" for v in coords) + "\n")

        converted += 1

    print(f"Done. converted={converted}, skipped_no_mask={skipped}, out={args.out}")


if __name__ == "__main__":
    main()
