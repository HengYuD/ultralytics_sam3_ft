"""Test/evaluate SAM3 LoveDA checkpoint on selected semantic classes.

Features:
- Supports png/jpg/tif/tiff input images.
- Supports multi-channel TIF input (first 3 bands used for model input, robust percentile stretch).
- Preserves geospatial metadata for TIF outputs and writes TIF prediction masks aligned to source image.
- Saves side-by-side visualization (original vs overlay result).

Example:
python examples/sam3_loveda/test_sam3_loveda.py \
  --sam3-ckpt /models/sam3_b.pt \
  --weights runs/sam3_loveda/best.pt \
  --images /data/LoveDA/Val/images_tif \
  --masks /data/LoveDA/Val/masks_png \
  --save-vis --save-geotiff
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ultralytics.models.sam.build_sam3 import build_sam3_image_model

try:
    import rasterio
except Exception:  # pragma: no cover - optional runtime dependency
    rasterio = None


@dataclass
class GeoInfo:
    transform: object | None = None
    crs: object | None = None
    width: int | None = None
    height: int | None = None


@dataclass
class Sample:
    image: torch.Tensor
    masks: torch.Tensor
    stem: str
    image_path: str
    orig_hw: tuple[int, int]
    viz_bgr: np.ndarray
    geoinfo: GeoInfo


def _percentile_to_uint8(arr: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    arr = arr.astype(np.float32)
    low = np.percentile(arr, p_low)
    high = np.percentile(arr, p_high)
    if high <= low:
        high = low + 1.0
    arr = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return (arr * 255.0).astype(np.uint8)


def _read_tif(path: Path) -> tuple[np.ndarray, np.ndarray, GeoInfo]:
    if rasterio is None:
        raise ImportError("rasterio is required for tif/tiff input and geoinfo-preserving output.")

    with rasterio.open(path) as src:
        data = src.read()  # (C,H,W)
        geoinfo = GeoInfo(transform=src.transform, crs=src.crs, width=src.width, height=src.height)

    c, h, w = data.shape
    if c == 1:
        rgb = np.repeat(data[:1], 3, axis=0)
    elif c >= 3:
        rgb = data[:3]
    else:
        pad = np.zeros((3 - c, h, w), dtype=data.dtype)
        rgb = np.concatenate([data, pad], axis=0)

    rgb_uint8 = np.stack([_percentile_to_uint8(rgb[i]) for i in range(3)], axis=0)  # (3,H,W)
    rgb_hwc = np.transpose(rgb_uint8, (1, 2, 0))  # RGB
    bgr_vis = rgb_hwc[:, :, ::-1].copy()  # BGR for cv2
    return rgb_hwc, bgr_vis, geoinfo


def _read_non_tif(path: Path) -> tuple[np.ndarray, np.ndarray, GeoInfo]:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)

    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.ndim == 3 and im.shape[2] > 3:
        im = im[:, :, :3]

    rgb = im[:, :, ::-1].copy()  # BGR->RGB
    return rgb, im, GeoInfo()


class LoveDAMultiClassDataset(Dataset):
    def __init__(self, image_dir: Path, mask_dir: Path, semantic_ids: list[int], imgsz: int = 1024):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.semantic_ids = semantic_ids
        self.imgsz = imgsz
        self.files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])

    def __len__(self) -> int:
        return len(self.files)

    def _read_mask(self, stem: str, out_hw: tuple[int, int]) -> np.ndarray:
        for ext in (".png", ".tif", ".tiff", ".jpg"):
            p = self.mask_dir / f"{stem}{ext}"
            if p.exists():
                if p.suffix.lower() in {".tif", ".tiff"}:
                    if rasterio is None:
                        raise ImportError("rasterio is required to read tif/tiff masks.")
                    with rasterio.open(p) as src:
                        m = src.read(1)
                else:
                    m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                    if m is None:
                        continue
                    if m.ndim == 3:
                        m = m[:, :, 0]
                m = cv2.resize(m, (out_hw[1], out_hw[0]), interpolation=cv2.INTER_NEAREST)
                return m
        raise FileNotFoundError(f"Mask not found for stem={stem}")

    def __getitem__(self, idx: int) -> Sample:
        p = self.files[idx]
        if p.suffix.lower() in {".tif", ".tiff"}:
            rgb_hwc, viz_bgr, geoinfo = _read_tif(p)
        else:
            rgb_hwc, viz_bgr, geoinfo = _read_non_tif(p)

        orig_h, orig_w = rgb_hwc.shape[:2]

        input_rgb = cv2.resize(rgb_hwc, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        raw_mask = self._read_mask(p.stem, out_hw=(self.imgsz, self.imgsz))

        image_t = torch.from_numpy(input_rgb.copy()).permute(2, 0, 1).float() / 255.0
        masks = [torch.from_numpy((raw_mask == sid).astype(np.float32)) for sid in self.semantic_ids]
        masks_t = torch.stack(masks, dim=0)

        return Sample(
            image=image_t,
            masks=masks_t,
            stem=p.stem,
            image_path=str(p),
            orig_hw=(orig_h, orig_w),
            viz_bgr=viz_bgr,
            geoinfo=geoinfo,
        )


def collate_fn(batch: list[Sample]) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[tuple[int, int]], list[np.ndarray], list[GeoInfo]]:
    images = torch.stack([b.image for b in batch], dim=0)
    masks = torch.stack([b.masks for b in batch], dim=0)
    stems = [b.stem for b in batch]
    image_paths = [b.image_path for b in batch]
    orig_hws = [b.orig_hw for b in batch]
    viz_bgrs = [b.viz_bgr for b in batch]
    geoinfos = [b.geoinfo for b in batch]
    return images, masks, stems, image_paths, orig_hws, viz_bgrs, geoinfos


def parse_csv_ints(v: str) -> list[int]:
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def parse_csv_strs(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate SAM3 LoveDA checkpoint.")
    p.add_argument("--sam3-ckpt", type=str, required=True, help="Base SAM3 checkpoint path, e.g. sam3_b.pt")
    p.add_argument("--weights", type=Path, required=True, help="Fine-tuned best.pt")
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--masks", type=Path, required=True)
    p.add_argument("--focus-class-names", type=str, default="", help="Optional override: comma separated class names.")
    p.add_argument("--focus-semantic-ids", type=str, default="", help="Optional override: comma separated semantic ids.")
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save-vis", action="store_true", help="Save visualization comparisons.")
    p.add_argument("--vis-dir", type=Path, default=Path("runs/sam3_loveda/vis_compare"))
    p.add_argument("--save-geotiff", action="store_true", help="When source is tif/tiff, save georeferenced tif predictions.")
    p.add_argument("--tif-out-dir", type=Path, default=Path("runs/sam3_loveda/tif_preds"))
    p.add_argument("--out-json", type=Path, default=Path("runs/sam3_loveda/test_metrics.json"))
    return p.parse_args()


def select_logit_for_class(model, images: torch.Tensor, class_idx: int) -> torch.Tensor:
    backbone_out = model.backbone(images)
    text_ids = torch.full((images.shape[0],), class_idx, dtype=torch.long, device=images.device)
    out = model.forward_grounding(backbone_out, text_ids=text_ids)
    return out["pred_masks"][:, :1]


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    color_layer = np.zeros_like(image_bgr)
    color_layer[mask > 0] = color
    return cv2.addWeighted(image_bgr, 0.7, color_layer, 0.3, 0)


def save_geo_tif(mask: np.ndarray, geoinfo: GeoInfo, out_path: Path) -> None:
    if rasterio is None:
        raise ImportError("rasterio is required to save georeferenced tif outputs.")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "count": 1,
        "dtype": "uint8",
        "transform": geoinfo.transform,
        "crs": geoinfo.crs,
        "compress": "lzw",
    }

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(mask.astype(np.uint8), 1)


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.weights, map_location="cpu")
    class_names = ckpt.get("class_names", ["forest"])
    semantic_ids = ckpt.get("semantic_ids", [5])
    imgsz = ckpt.get("imgsz", args.imgsz)

    if args.focus_class_names:
        class_names = parse_csv_strs(args.focus_class_names)
    if args.focus_semantic_ids:
        semantic_ids = parse_csv_ints(args.focus_semantic_ids)

    if len(class_names) != len(semantic_ids):
        raise ValueError("class names and semantic ids length mismatch")

    dataset = LoveDAMultiClassDataset(args.images, args.masks, semantic_ids, imgsz)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_sam3_image_model(args.sam3_ckpt)
    model.set_classes(class_names)
    model.set_imgsz((imgsz, imgsz))
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()

    cls_inter = {c: 0.0 for c in class_names}
    cls_union = {c: 0.0 for c in class_names}
    cls_dice_num = {c: 0.0 for c in class_names}
    cls_dice_den = {c: 0.0 for c in class_names}

    if args.save_vis:
        args.vis_dir.mkdir(parents=True, exist_ok=True)
    if args.save_geotiff:
        args.tif_out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for images, targets, stems, image_paths, orig_hws, viz_bgrs, geoinfos in loader:
            images = images.to(device)
            targets = targets.to(device)

            # class-wise logits (at network input size)
            class_preds_resized: list[torch.Tensor] = []
            for ci, cname in enumerate(class_names):
                logits = select_logit_for_class(model, images, ci)
                logits = F.interpolate(logits, size=targets[:, ci : ci + 1].shape[-2:], mode="bilinear", align_corners=False)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                class_preds_resized.append(preds)

                gts = targets[:, ci : ci + 1]
                inter = (preds * gts).sum().item()
                union = (preds + gts - preds * gts).sum().item()
                cls_inter[cname] += inter
                cls_union[cname] += max(union, 1.0)
                cls_dice_num[cname] += 2 * inter
                cls_dice_den[cname] += (preds.sum() + gts.sum()).item() + 1e-6

            # per image outputs
            for bi, stem in enumerate(stems):
                orig_h, orig_w = orig_hws[bi]
                src_path = Path(image_paths[bi])
                src_suffix = src_path.suffix.lower()
                viz_bgr = viz_bgrs[bi]
                geoinfo = geoinfos[bi]

                # merge multi-class binary masks to one label map (0=bg, 1..C=class index+1)
                pred_label = np.zeros((imgsz, imgsz), dtype=np.uint8)
                for ci, pred_tensor in enumerate(class_preds_resized):
                    pred = pred_tensor[bi, 0].detach().cpu().numpy().astype(np.uint8)
                    pred_label[pred > 0] = ci + 1

                # resize prediction to source size
                pred_label_orig = cv2.resize(pred_label, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

                # save georeferenced tif prediction if source is tif/tiff
                if args.save_geotiff and src_suffix in {".tif", ".tiff"}:
                    tif_out = args.tif_out_dir / f"{stem}_pred.tif"
                    save_geo_tif(pred_label_orig, geoinfo, tif_out)

                # save side-by-side visualization comparison
                if args.save_vis:
                    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
                    palette = [
                        (0, 0, 0),
                        (0, 255, 0),
                        (0, 165, 255),
                        (255, 0, 0),
                        (255, 255, 0),
                        (255, 0, 255),
                        (0, 255, 255),
                    ]
                    for ci in range(len(class_names)):
                        color_mask[pred_label_orig == (ci + 1)] = palette[(ci + 1) % len(palette)]

                    overlay = cv2.addWeighted(viz_bgr, 0.65, color_mask, 0.35, 0)
                    compare = np.concatenate([viz_bgr, overlay], axis=1)
                    cv2.imwrite(str(args.vis_dir / f"{stem}_compare.jpg"), compare)

                    # keep same-format result output for source type
                    if src_suffix in {".png", ".jpg", ".jpeg"}:
                        out_ext = ".png" if src_suffix == ".png" else ".jpg"
                        cv2.imwrite(str(args.vis_dir / f"{stem}_pred{out_ext}"), pred_label_orig)
                    elif src_suffix in {".tif", ".tiff"}:
                        # additionally save non-georeferenced quick-look tif mask
                        cv2.imwrite(str(args.vis_dir / f"{stem}_pred.tif"), pred_label_orig)

    per_class = {}
    for cname in class_names:
        iou = cls_inter[cname] / max(cls_union[cname], 1.0)
        dice = cls_dice_num[cname] / max(cls_dice_den[cname], 1e-6)
        per_class[cname] = {"iou": iou, "dice": dice}

    mean_iou = float(np.mean([v["iou"] for v in per_class.values()]))
    mean_dice = float(np.mean([v["dice"] for v in per_class.values()]))
    out = {"per_class": per_class, "mean_iou": mean_iou, "mean_dice": mean_dice}

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Saved metrics to {args.out_json}")


if __name__ == "__main__":
    main()
