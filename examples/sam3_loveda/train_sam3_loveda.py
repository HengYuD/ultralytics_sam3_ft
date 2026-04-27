"""SAM3 fine-tuning script for LoveDA remote sensing TIFF images.

Supports single-class or multi-class focused fine-tuning with text prompts.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ultralytics.models.sam.build_sam3 import build_sam3_image_model


@dataclass
class Sample:
    image: torch.Tensor
    masks: torch.Tensor  # (C, H, W), one channel per selected class
    stem: str


def parse_csv_ints(v: str) -> list[int]:
    return [int(x.strip()) for x in v.split(",") if x.strip()]


def parse_csv_strs(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


class LoveDAMultiClassDataset(Dataset):
    def __init__(
        self,
        image_dir: Path,
        mask_dir: Path,
        semantic_ids: list[int],
        imgsz: int = 1024,
        augment: bool = True,
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.semantic_ids = semantic_ids
        self.imgsz = imgsz
        self.augment = augment
        self.files = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}])

    def __len__(self) -> int:
        return len(self.files)

    def _read_image(self, path: Path) -> np.ndarray:
        im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if im is None:
            raise FileNotFoundError(path)
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        elif im.ndim == 3 and im.shape[2] > 3:
            im = im[:, :, :3]
        return im

    def _read_mask(self, stem: str) -> np.ndarray:
        for ext in (".png", ".tif", ".tiff", ".jpg"):
            p = self.mask_dir / f"{stem}{ext}"
            if p.exists():
                m = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if m is None:
                    continue
                if m.ndim == 3:
                    m = m[:, :, 0]
                return m
        raise FileNotFoundError(f"Mask not found for stem={stem}")

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.flip(image, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if random.random() < 0.5:
            image = np.flip(image, axis=0).copy()
            mask = np.flip(mask, axis=0).copy()
        if random.random() < 0.25:
            k = random.choice([1, 2, 3])
            image = np.rot90(image, k=k).copy()
            mask = np.rot90(mask, k=k).copy()
        if random.random() < 0.3:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-12, 12)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return image, mask

    def __getitem__(self, idx: int) -> Sample:
        p = self.files[idx]
        image = self._read_image(p)
        raw_mask = self._read_mask(p.stem)

        if self.augment:
            image, raw_mask = self._augment(image, raw_mask)

        image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        raw_mask = cv2.resize(raw_mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)

        image_t = torch.from_numpy(image[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0
        masks = [torch.from_numpy((raw_mask == sid).astype(np.float32)) for sid in self.semantic_ids]
        masks_t = torch.stack(masks, dim=0)
        return Sample(image=image_t, masks=masks_t, stem=p.stem)


def collate_fn(batch: list[Sample]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    images = torch.stack([b.image for b in batch], dim=0)
    masks = torch.stack([b.masks for b in batch], dim=0)
    stems = [b.stem for b in batch]
    return images, masks, stems


def dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    num = 2 * (prob * target).sum(dim=(1, 2, 3))
    den = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def forward_class_logits(model: nn.Module, images: torch.Tensor, class_idx: int) -> torch.Tensor:
    backbone_out = model.backbone(images)
    text_ids = torch.full((images.shape[0],), class_idx, dtype=torch.long, device=images.device)
    out = model.forward_grounding(backbone_out, text_ids=text_ids)
    logits = out["pred_masks"][:, :1]
    return logits


def compute_loss_and_metrics(model: nn.Module, images: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, list[float]]:
    # targets: (B, C, H, W)
    cls_losses = []
    cls_ious = []
    for class_idx in range(targets.shape[1]):
        target = targets[:, class_idx : class_idx + 1]
        pred_logits = forward_class_logits(model, images, class_idx)
        pred_logits = F.interpolate(pred_logits, size=target.shape[-2:], mode="bilinear", align_corners=False)
        cls_loss = F.binary_cross_entropy_with_logits(pred_logits, target) + dice_loss(pred_logits, target)
        cls_losses.append(cls_loss)

        pred_bin = (torch.sigmoid(pred_logits) > 0.5).float()
        inter = (pred_bin * target).sum(dim=(1, 2, 3))
        union = (pred_bin + target - pred_bin * target).sum(dim=(1, 2, 3)).clamp_min(1.0)
        cls_ious.append((inter / union).mean().item())

    total = torch.stack(cls_losses).mean()
    return total, cls_ious


def validate(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]) -> tuple[float, dict[str, float]]:
    model.eval()
    total_loss, n = 0.0, 0
    class_iou_sum = {name: 0.0 for name in class_names}

    with torch.no_grad():
        for images, targets, _ in loader:
            images, targets = images.to(device), targets.to(device)
            loss, cls_ious = compute_loss_and_metrics(model, images, targets)

            total_loss += loss.item()
            for i, name in enumerate(class_names):
                class_iou_sum[name] += cls_ious[i]
            n += 1

    class_iou = {name: class_iou_sum[name] / max(n, 1) for name in class_names}
    return total_loss / max(n, 1), class_iou


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune SAM3 on LoveDA selected classes.")
    p.add_argument("--sam3-ckpt", type=str, required=True, help="SAM3 checkpoint path, e.g. sam3_b.pt")
    p.add_argument("--train-images", type=Path, required=True)
    p.add_argument("--train-masks", type=Path, required=True)
    p.add_argument("--val-images", type=Path, required=True)
    p.add_argument("--val-masks", type=Path, required=True)
    p.add_argument("--focus-class-names", type=str, default="forest", help="Comma separated class names.")
    p.add_argument("--focus-semantic-ids", type=str, default="5", help="Comma separated semantic ids aligned with names.")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--freeze-backbone-epochs", type=int, default=3)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", type=Path, default=Path("runs/sam3_loveda"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    class_names = parse_csv_strs(args.focus_class_names)
    semantic_ids = parse_csv_ints(args.focus_semantic_ids)
    if len(class_names) != len(semantic_ids):
        raise ValueError("--focus-class-names and --focus-semantic-ids must have the same length")

    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LoveDAMultiClassDataset(args.train_images, args.train_masks, semantic_ids, args.imgsz, augment=True)
    val_ds = LoveDAMultiClassDataset(args.val_images, args.val_masks, semantic_ids, args.imgsz, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True, collate_fn=collate_fn)

    model = build_sam3_image_model(args.sam3_ckpt)
    model.set_classes(class_names)
    model.set_imgsz((args.imgsz, args.imgsz))
    model.to(device)

    for p in model.backbone.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    best_mean_iou = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            for p in model.backbone.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * 0.5, weight_decay=args.weight_decay)

        model.train()
        epoch_loss = 0.0

        for images, targets, _ in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
                loss, _ = compute_loss_and_metrics(model, images, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        train_loss = epoch_loss / max(len(train_loader), 1)
        val_loss, class_iou = validate(model, val_loader, device, class_names)
        mean_iou = float(np.mean(list(class_iou.values())))

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_iou": mean_iou,
            "class_iou": class_iou,
        }
        history.append(log)
        cls_iou_text = " ".join([f"{k}:{v:.4f}" for k, v in class_iou.items()])
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} mean_iou={mean_iou:.4f} {cls_iou_text}")

        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            torch.save(
                {
                    "model": model.state_dict(),
                    "best_mean_iou": best_mean_iou,
                    "class_names": class_names,
                    "semantic_ids": semantic_ids,
                    "imgsz": args.imgsz,
                },
                args.out / "best.pt",
            )

    with (args.out / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"Done. best_mean_iou={best_mean_iou:.4f}, saved to {args.out}")


if __name__ == "__main__":
    main()
