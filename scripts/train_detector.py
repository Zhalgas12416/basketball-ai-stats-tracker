from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a basketball object detector with Ultralytics YOLO.")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "configs" / "basketball-detector.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training dataset to use.")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=None, help="Initial learning rate for fine-tuning.")
    parser.add_argument("--lrf", type=float, default=None, help="Final learning-rate factor.")
    parser.add_argument("--optimizer", default=None, help="Ultralytics optimizer name, for example AdamW or SGD.")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze the first N model layers during training.")
    parser.add_argument("--mosaic", type=float, default=None, help="Mosaic augmentation probability.")
    parser.add_argument("--copy-paste", type=float, default=None, help="Copy-paste augmentation probability.")
    parser.add_argument("--close-mosaic", type=int, default=None, help="Disable mosaic for the final N epochs.")
    parser.add_argument("--rect", action="store_true", help="Use rectangular training batches.")
    parser.add_argument("--scale", type=float, default=None, help="Image scale augmentation gain.")
    parser.add_argument("--translate", type=float, default=None, help="Image translation augmentation gain.")
    parser.add_argument("--erasing", type=float, default=None, help="Random erasing augmentation probability.")
    parser.add_argument("--hsv-h", type=float, default=None, help="HSV hue augmentation gain.")
    parser.add_argument("--hsv-s", type=float, default=None, help="HSV saturation augmentation gain.")
    parser.add_argument("--hsv-v", type=float, default=None, help="HSV value augmentation gain.")
    parser.add_argument("--box", type=float, default=None, help="Box loss gain.")
    parser.add_argument("--cls", type=float, default=None, help="Classification loss gain.")
    parser.add_argument("--dfl", type=float, default=None, help="Distribution focal loss gain.")
    parser.add_argument("--cos-lr", action="store_true", help="Use cosine learning-rate schedule.")
    parser.add_argument("--single-cls", action="store_true", help="Train all labeled objects as one class.")
    parser.add_argument(
        "--freeze-bn",
        action="store_true",
        help="Keep all BatchNorm layers in eval mode and disable their gradients during training.",
    )
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT / "models" / "runs")
    parser.add_argument("--name", default="basketball-detector")
    parser.add_argument("--export-to", type=Path, default=PROJECT_ROOT / "models" / "basketball-detector.pt")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -e '.[train]'") from exc

    model = YOLO(args.model)
    if args.freeze_bn:
        model.add_callback("on_train_start", _freeze_batch_norm)
        model.add_callback("on_train_epoch_start", _freeze_batch_norm)
        model.add_callback("on_train_batch_start", _freeze_batch_norm)
    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "fraction": args.fraction,
        "patience": args.patience,
        "project": str(args.project),
        "name": args.name,
    }
    if args.device:
        train_kwargs["device"] = args.device
    optional_train_kwargs = {
        "lr0": args.lr0,
        "lrf": args.lrf,
        "optimizer": args.optimizer,
        "freeze": args.freeze,
        "mosaic": args.mosaic,
        "copy_paste": args.copy_paste,
        "close_mosaic": args.close_mosaic,
        "scale": args.scale,
        "translate": args.translate,
        "erasing": args.erasing,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "box": args.box,
        "cls": args.cls,
        "dfl": args.dfl,
    }
    for key, value in optional_train_kwargs.items():
        if value is not None:
            train_kwargs[key] = value
    if args.cos_lr:
        train_kwargs["cos_lr"] = True
    if args.single_cls:
        train_kwargs["single_cls"] = True
    if args.rect:
        train_kwargs["rect"] = True

    results = model.train(**train_kwargs)
    save_dir = Path(results.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    print(f"Best weights: {best_weights}")

    if args.export_to and best_weights.exists():
        args.export_to.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_weights, args.export_to)
        print(f"Copied best weights to: {args.export_to}")

def _freeze_batch_norm(trainer) -> None:
    import torch.nn as nn

    for module in trainer.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False


if __name__ == "__main__":
    main()
