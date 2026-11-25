from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a fused Ultralytics YOLO detection checkpoint into a trainable checkpoint."
    )
    parser.add_argument("--source", type=Path, required=True, help="Fused YOLO .pt checkpoint.")
    parser.add_argument("--output", type=Path, required=True, help="Output trainable YOLO .pt checkpoint.")
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
        from ultralytics.nn.tasks import DetectionModel
    except ImportError as exc:
        raise RuntimeError("Install training dependencies with: pip install -e '.[train]'") from exc

    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()
    fused_yolo = YOLO(str(source))
    fused_model = fused_yolo.model.float().eval()

    trainable_model = DetectionModel(copy.deepcopy(fused_model.yaml), ch=3, nc=fused_model.yaml.get("nc"), verbose=False)
    trainable_model.names = copy.deepcopy(getattr(fused_model, "names", {}))
    trainable_model.float().eval()

    copied_tensors = _copy_matching_tensors(fused_model, trainable_model)
    converted_bn = _copy_fused_conv_biases_to_bn(fused_model, trainable_model)
    trainable_model.float().eval()

    checkpoint = {
        "epoch": -1,
        "best_fitness": None,
        "model": None,
        "ema": trainable_model,
        "updates": 0,
        "optimizer": None,
        "train_args": {
            "task": "detect",
            "mode": "train",
            "model": str(source),
        },
        "train_metrics": None,
        "train_results": None,
        "date": None,
        "version": "converted",
        "license": None,
        "docs": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)

    print(f"Converted fused checkpoint: {source}")
    print(f"Output trainable checkpoint: {output}")
    print(f"Copied matching tensors: {copied_tensors}")
    print(f"Converted fused conv biases to BatchNorm: {converted_bn}")


def _copy_matching_tensors(fused_model: torch.nn.Module, trainable_model: torch.nn.Module) -> int:
    fused_state = fused_model.state_dict()
    trainable_state = trainable_model.state_dict()
    matching_state = {
        key: value
        for key, value in fused_state.items()
        if key in trainable_state and trainable_state[key].shape == value.shape
    }
    trainable_model.load_state_dict(matching_state, strict=False)
    return len(matching_state)


def _copy_fused_conv_biases_to_bn(fused_model: torch.nn.Module, trainable_model: torch.nn.Module) -> int:
    fused_modules = dict(fused_model.named_modules())
    converted = 0
    for name, train_module in trainable_model.named_modules():
        if not hasattr(train_module, "conv") or not hasattr(train_module, "bn"):
            continue
        fused_module = fused_modules.get(name)
        if fused_module is None or not hasattr(fused_module, "conv"):
            continue

        fused_bias = getattr(fused_module.conv, "bias", None)
        bn = train_module.bn
        bn.weight.data.fill_(1.0)
        bn.bias.data.zero_()
        bn.running_mean.data.zero_()
        bn.running_var.data.fill_(max(1.0 - float(bn.eps), 0.0))
        if fused_bias is not None:
            bn.bias.data.copy_(fused_bias.data.float())
        converted += 1
    return converted


if __name__ == "__main__":
    main()
