# E-BARD Ball-Focused Dataset

Generated from `/Users/zhalgas/Documents/Codex/2026-06-03/i-want-to-create-ai-project/datasets/ebard-yolo-original` using `scripts/build_ebard_ball_focus_dataset.py`.

This dataset keeps only the E-BARD basketball labels and maps them to class `0: ball`.
Training split also includes ball-centered crops to improve small-object learning.

Generation settings:

- crop_size: 224
- crop_scales: 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0

Source: https://huggingface.co/datasets/GabrieleGiudici/E-BARD-detection
License: CC-BY-4.0
