# E-BARD 4-Class Ball-Focused Dataset

Generated from `/Users/zhalgas/Documents/Codex/2026-06-03/i-want-to-create-ai-project/datasets/ebard-yolo-original` using `scripts/build_ebard_4class_ball_focus_dataset.py`.

This dataset preserves the E-BARD detection labels and maps them to the app's canonical 4-class schema:

- `0: ball`
- `1: rim`
- `2: player`
- `3: referee`

The full-frame train/validation/test images are preserved. The training split also includes ball-centered crop
images to oversample small-ball examples while keeping any visible rim/player/referee labels inside each crop.
Optionally, full-context ball-positive train frames can also be duplicated so the model sees more ball examples
without changing the broadcast-camera scale.

Generation settings:

- crop_size: 320
- crop_scales: 6.0, 10.0, 14.0
- min_visibility: 0.15
- positive_frame_copies: 0
- output: `/Users/zhalgas/Documents/Codex/2026-06-03/i-want-to-create-ai-project/datasets/ebard-4class-fullframe-v1`

Source: https://huggingface.co/datasets/GabrieleGiudici/E-BARD-detection
License: CC-BY-4.0
