# Registration reference image

The bot needs a **1920 x 1080 screenshot captured at the recorded absolute map anchor** to register live terrain against the strategy's coordinates.

Copy the original file from your development machine:

```text
test_output/recording_anchor_absolute/20260724_171710/03_FINAL_RECORDING_ANCHOR.png
```

to:

```text
assets/registration_reference.png
```

Keep the image dimensions and contents unchanged. Do not substitute a screenshot taken at a different camera or player position: the strategy's coordinates were calibrated against the original reference.

The production script checks `assets/registration_reference.png` first, then the original timestamped local capture, then other timestamped `03_FINAL_RECORDING_ANCHOR.png` files under `test_output/recording_anchor_absolute/`. The first two paths preserve the intended exact reference; the final fallback is a diagnostic convenience and may not match the recorded strategy.

The binary reference image is not included in this repository yet. The code will raise an actionable error if neither the packaged image nor a local fallback exists. Until the original image is added, a fresh clone is **not fully ready to run**.

Before publishing the screenshot, inspect it for personal/account information and anything else you would not want public.
