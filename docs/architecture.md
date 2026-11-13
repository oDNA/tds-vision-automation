# Architecture

## Overview

The production bot is a hybrid automation system: a recorded high-level strategy provides action ordering and timing, while computer-vision detectors verify critical states and recover from several classes of drift.

The core design goal is **repeatability despite small run-to-run differences**.

## Capture

Screen frames are captured continuously with DXcam using the WinRT backend:

```python
dxcam.create(
    backend="winrt",
    output_color="BGR",
)
```

Continuous capture is used so visual checks can operate on fresh frames rather than isolated stale screenshots.

## Input

Keyboard and normal gameplay input use PyDirectInput because Roblox accepted it more reliably than PyAutoGUI during development.

Camera pitch uses Win32 relative mouse movement while the right mouse button is held. This was required to reproduce Roblox's raw camera behavior.

## Initial normalization

The one-time startup routine:

1. verifies that READY is visible;
2. moves the player along a fixed route;
3. forces maximum zoom-out with repeated `O` presses;
4. forces the camera against the maximum top-down pitch clamp.

The purpose is not to make timed movement perfectly deterministic. Instead, it produces a sufficiently consistent view for the registration stage to measure the remaining translation.

## Map registration

Even after camera normalization, the terrain can shift by a few pixels between runs.

The bot compares several static terrain patches against a saved reference image:

1. convert the reference and live patches to grayscale;
2. blur and run Canny edge detection;
3. search a bounded area with `cv2.matchTemplate`;
4. reject low-confidence or implausibly large shifts;
5. use agreement between multiple patches to select the final `(dx, dy)`.

The resulting translation is added to recorded map coordinates.

## Tower placement verification

A tower hotkey arms the next click as a placement.

After the click, the bot checks for the selected-tower panel. The panel detector uses two visual features:

- the blue question-mark button;
- the red Sell button.

Because hover tooltips can obscure part of the Sell button, the detector uses debounced logic and can fall back to repeated question-mark detection.

If placement is not verified, nearby coordinate offsets are tried. Successful local offsets are stored in `placement_offsets.json` so future runs can start with a previously successful correction.

## Tower level detection

The selected-tower panel represents purchased levels using progression segments.

A purchased segment can appear:

- bright green when the next upgrade is affordable;
- dark grey when the next upgrade is unaffordable.

The detector therefore combines both color classes and counts connected components in the progression-bar region.

This lets the bot verify that an upgrade increased the tower by exactly one level and prevents accidental extra upgrades on capped tower types.

## Upgrade affordability

Affordability is detected separately from tower level.

A fixed region of the upgrade card is sampled in HSV space:

- high saturation indicates the green/affordable state;
- low saturation indicates the grey/unaffordable state.

The bot waits for the affordable state before sending another upgrade input.

## Strategy playback

The current production build still parses an embedded recorded strategy consisting of:

- sleeps;
- key presses;
- mouse down/up events.

Visual state is layered around that sequence:

- READY and restart are detector-controlled;
- placement coordinates receive map-registration correction;
- placements are verified and retried;
- upgrades are verified against tower level and affordability;
- selected tower types are capped at strategy-appropriate levels.

This makes the current implementation more robust than a blind macro, but it is not a fully state-driven strategy engine.

## Safety and diagnostics

- F8 raises a stop request and releases held inputs.
- Input sleeps are interruptible.
- Failure states are saved under `test_output/`.
- Registration and placement diagnostics are saved for later inspection.
- `test_output/` and local learned placement state are ignored by Git.
