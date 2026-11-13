# TDS Vision Automation

A Windows automation project for **Roblox Tower Defense Simulator (TDS)** that augments a recorded strategy with real-time computer-vision checks and recovery logic.

The project began as a timing-only macro for the Wretched Front map. Fixed delays were unreliable: small differences in player position, camera state, tower placement, UI timing, or economy could cause later actions to drift out of sync. This project adds visual feedback so the automation can verify important game states instead of blindly replaying every input.

> This repository is an independent educational/personal project and is not affiliated with Roblox or Paradoxum Games.

## What it does

The current working bot combines a recorded strategy sequence with visual verification:

- normalizes the player/camera to a repeatable starting view;
- detects the **READY** and **Restart Match** buttons;
- registers the live map view against a reference using OpenCV template matching;
- corrects recorded placement coordinates for small per-run map shifts;
- detects whether a tower panel is open;
- verifies tower placements and retries nearby coordinates when necessary;
- reads tower upgrade progress from the panel;
- distinguishes affordable and unaffordable upgrades;
- prevents selected tower types from being upgraded past their intended strategy level;
- persists successful placement offsets between runs;
- supports an F8 emergency stop and saves diagnostic screenshots on failure.

The repository also contains standalone experiments used to develop and validate the individual perception and input subsystems.

## Architecture

```text
Roblox window
    |
    v
DXcam screen capture
    |
    +--> READY / restart detection
    +--> terrain registration
    +--> tower-panel detection
    +--> tower-level detection
    +--> upgrade-affordability detection
    |
    v
Recorded strategy + visual safeguards
    |
    v
PyDirectInput / Win32 input
    |
    v
Visual verification and retry
```

This is best described as **vision-assisted, feedback-driven automation** rather than a fully autonomous game-playing agent. The recorded strategy still supplies the high-level ordering and many timings, while computer vision validates critical actions and prevents several common failure modes.

See [docs/architecture.md](docs/architecture.md) for more detail.

## Requirements

Tested on:

- Windows
- Python 3.12
- 1920x1080 display
- Roblox in the expected fullscreen/window configuration

Core libraries:

- OpenCV
- DXcam
- NumPy
- PyDirectInput

Some diagnostic scripts also use Windows OCR through PyWinRT.

Install dependencies:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

## Running the bot

The production script is:

```text
tds_bot.py
```

Run it from the project directory:

```bat
.venv\Scripts\python.exe tds_bot.py
```

The script currently expects a 1920x1080 environment and checks the screen size before starting.

**F8** is the emergency stop.

### Important: registration reference

The current working build was developed with a local terrain-reference screenshot used for map registration. That image is intentionally not included in this first public cleanup because the original development copy lives under the ignored `test_output/` tree.

Until the reference is moved into a public `assets/` path and the production script is updated accordingly, a fresh clone will need the same reference image available at the path expected by `tds_bot.py`.

This is the main portability issue remaining in the repository.

## Repository contents

### Production

- `tds_bot.py` — current working automation script.

### Perception / UI diagnostics

- `ready_detector.py` — READY-button detector.
- `restart_detector.py` — Restart Match detector.
- `tower_panel_detector_test.py` — selected-tower panel detector test.
- `tower_level_detector_test.py` — tower level/progression detector test.
- `tds_money_wave_detector_test.py` — experimental money/wave OCR tool.
- `tds_visual_wave_tracker_test.py` — visual wave-transition tracker experiment.
- `tds_wave_detector_test.py` and `tds_wave_detector_strict.py` — earlier wave OCR experiments.

### Camera / positioning experiments

The remaining `*_test.py`, pitch, zoom, route, and anchor scripts document the calibration work used to find a reproducible player/camera state. They are development utilities rather than required runtime modules.

## Development history

The system evolved through several iterations:

1. timing-only Pulover/TinyTask-style replay;
2. direct Roblox-compatible keyboard/mouse input;
3. READY and restart detection;
4. deterministic camera normalization;
5. terrain-based registration to correct small placement shifts;
6. verified tower placement with retry offsets;
7. tower-panel and tower-level perception;
8. affordability-aware upgrade verification.

This progression is the main engineering motivation for the project: replacing fragile assumptions with increasingly strong visual feedback.

## Current limitations

- Designed around one specific TDS strategy/map/setup.
- Assumes 1920x1080 coordinates.
- The production strategy is still partially timing-driven.
- UI/theme/game updates can invalidate color thresholds or ROIs.
- The terrain registration reference is not yet packaged portably.
- Calibration and experimental scripts have not yet been consolidated into a formal test suite.

## Project status

The current `tds_bot.py` is the working build used during development. The next cleanup steps are primarily software-engineering improvements: packaging the registration asset, separating reusable modules, and organizing the diagnostic scripts without changing the verified gameplay behavior.
