from __future__ import annotations

import ctypes
import json
import math
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys
import time

import cv2
import dxcam
import numpy as np
import pydirectinput


# ============================================================
# USER CONFIG
# ============================================================

# None = repeat forever until F8 / Ctrl+C.
# For the first test, keeping this at 1 is recommended.
MAX_RUNS: int | None = 1

RUN_INITIAL_SETUP = True

STARTUP_DELAY_SECONDS = 5.0
EXPECTED_SCREEN_SIZE = (1920, 1080)

# F8 emergency stop.
STOP_VK = 0x77


# ============================================================
# ONE-TIME PLAYER ROUTE
# ============================================================

A_HOLD_SECONDS = 0.625
AFTER_A_PAUSE_SECONDS = 0.547

S_HOLD_SECONDS = 2.000
JUMP_AFTER_S_SECONDS = 0.140
SPACE_HOLD_SECONDS = 0.080

AFTER_S_PAUSE_SECONDS = 0.422

D_HOLD_SECONDS = 0.968

CAMERA_RECOVERY_SECONDS = 1.5


# ============================================================
# ABSOLUTE CAMERA NORMALIZATION
#
# O x20         -> maximum zoom-out
# pitch +120 x8 -> maximum top-down
# I x0
# ============================================================

MAX_ZOOM_OUT_PRESSES = 20
ZOOM_KEY_HOLD_SECONDS = 0.080
ZOOM_BETWEEN_PRESSES_SECONDS = 0.120
ZOOM_SETTLE_SECONDS = 1.0

PITCH_STEP_DY = 120
PITCH_SATURATION_STEPS = 8
PITCH_BETWEEN_STEPS_SECONDS = 0.30
PITCH_SETTLE_SECONDS = 1.0


# ============================================================
# READY / RESTART DETECTION
# ============================================================

DETECT_POLL_SECONDS = 0.15

READY_WAIT_TIMEOUT = None
RESTART_WAIT_TIMEOUT = None

READY_DISAPPEAR_TIMEOUT = 2.0
RESTART_DISAPPEAR_TIMEOUT = 20.0

READY_ROI = (
    930 / 1920,
    160 / 1080,
    1120 / 1920,
    290 / 1080,
)

RESTART_ROI = (
    0.34,
    0.68,
    0.52,
    0.79,
)


# ============================================================
# VISUAL MAP REGISTRATION
# ============================================================

REFERENCE_IMAGE_EXACT = (
    Path(__file__).resolve().parent
    / "assets"
    / "registration_reference.png"
)

# Retain the exact original capture as a local-only fallback.
# The preferred asset above is stable across clone locations.
REFERENCE_IMAGE_LEGACY = (
    Path(__file__).resolve().parent
    / "test_output"
    / "recording_anchor_absolute"
    / "20260724_171710"
    / "03_FINAL_RECORDING_ANCHOR.png"
)

REGISTRATION_PATCHES = [
    ("left_rock_primary", 430, 180, 760, 500),
    ("left_lower", 300, 450, 580, 700),
    ("right_middle", 1060, 250, 1330, 530),
    ("lower_middle_right", 880, 500, 1150, 700),
]

REGISTRATION_SEARCH_MARGIN = 35
REGISTRATION_MIN_PATCH_CONFIDENCE = 0.30
REGISTRATION_MIN_PRIMARY_CONFIDENCE = 0.43
REGISTRATION_MAX_SHIFT = 25
REGISTRATION_INLIER_TOLERANCE = 5
REGISTRATION_MIN_CONSENSUS_CONFIDENCE = 0.38


# ============================================================
# TOWER PANEL DETECTION
#
# A successful placement selects the newly placed tower and
# opens this panel.
#
# Primary proof:
#   blue ? button in the fixed lower-middle panel.
#
# Secondary proof:
#   large red Sell button.
#
# The Sell button can be temporarily covered by a loadout
# hover tooltip. Therefore:
#
#   - 2 consecutive frames with BOTH features = PRESENT
#   - OR 4 consecutive frames with the ? button = PRESENT
#
# This avoids the false-negative seen during the detector test.
# ============================================================

QUESTION_ROI = (
    1190,
    490,
    1340,
    650,
)

SELL_ROI = (
    620,
    800,
    970,
    950,
)

QUESTION_MIN_AREA = 850
QUESTION_MIN_WIDTH = 34
QUESTION_MIN_HEIGHT = 34

SELL_MIN_AREA = 2500
SELL_MIN_WIDTH = 150
SELL_MIN_HEIGHT = 20

PANEL_SAMPLE_INTERVAL = 0.045
PANEL_PRESENT_TIMEOUT = 0.75
PANEL_ABSENT_TIMEOUT = 0.40

PANEL_BOTH_STREAK_REQUIRED = 2
PANEL_QUESTION_STREAK_REQUIRED = 4
PANEL_ABSENT_STREAK_REQUIRED = 2


# ============================================================
# VERIFIED PLACEMENT RETRIES
#
# (0,0) is the original registered coordinate.
# Remaining offsets are tried only if the panel did not open.
#
# Slight upward bias because the remaining failures have
# generally landed too low / too close to the track.
# ============================================================

PLACEMENT_HOTKEYS = {
    "1",  # Farm
    "2",  # Boomerang
    "3",  # Freezer
    "4",  # Crook Boss
}

PLACEMENT_RETRY_OFFSETS = [
    (0, -4),
    (-4, -2),
    (+4, -2),
    (0, +4),
    (-6, -5),
    (+6, -5),
    (-6, +3),
    (+6, +3),
    (0, -9),
    (-9, -6),
    (+9, -6),
    (-9, +4),
    (+9, +4),
]

PLACEMENT_RETRY_CLICK_HOLD = 0.110


# ============================================================
# PERSISTENT PLACEMENT LEARNING
#
# These are local offsets relative to the visually registered
# map coordinate. They were learned from the last successful
# placement test and will be updated automatically whenever a
# different retry succeeds.
# ============================================================

SEED_PLACEMENT_OFFSETS = {
    1: (0, 0),
    2: (0, 0),
    3: (-4, -2),
    4: (+9, -6),
    5: (0, 0),
    6: (-4, -2),
    7: (-4, -2),
    8: (0, 0),
    9: (0, 0),
    10: (0, -4),
    11: (0, 0),
    12: (0, 0),
    13: (-4, -2),
    14: (+4, -2),
}

TOWER_NAMES = {
    "1": "Farm",
    "2": "Boomerang",
    "3": "Freezer",
    "4": "Crook Boss",
}

# A normal strategy selection click is associated with the
# nearest tower we previously placed.
SELECTION_MATCH_DISTANCE = 60.0

# When an upgrade is about to happen, click the remembered
# tower position again to guarantee that it is selected.
SELECTION_RETRY_OFFSETS = [
    (0, 0),
    (0, -4),
    (-4, 0),
    (+4, 0),
    (0, +4),
    (-6, -4),
    (+6, -4),
    (-6, +4),
    (+6, +4),
]

# Move the cursor here before inspecting the tower panel so a
# loadout hover tooltip cannot cover the panel.
NEUTRAL_MOUSE_POSITION = (
    1500,
    300,
)


# ============================================================
# TOWER LEVEL DETECTION
#
# Current level = number of bright green progression segments.
# The total number of segments may differ by tower; that does
# not matter because only the lit segments are counted.
# ============================================================

LEVEL_BAR_ROI_NORM = (
    930 / 1917,
    660 / 1077,
    1280 / 1917,
    695 / 1077,
)

# An upgraded level segment has TWO possible appearances:
#
# 1. Enough money for the next upgrade:
#       bright saturated green
#
# 2. Not enough money for the next upgrade:
#       dark neutral grey
#
# Unpurchased future levels are:
#       dark green when affordable
#       lighter neutral grey when unaffordable
#
# Therefore level detection combines:
#       bright-green upgraded fills
#       +
#       dark-grey upgraded fills

UPGRADED_GREEN_HSV_LOW = np.array(
    [50, 155, 180],
    dtype=np.uint8,
)

UPGRADED_GREEN_HSV_HIGH = np.array(
    [82, 255, 255],
    dtype=np.uint8,
)

UPGRADED_DARK_GREY_HSV_LOW = np.array(
    [0, 0, 25],
    dtype=np.uint8,
)

UPGRADED_DARK_GREY_HSV_HIGH = np.array(
    [179, 45, 52],
    dtype=np.uint8,
)

LEVEL_SEGMENT_MIN_WIDTH = 35
LEVEL_SEGMENT_MAX_WIDTH = 75
LEVEL_SEGMENT_MIN_HEIGHT = 7
LEVEL_SEGMENT_MAX_HEIGHT = 22
LEVEL_SEGMENT_MIN_AREA = 300

LEVEL_STABLE_FRAMES_REQUIRED = 3
LEVEL_READ_TIMEOUT = 1.25


# Upgrade-card affordability detector.
#
# Green card:
#     high saturation
#
# Grey card:
#     near-zero saturation
#
# The sampled region deliberately avoids the progression bars.
UPGRADE_CARD_SAMPLE_ROI_NORM = (
    1160 / 1917,
    595 / 1077,
    1255 / 1917,
    648 / 1077,
)

UPGRADE_AFFORDABLE_MIN_MEDIAN_SATURATION = 90

AFFORDABILITY_STABLE_FRAMES_REQUIRED = 2
AFFORDABILITY_READ_TIMEOUT = 1.0
UPGRADE_AFFORDABLE_WAIT_TIMEOUT = 75.0


# ============================================================
# VERIFIED UPGRADES
#
# Every recorded E now means:
#   "increase the selected tower's level by exactly one".
#
# Farms / Freezers / Crook Bosses are capped at level 3 for
# this strategy. This specifically prevents the accidental
# extra Farm E in the recording from buying level 4.
#
# Boomerangs intentionally have NO fixed cap here. The macro
# ordering is preserved:
#   first Boomerang -> level 2
#   second Boomerang -> level 2
#   first -> level 3
#   second -> level 3
#   later each receives its recorded max-level upgrade.
# ============================================================

STRATEGY_LEVEL_CAPS = {
    "1": 3,  # Farm
    "3": 3,  # Freezer
    "4": 3,  # Crook Boss
}

UPGRADE_TOTAL_TIMEOUT = 60.0
UPGRADE_LEVEL_CHANGE_TIMEOUT = 2.50
UPGRADE_RETRY_DELAY = 0.75
UPGRADE_TRANSIENT_SETTLE_SECONDS = 0.35
UPGRADE_REGRESSION_RECHECKS = 3

# Only small normal detector overhead is compensated against
# later recorded sleeps. Long waits for money are NOT erased.
PLACEMENT_TIMING_CREDIT_CAP = 0.35
UPGRADE_TIMING_CREDIT_CAP = 0.35


# ============================================================
# OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


PLACEMENT_OFFSETS_PATH = (
    BASE_DIR
    / "placement_offsets.json"
)

ERROR_OUTPUT_ROOT = (
    BASE_DIR
    / "test_output"
    / "tds_bot"
    / "errors"
)

REGISTRATION_OUTPUT_ROOT = (
    BASE_DIR
    / "test_output"
    / "tds_bot"
    / "registration"
)

PLACEMENT_OUTPUT_ROOT = (
    BASE_DIR
    / "test_output"
    / "tds_bot"
    / "placement"
)


# ============================================================
# CURRENT STRATEGY
#
# This is ONLY the second recording made from the absolute
# recording anchor.
#
# Removed:
#   - WinActivate
#   - pre-READY sleeps
#   - READY click
#   - final 7922 ms sleep
#   - Restart Match click
#
# Python handles READY and Restart separately.
# ============================================================

STRATEGY_AHK = r"""
Sleep, 984
Send, {1}
Sleep, 1188
Click, 602, 432 Left, Down
Sleep, 109
Click, 602, 432 Left, Up
Sleep, 657
Send, {e}
Sleep, 453
Send, {e}
Sleep, 734
Click, 759, 450 Left, Down
Sleep, 125
Click, 759, 450 Left, Up
Sleep, 2750
Send, {1}
Sleep, 4109
Click, 500, 500 Left, Down
Sleep, 110
Click, 500, 500 Left, Up
Sleep, 1031
Send, {e}
Sleep, 9000
Send, {e}
Sleep, 859
Click, 590, 513 Left, Down
Sleep, 125
Click, 590, 513 Left, Up
Sleep, 485
Send, {1}
Sleep, 3359
Click, 600, 498 Left, Down
Sleep, 125
Click, 600, 498 Left, Up
Sleep, 1360
Send, {e}
Sleep, 3515
Send, {e}
Sleep, 750
Click, 654, 506 Left, Down
Sleep, 125
Click, 654, 506 Left, Up
Sleep, 766
Send, {1}
Sleep, 2515
Click, 658, 496 Left, Down
Sleep, 125
Click, 658, 496 Left, Up
Sleep, 1063
Send, {e}
Sleep, 5969
Send, {e}
Sleep, 984
Click, 881, 489 Left, Down
Sleep, 125
Click, 881, 489 Left, Up
Sleep, 641
Send, {2}
Sleep, 3000
Click, 986, 402 Left, Down
Sleep, 109
Click, 986, 402 Left, Up
Sleep, 906
Send, {e}
Sleep, 5250
Send, {e}
Sleep, 1657
Click, 1022, 428 Left, Down
Sleep, 109
Click, 1022, 428 Left, Up
Sleep, 984
Send, {1}
Sleep, 1375
Click, 478, 567 Left, Down
Sleep, 110
Click, 478, 567 Left, Up
Sleep, 828
Send, {e}
Sleep, 3453
Send, {e}
Sleep, 2453
Send, {1}
Sleep, 1422
Click, 546, 565 Left, Down
Sleep, 94
Click, 546, 565 Left, Up
Sleep, 1203
Send, {e}
Sleep, 1250
Send, {e}
Sleep, 1328
Send, {1}
Sleep, 2437
Click, 610, 561 Left, Down
Sleep, 94
Click, 610, 561 Left, Up
Sleep, 703
Send, {e}
Sleep, 4063
Send, {e}
Sleep, 719
Send, {1}
Sleep, 1453
Click, 677, 565 Left, Down
Sleep, 125
Click, 677, 565 Left, Up
Sleep, 1328
Send, {e}
Sleep, 1781
Send, {e}
Sleep, 531
Send, {1}
Sleep, 1969
Click, 670, 630 Left, Down
Sleep, 94
Click, 670, 630 Left, Up
Sleep, 1047
Send, {e}
Sleep, 859
Send, {e}
Sleep, 656
Send, {1}
Sleep, 2594
Click, 550, 638 Left, Down
Sleep, 109
Click, 550, 638 Left, Up
Sleep, 1094
Send, {e}
Sleep, 2735
Send, {e}
Sleep, 1171
Click, 778, 492 Left, Down
Sleep, 110
Click, 778, 492 Left, Up
Sleep, 390
Send, {2}
Sleep, 1313
Click, 1035, 403 Left, Down
Sleep, 109
Click, 1035, 403 Left, Up
Sleep, 891
Send, {e}
Sleep, 859
Send, {e}
Sleep, 2688
Click, 992, 399 Left, Down
Sleep, 109
Click, 992, 399 Left, Up
Sleep, 5703
Send, {e}
Sleep, 3000
Click, 1034, 404 Left, Down
Sleep, 125
Click, 1034, 404 Left, Up
Sleep, 7657
Send, {e}
Sleep, 1828
Click, 1031, 470 Left, Down
Sleep, 140
Click, 1031, 470 Left, Up
Sleep, 547
Send, {3}
Sleep, 2844
Click, 1030, 359 Left, Down
Sleep, 141
Click, 1030, 359 Left, Up
Sleep, 1062
Send, {e}
Sleep, 906
Send, {e}
Sleep, 3657
Send, {e}
Sleep, 843
Send, {3}
Sleep, 8875
Click, 981, 361 Left, Down
Sleep, 141
Click, 981, 361 Left, Up
Sleep, 1391
Send, {e}
Sleep, 703
Send, {e}
Sleep, 718
Send, {e}
Sleep, 2579
Click, 608, 446 Left, Down
Sleep, 109
Click, 608, 446 Left, Up
Sleep, 281
Send, {e}
Sleep, 719
Click, 497, 486 Left, Down
Sleep, 94
Click, 497, 486 Left, Up
Sleep, 437
Send, {e}
Sleep, 547
Click, 603, 482 Left, Down
Sleep, 109
Click, 603, 482 Left, Up
Sleep, 610
Send, {e}
Sleep, 531
Click, 688, 482 Left, Down
Sleep, 109
Click, 688, 482 Left, Up
Sleep, 2922
Send, {e}
Sleep, 641
Click, 469, 559 Left, Down
Sleep, 94
Click, 469, 559 Left, Up
Sleep, 312
Send, {e}
Sleep, 672
Click, 544, 555 Left, Down
Sleep, 94
Click, 544, 555 Left, Up
Sleep, 5484
Send, {e}
Sleep, 1656
Send, {e}
Sleep, 719
Click, 620, 545 Left, Down
Sleep, 110
Click, 620, 545 Left, Up
Sleep, 2656
Send, {e}
Sleep, 1094
Click, 676, 548 Left, Down
Sleep, 125
Click, 676, 548 Left, Up
Sleep, 484
Send, {e}
Sleep, 437
Click, 759, 526 Left, Down
Sleep, 110
Click, 759, 526 Left, Up
Sleep, 703
Click, 556, 619 Left, Down
Sleep, 109
Click, 556, 619 Left, Up
Sleep, 516
Send, {e}
Sleep, 594
Click, 811, 488 Left, Down
Sleep, 93
Click, 811, 488 Left, Up
Sleep, 391
Click, 660, 643 Left, Down
Sleep, 109
Click, 660, 643 Left, Up
Sleep, 579
Send, {e}
Sleep, 1671
Click, 926, 455 Left, Down
Sleep, 110
Click, 926, 455 Left, Up
Sleep, 656
Click, 987, 398 Left, Down
Sleep, 125
Click, 987, 398 Left, Up
Sleep, 19109
Send, {e}
Sleep, 2547
Click, 1032, 399 Left, Down
Sleep, 110
Click, 1032, 399 Left, Up
Sleep, 23015
Send, {e}
Sleep, 3297
Send, {4}
Sleep, 1703
Click, 934, 401 Left, Down
Sleep, 110
Click, 934, 401 Left, Up
Sleep, 1281
Send, {e}
Sleep, 9250
Send, {e}
Sleep, 3562
Send, {4}
Sleep, 1297
Click, 927, 355 Left, Down
Sleep, 110
Click, 927, 355 Left, Up
Sleep, 1234
Send, {e}
Sleep, 28813
Send, {e}
Sleep, 1765
Send, {4}
Sleep, 2063
Click, 883, 401 Left, Down
Sleep, 156
Click, 883, 401 Left, Up
Sleep, 937
Send, {e}
Sleep, 10516
Send, {e}
Sleep, 2297
Click, 996, 450 Left, Down
Sleep, 94
Click, 996, 450 Left, Up
Sleep, 906
Send, {4}
Sleep, 6953
Click, 979, 499 Left, Down
Sleep, 141
Click, 979, 499 Left, Up
Sleep, 953
Send, {e}
Sleep, 4703
Send, {e}
Sleep, 2078
Click, 905, 498 Left, Down
Sleep, 109
Click, 905, 498 Left, Up
Sleep, 516
Send, {4}
Sleep, 2781
Click, 929, 499 Left, Down
Sleep, 125
Click, 929, 499 Left, Up
Sleep, 1063
Send, {e}
Sleep, 13594
Send, {e}
Sleep, 2921
Click, 874, 384 Left, Down
Sleep, 110
Click, 874, 384 Left, Up
Sleep, 22875
Click, 934, 353 Left, Down
Sleep, 109
Click, 934, 353 Left, Up
Sleep, 38016
Send, {e}
Sleep, 2031
Click, 934, 387 Left, Down
Sleep, 125
Click, 934, 387 Left, Up
Sleep, 23641
Send, {e}
Sleep, 4281
Click, 934, 504 Left, Down
Sleep, 141
Click, 934, 504 Left, Up
Sleep, 13687
Send, {e}
Sleep, 11391
Send, {x}
Sleep, 484
Click, 988, 496 Left, Down
Sleep, 94
Click, 988, 496 Left, Up
Sleep, 281
Send, {x}
Sleep, 1000
Click, 929, 398 Left, Down
Sleep, 109
Click, 929, 398 Left, Up
Sleep, 250
Send, {x}
Sleep, 485
Click, 890, 390 Left, Down
Sleep, 109
Click, 890, 390 Left, Up
Sleep, 297
Send, {x}
Sleep, 500
Click, 933, 352 Left, Down
Sleep, 94
Click, 933, 352 Left, Up
Sleep, 265
Send, {x}
Sleep, 516
Click, 976, 404 Left, Down
Sleep, 109
Click, 976, 405 Left, Up
Sleep, 282
Click, 985, 398 Left, Down
Sleep, 109
Click, 985, 398 Left, Up
Sleep, 250
Send, {x}
Sleep, 375
Click, 984, 355 Left, Down
Sleep, 125
Click, 984, 355 Left, Up
Sleep, 281
Send, {x}
Sleep, 469
Click, 1036, 357 Left, Down
Sleep, 109
Click, 1036, 357 Left, Up
Sleep, 219
Send, {x}
Sleep, 422
Click, 1035, 395 Left, Down
Sleep, 125
Click, 1035, 395 Left, Up
Sleep, 219
Send, {x}
"""


# ============================================================
# WINDOWS / INPUT
# ============================================================

pydirectinput.PAUSE = 0

user32 = ctypes.WinDLL(
    "user32",
    use_last_error=True,
)

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010


class StopRequested(Exception):
    pass


class POINT(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_long),
        ("y", ctypes.c_long),
    ]


def stop_requested() -> bool:
    return bool(
        user32.GetAsyncKeyState(STOP_VK)
        & 0x8000
    )


def check_stop() -> None:
    if stop_requested():
        raise StopRequested


def interruptible_sleep(seconds: float) -> None:
    if seconds <= 0:
        return

    deadline = time.perf_counter() + seconds

    while True:
        check_stop()

        remaining = deadline - time.perf_counter()

        if remaining <= 0:
            return

        time.sleep(
            min(
                0.05,
                remaining,
            )
        )


# ============================================================
# ROBLOX WINDOW
# ============================================================

def activate_roblox() -> bool:
    """
    Bring Roblox forward without changing fullscreen state.
    """

    found_hwnd = None

    @ctypes.WINFUNCTYPE(
        ctypes.c_bool,
        wintypes.HWND,
        wintypes.LPARAM,
    )
    def enum_proc(hwnd, _lparam):
        nonlocal found_hwnd

        if not user32.IsWindowVisible(hwnd):
            return True

        length = user32.GetWindowTextLengthW(hwnd)

        if length <= 0:
            return True

        buf = ctypes.create_unicode_buffer(
            length + 1
        )

        user32.GetWindowTextW(
            hwnd,
            buf,
            length + 1,
        )

        title = buf.value.strip()

        if (
            title == "Roblox"
            or title.startswith("Roblox")
        ):
            found_hwnd = hwnd
            return False

        return True

    user32.EnumWindows(
        enum_proc,
        0,
    )

    if not found_hwnd:
        return False

    # Do not blindly SW_RESTORE: that previously changed
    # fullscreen state. Restore only if actually minimized.
    if user32.IsIconic(found_hwnd):
        user32.ShowWindow(
            found_hwnd,
            9,  # SW_RESTORE
        )

        time.sleep(0.25)

    user32.SetForegroundWindow(
        found_hwnd
    )

    return True


def current_screen_size() -> tuple[int, int]:
    return (
        user32.GetSystemMetrics(0),
        user32.GetSystemMetrics(1),
    )


# ============================================================
# RAW MOUSE / CURSOR
# ============================================================

def relative_mouse_move(
    dx: int,
    dy: int,
) -> None:
    user32.mouse_event(
        MOUSEEVENTF_MOVE,
        int(dx),
        int(dy),
        0,
        0,
    )


def right_mouse_down() -> None:
    user32.mouse_event(
        MOUSEEVENTF_RIGHTDOWN,
        0,
        0,
        0,
        0,
    )


def right_mouse_up() -> None:
    user32.mouse_event(
        MOUSEEVENTF_RIGHTUP,
        0,
        0,
        0,
        0,
    )


def get_cursor_position() -> tuple[int, int]:
    point = POINT()

    if not user32.GetCursorPos(
        ctypes.byref(point)
    ):
        raise ctypes.WinError(
            ctypes.get_last_error()
        )

    return (
        int(point.x),
        int(point.y),
    )


def set_cursor_position_verified(
    x: int,
    y: int,
    timeout: float = 0.35,
) -> None:
    x = int(x)
    y = int(y)

    if not user32.SetCursorPos(
        x,
        y,
    ):
        raise ctypes.WinError(
            ctypes.get_last_error()
        )

    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:
        check_stop()

        current_x, current_y = (
            get_cursor_position()
        )

        if (
            abs(current_x - x) <= 1
            and abs(current_y - y) <= 1
        ):
            return

        time.sleep(0.005)

    current_x, current_y = (
        get_cursor_position()
    )

    raise RuntimeError(
        "Cursor failed to reach target. "
        f"Target=({x}, {y}), "
        f"actual=({current_x}, {current_y})"
    )


def prepare_roblox_click(
    x: int,
    y: int,
) -> None:
    """
    Cursor synchronization proven useful with Roblox:
      exact SetCursorPos
      -> verify
      -> 50 ms
      -> +1 px
      -> -1 px
      -> mouse-down may occur
    """

    set_cursor_position_verified(
        x,
        y,
    )

    interruptible_sleep(0.050)

    relative_mouse_move(
        1,
        0,
    )

    interruptible_sleep(0.020)

    relative_mouse_move(
        -1,
        0,
    )

    interruptible_sleep(0.030)


def deliberate_click(
    x: int,
    y: int,
    hold_seconds: float = 0.125,
) -> float:
    prepare_roblox_click(
        x,
        y,
    )

    pydirectinput.mouseDown(
        button="left"
    )

    interruptible_sleep(
        hold_seconds
    )

    pydirectinput.mouseUp(
        button="left"
    )

    return time.perf_counter()


# ============================================================
# INPUT SAFETY
# ============================================================

def safe_release_all_inputs() -> None:
    for key in [
        "w",
        "a",
        "s",
        "d",
        "space",
        "o",
        "i",
        "1",
        "2",
        "3",
        "4",
        "5",
        "e",
        "q",
        "x",
    ]:
        try:
            pydirectinput.keyUp(
                key
            )
        except Exception:
            pass

    for button in [
        "left",
        "right",
    ]:
        try:
            pydirectinput.mouseUp(
                button=button
            )
        except Exception:
            pass

    try:
        right_mouse_up()
    except Exception:
        pass


# ============================================================
# KEYBOARD / CHARACTER
# ============================================================

def hold_key(
    key: str,
    duration: float,
) -> None:
    pydirectinput.keyDown(key)

    try:
        interruptible_sleep(
            duration
        )

    finally:
        pydirectinput.keyUp(key)


def tap_key(
    key: str,
    duration: float = 0.080,
) -> None:
    pydirectinput.keyDown(key)

    try:
        interruptible_sleep(
            duration
        )

    finally:
        pydirectinput.keyUp(key)


def strategy_key_tap(
    key: str,
) -> None:
    # Keep strategy keyboard behavior close to the existing bot.
    pydirectinput.keyDown(key)

    try:
        interruptible_sleep(
            0.015
        )

    finally:
        pydirectinput.keyUp(key)


def hold_s_and_jump() -> None:
    pydirectinput.keyDown("s")

    started = time.perf_counter()

    try:
        interruptible_sleep(
            JUMP_AFTER_S_SECONDS
        )

        pydirectinput.keyDown(
            "space"
        )

        try:
            interruptible_sleep(
                SPACE_HOLD_SECONDS
            )

        finally:
            pydirectinput.keyUp(
                "space"
            )

        elapsed = (
            time.perf_counter()
            - started
        )

        remaining = (
            S_HOLD_SECONDS
            - elapsed
        )

        if remaining > 0:
            interruptible_sleep(
                remaining
            )

    finally:
        pydirectinput.keyUp(
            "space"
        )

        pydirectinput.keyUp(
            "s"
        )


# ============================================================
# CAMERA
# ============================================================

def raw_relative_right_drag(
    dx: int,
    dy: int,
) -> None:
    set_cursor_position_verified(
        1200,
        500,
    )

    interruptible_sleep(0.20)

    relative_mouse_move(
        1,
        0,
    )

    interruptible_sleep(0.05)

    relative_mouse_move(
        -1,
        0,
    )

    interruptible_sleep(0.10)

    right_mouse_down()

    try:
        interruptible_sleep(0.20)

        relative_mouse_move(
            dx,
            dy,
        )

        interruptible_sleep(0.20)

    finally:
        right_mouse_up()

    interruptible_sleep(
        PITCH_BETWEEN_STEPS_SECONDS
    )


def force_max_zoom_out() -> None:
    print(
        "Forcing maximum zoom-out..."
    )

    for press in range(
        1,
        MAX_ZOOM_OUT_PRESSES + 1,
    ):
        tap_key(
            "o",
            ZOOM_KEY_HOLD_SECONDS,
        )

        print(
            f"  O press "
            f"{press}/"
            f"{MAX_ZOOM_OUT_PRESSES}"
        )

        interruptible_sleep(
            ZOOM_BETWEEN_PRESSES_SECONDS
        )

    interruptible_sleep(
        ZOOM_SETTLE_SECONDS
    )


def force_max_topdown_pitch() -> None:
    print(
        "Forcing maximum top-down pitch..."
    )

    cumulative = 0

    for step in range(
        1,
        PITCH_SATURATION_STEPS + 1,
    ):
        raw_relative_right_drag(
            0,
            PITCH_STEP_DY,
        )

        cumulative += PITCH_STEP_DY

        print(
            f"  pitch push "
            f"{step}/"
            f"{PITCH_SATURATION_STEPS} "
            f"(cumulative dy={cumulative})"
        )

    interruptible_sleep(
        PITCH_SETTLE_SECONDS
    )


# ============================================================
# DXCAM
# ============================================================

def latest_frame(
    camera,
    timeout: float = 2.0,
) -> np.ndarray:
    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:
        check_stop()

        frame = camera.get_latest_frame()

        if frame is not None:
            return frame

        time.sleep(0.01)

    raise RuntimeError(
        "DXcam did not provide a frame."
    )


def next_fresh_frame(
    camera,
    last_timestamp=None,
    timeout: float = 1.0,
):
    """
    Stable panel checks must not count the same stale DXcam
    image several times.
    """

    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:
        check_stop()

        result = camera.get_latest_frame(
            with_timestamp=True
        )

        if result is not None:
            frame, timestamp = result

            if (
                frame is not None
                and (
                    last_timestamp is None
                    or timestamp != last_timestamp
                )
            ):
                return frame, timestamp

        time.sleep(0.005)

    return None, last_timestamp


# ============================================================
# READY / RESTART BUTTON DETECTOR
# ============================================================

@dataclass(frozen=True)
class ButtonHit:
    center_x: int
    center_y: int
    width: int
    height: int
    area: int


def detect_green_button(
    frame: np.ndarray,
    roi: tuple[
        float,
        float,
        float,
        float,
    ],
) -> ButtonHit | None:
    height, width = frame.shape[:2]

    x1 = int(roi[0] * width)
    y1 = int(roi[1] * height)
    x2 = int(roi[2] * width)
    y2 = int(roi[3] * height)

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    mask = cv2.inRange(
        hsv,
        np.array(
            [35, 100, 100],
            dtype=np.uint8,
        ),
        np.array(
            [95, 255, 255],
            dtype=np.uint8,
        ),
    )

    (
        count,
        _labels,
        stats,
        centroids,
    ) = cv2.connectedComponentsWithStats(
        mask
    )

    if count <= 1:
        return None

    best_index = (
        1
        + int(
            np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA,
                ]
            )
        )
    )

    (
        _x,
        _y,
        box_width,
        box_height,
        area,
    ) = [
        int(value)
        for value in stats[best_index]
    ]

    if (
        area < 2500
        or box_width < 90
        or box_height < 35
    ):
        return None

    center_x, center_y = (
        centroids[
            best_index
        ]
    )

    return ButtonHit(
        center_x=int(
            round(
                x1 + center_x
            )
        ),
        center_y=int(
            round(
                y1 + center_y
            )
        ),
        width=box_width,
        height=box_height,
        area=area,
    )


def ready_hit(
    camera,
) -> ButtonHit | None:
    return detect_green_button(
        latest_frame(camera),
        READY_ROI,
    )


def restart_hit(
    camera,
) -> ButtonHit | None:
    return detect_green_button(
        latest_frame(camera),
        RESTART_ROI,
    )


def wait_for_button(
    camera,
    name: str,
    detector,
    timeout: float | None,
) -> ButtonHit:
    print(
        f"Waiting for {name}..."
    )

    started = time.perf_counter()
    last_status = started

    while True:
        check_stop()

        hit = detector(camera)

        if hit is not None:
            print(
                f"{name} FOUND at "
                f"({hit.center_x}, {hit.center_y})"
            )

            return hit

        now = time.perf_counter()

        if (
            timeout is not None
            and now - started >= timeout
        ):
            raise TimeoutError(
                f"Timed out waiting for {name}."
            )

        if now - last_status >= 30.0:
            print(
                f"Still waiting for {name}... "
                f"{int(now - started)}s"
            )

            last_status = now

        interruptible_sleep(
            DETECT_POLL_SECONDS
        )


def wait_until_button_gone(
    camera,
    name: str,
    detector,
    timeout: float,
) -> bool:
    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:
        check_stop()

        if detector(camera) is None:
            print(
                f"{name} disappeared."
            )

            return True

        interruptible_sleep(
            DETECT_POLL_SECONDS
        )

    return False


def click_ready(
    camera,
) -> float:
    hit = wait_for_button(
        camera,
        "READY",
        ready_hit,
        READY_WAIT_TIMEOUT,
    )

    print(
        "Clicking READY once..."
    )

    released_at = deliberate_click(
        hit.center_x,
        hit.center_y,
    )

    if not wait_until_button_gone(
        camera,
        "READY",
        ready_hit,
        READY_DISAPPEAR_TIMEOUT,
    ):
        raise RuntimeError(
            "READY was still visible after the click. "
            "Stopping instead of clicking repeatedly."
        )

    # The macro's first strategy sleep begins after READY
    # mouse-up, so only post-release verification time counts.
    return (
        time.perf_counter()
        - released_at
    )


def click_restart(
    camera,
) -> None:
    hit = wait_for_button(
        camera,
        "RESTART",
        restart_hit,
        RESTART_WAIT_TIMEOUT,
    )

    print(
        "Clicking Restart Match once..."
    )

    deliberate_click(
        hit.center_x,
        hit.center_y,
    )

    if not wait_until_button_gone(
        camera,
        "RESTART",
        restart_hit,
        RESTART_DISAPPEAR_TIMEOUT,
    ):
        raise RuntimeError(
            "Restart Match remained visible after the click."
        )


# ============================================================
# TOWER PANEL DETECTOR
# ============================================================

def largest_component(
    mask: np.ndarray,
):
    (
        count,
        _labels,
        stats,
        _centroids,
    ) = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    if count <= 1:
        return None

    best_index = (
        1
        + int(
            np.argmax(
                stats[
                    1:,
                    cv2.CC_STAT_AREA,
                ]
            )
        )
    )

    x = int(
        stats[
            best_index,
            cv2.CC_STAT_LEFT,
        ]
    )
    y = int(
        stats[
            best_index,
            cv2.CC_STAT_TOP,
        ]
    )
    width = int(
        stats[
            best_index,
            cv2.CC_STAT_WIDTH,
        ]
    )
    height = int(
        stats[
            best_index,
            cv2.CC_STAT_HEIGHT,
        ]
    )
    area = int(
        stats[
            best_index,
            cv2.CC_STAT_AREA,
        ]
    )

    return (
        x,
        y,
        width,
        height,
        area,
    )


def detect_question_button(
    frame: np.ndarray,
):
    (
        x1,
        y1,
        x2,
        y2,
    ) = QUESTION_ROI

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    mask = cv2.inRange(
        hsv,
        np.array(
            [85, 100, 100],
            dtype=np.uint8,
        ),
        np.array(
            [115, 255, 255],
            dtype=np.uint8,
        ),
    )

    component = largest_component(
        mask
    )

    if component is None:
        return False, None

    (
        x,
        y,
        width,
        height,
        area,
    ) = component

    found = (
        area >= QUESTION_MIN_AREA
        and width >= QUESTION_MIN_WIDTH
        and height >= QUESTION_MIN_HEIGHT
    )

    return (
        found,
        (
            x1 + x,
            y1 + y,
            width,
            height,
            area,
        ),
    )


def detect_sell_button(
    frame: np.ndarray,
):
    (
        x1,
        y1,
        x2,
        y2,
    ) = SELL_ROI

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    low_red = cv2.inRange(
        hsv,
        np.array(
            [0, 120, 120],
            dtype=np.uint8,
        ),
        np.array(
            [10, 255, 255],
            dtype=np.uint8,
        ),
    )

    high_red = cv2.inRange(
        hsv,
        np.array(
            [170, 120, 120],
            dtype=np.uint8,
        ),
        np.array(
            [179, 255, 255],
            dtype=np.uint8,
        ),
    )

    mask = cv2.bitwise_or(
        low_red,
        high_red,
    )

    component = largest_component(
        mask
    )

    if component is None:
        return False, None

    (
        x,
        y,
        width,
        height,
        area,
    ) = component

    found = (
        area >= SELL_MIN_AREA
        and width >= SELL_MIN_WIDTH
        and height >= SELL_MIN_HEIGHT
    )

    return (
        found,
        (
            x1 + x,
            y1 + y,
            width,
            height,
            area,
        ),
    )


@dataclass(frozen=True)
class TowerPanelSignal:
    question_found: bool
    sell_found: bool
    question_component: tuple | None
    sell_component: tuple | None


def tower_panel_signal(
    frame: np.ndarray,
) -> TowerPanelSignal:
    (
        question_found,
        question_component,
    ) = detect_question_button(
        frame
    )

    (
        sell_found,
        sell_component,
    ) = detect_sell_button(
        frame
    )

    return TowerPanelSignal(
        question_found=question_found,
        sell_found=sell_found,
        question_component=question_component,
        sell_component=sell_component,
    )


def wait_for_tower_panel_present(
    camera,
    timeout: float = PANEL_PRESENT_TIMEOUT,
) -> bool:
    """
    Debounced detection.

    Normal path:
        ? + Sell for 2 consecutive fresh frames.

    Tooltip-occlusion fallback:
        ? alone for 4 consecutive fresh frames.

    The second rule specifically handles the loadout stats
    tooltip covering part of the Sell button.
    """

    deadline = time.perf_counter() + timeout

    both_streak = 0
    question_streak = 0

    last_timestamp = None

    while time.perf_counter() < deadline:
        check_stop()

        frame, timestamp = next_fresh_frame(
            camera,
            last_timestamp=last_timestamp,
            timeout=0.20,
        )

        if frame is None:
            continue

        last_timestamp = timestamp

        signal = tower_panel_signal(
            frame
        )

        if signal.question_found:
            question_streak += 1
        else:
            question_streak = 0

        if (
            signal.question_found
            and signal.sell_found
        ):
            both_streak += 1
        else:
            both_streak = 0

        if (
            both_streak
            >= PANEL_BOTH_STREAK_REQUIRED
        ):
            return True

        if (
            question_streak
            >= PANEL_QUESTION_STREAK_REQUIRED
        ):
            return True

        interruptible_sleep(
            PANEL_SAMPLE_INTERVAL
        )

    return False


def wait_for_tower_panel_absent(
    camera,
    timeout: float = PANEL_ABSENT_TIMEOUT,
) -> bool:
    """
    Before a placement click, make sure the old selected-tower
    panel really disappeared after pressing 1/2/3/4.

    We use the ? button for absence because the Sell area can
    be affected by hover tooltips.
    """

    deadline = time.perf_counter() + timeout

    absent_streak = 0
    last_timestamp = None

    while time.perf_counter() < deadline:
        check_stop()

        frame, timestamp = next_fresh_frame(
            camera,
            last_timestamp=last_timestamp,
            timeout=0.20,
        )

        if frame is None:
            continue

        last_timestamp = timestamp

        signal = tower_panel_signal(
            frame
        )

        if not signal.question_found:
            absent_streak += 1
        else:
            absent_streak = 0

        if (
            absent_streak
            >= PANEL_ABSENT_STREAK_REQUIRED
        ):
            return True

        interruptible_sleep(
            PANEL_SAMPLE_INTERVAL
        )

    return False



# ============================================================
# TOWER LEVEL DETECTOR
# ============================================================

def scaled_roi(
    frame: np.ndarray,
    normalized_roi,
):
    height, width = frame.shape[:2]

    return (
        int(
            round(
                normalized_roi[0]
                * width
            )
        ),
        int(
            round(
                normalized_roi[1]
                * height
            )
        ),
        int(
            round(
                normalized_roi[2]
                * width
            )
        ),
        int(
            round(
                normalized_roi[3]
                * height
            )
        ),
    )


def detect_upgraded_level_segments(
    frame: np.ndarray,
):
    """
    Return progression segments that represent levels already
    purchased.

    TDS recolors the bar row based on affordability:

      affordable:
          purchased = bright green
          future    = dark green

      unaffordable:
          purchased = dark grey
          future    = lighter grey

    Combining the bright-green and dark-grey masks therefore
    detects the same level in both states.
    """

    (
        x1,
        y1,
        x2,
        y2,
    ) = scaled_roi(
        frame,
        LEVEL_BAR_ROI_NORM,
    )

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    green_mask = cv2.inRange(
        hsv,
        UPGRADED_GREEN_HSV_LOW,
        UPGRADED_GREEN_HSV_HIGH,
    )

    dark_grey_mask = cv2.inRange(
        hsv,
        UPGRADED_DARK_GREY_HSV_LOW,
        UPGRADED_DARK_GREY_HSV_HIGH,
    )

    mask = cv2.bitwise_or(
        green_mask,
        dark_grey_mask,
    )

    (
        count,
        _labels,
        stats,
        _centroids,
    ) = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    segments = []

    for index in range(
        1,
        count,
    ):
        (
            local_x,
            local_y,
            width,
            height,
            area,
        ) = [
            int(value)
            for value in stats[index]
        ]

        if not (
            LEVEL_SEGMENT_MIN_WIDTH
            <= width
            <= LEVEL_SEGMENT_MAX_WIDTH
        ):
            continue

        if not (
            LEVEL_SEGMENT_MIN_HEIGHT
            <= height
            <= LEVEL_SEGMENT_MAX_HEIGHT
        ):
            continue

        if (
            area
            < LEVEL_SEGMENT_MIN_AREA
        ):
            continue

        segments.append(
            (
                x1 + local_x,
                y1 + local_y,
                width,
                height,
                area,
            )
        )

    segments.sort(
        key=lambda item: item[0]
    )

    return segments


def detect_tower_level(
    frame: np.ndarray,
):
    (
        question_found,
        _question_component,
    ) = detect_question_button(
        frame
    )

    if not question_found:
        return None

    # Current level = number of purchased progression segments.
    return len(
        detect_upgraded_level_segments(
            frame
        )
    )


def detect_upgrade_affordable(
    frame: np.ndarray,
):
    """
    Return:
        True  -> upgrade card is green / affordable
        False -> upgrade card is grey / unaffordable
        None  -> no selected-tower panel

    This is intentionally independent from the level-bar color.
    """

    (
        question_found,
        _question_component,
    ) = detect_question_button(
        frame
    )

    if not question_found:
        return None

    (
        x1,
        y1,
        x2,
        y2,
    ) = scaled_roi(
        frame,
        UPGRADE_CARD_SAMPLE_ROI_NORM,
    )

    crop = frame[
        y1:y2,
        x1:x2,
    ]

    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv[
        :,
        :,
        1,
    ]

    median_saturation = float(
        np.median(
            saturation
        )
    )

    return (
        median_saturation
        >= UPGRADE_AFFORDABLE_MIN_MEDIAN_SATURATION
    )


def read_stable_tower_level(
    camera,
    timeout: float = LEVEL_READ_TIMEOUT,
):
    """
    Return a stable selected-tower level, or None when no
    stable selected-tower panel can be read.

    This detector is affordability-independent.
    """

    deadline = (
        time.perf_counter()
        + timeout
    )

    last_timestamp = None

    candidate = object()
    streak = 0

    while (
        time.perf_counter()
        < deadline
    ):
        check_stop()

        (
            frame,
            timestamp,
        ) = next_fresh_frame(
            camera,
            last_timestamp=last_timestamp,
            timeout=0.20,
        )

        if frame is None:
            continue

        last_timestamp = timestamp

        level = detect_tower_level(
            frame
        )

        if level is None:
            candidate = object()
            streak = 0

        elif level == candidate:
            streak += 1

        else:
            candidate = level
            streak = 1

        if (
            streak
            >= LEVEL_STABLE_FRAMES_REQUIRED
        ):
            return int(candidate)

        interruptible_sleep(
            PANEL_SAMPLE_INTERVAL
        )

    return None


def read_stable_upgrade_affordability(
    camera,
    timeout: float = AFFORDABILITY_READ_TIMEOUT,
):
    """
    Return stable True/False affordability, or None if the
    tower panel cannot be read.
    """

    deadline = (
        time.perf_counter()
        + timeout
    )

    last_timestamp = None

    candidate = object()
    streak = 0

    while (
        time.perf_counter()
        < deadline
    ):
        check_stop()

        (
            frame,
            timestamp,
        ) = next_fresh_frame(
            camera,
            last_timestamp=last_timestamp,
            timeout=0.20,
        )

        if frame is None:
            continue

        last_timestamp = timestamp

        affordable = detect_upgrade_affordable(
            frame
        )

        if affordable is None:
            candidate = object()
            streak = 0

        elif affordable == candidate:
            streak += 1

        else:
            candidate = affordable
            streak = 1

        if (
            streak
            >= AFFORDABILITY_STABLE_FRAMES_REQUIRED
        ):
            return bool(candidate)

        interruptible_sleep(
            PANEL_SAMPLE_INTERVAL
        )

    return None


def wait_until_upgrade_affordable(
    camera,
    tower,
    baseline_level: int,
    timeout: float = UPGRADE_AFFORDABLE_WAIT_TIMEOUT,
):
    """
    Wait instead of repeatedly pressing E while money is short.

    During the wait:
      - level is periodically re-read;
      - if it already advanced, return that new level;
      - if the panel closes, re-select the intended tower.
    """

    deadline = (
        time.perf_counter()
        + timeout
    )

    announced_wait = False

    while (
        time.perf_counter()
        < deadline
    ):
        check_stop()

        level = read_existing_selected_level(
            camera,
            timeout=0.55,
        )

        if level is None:
            level = select_tower_for_upgrade(
                camera,
                tower,
            )

        if level > baseline_level:
            return level

        if level < baseline_level:
            level = reread_level_after_suspect_regression(
                camera,
                tower,
                baseline_level,
            )

            if level > baseline_level:
                return level

            if level < baseline_level:
                interruptible_sleep(
                    0.15
                )
                continue

        affordable = (
            read_stable_upgrade_affordability(
                camera,
                timeout=0.70,
            )
        )

        if affordable is True:
            if announced_wait:
                print(
                    "  upgrade is now affordable."
                )

            return baseline_level

        if not announced_wait:
            print(
                "  upgrade unaffordable; "
                "waiting for money..."
            )

            announced_wait = True

        interruptible_sleep(
            0.25
        )

    raise RuntimeError(
        "Timed out waiting for upgrade to become affordable. "
        f"{tower.name} "
        f"#{tower.placement_number}, "
        f"level {baseline_level}."
    )


def move_cursor_to_neutral() -> None:
    set_cursor_position_verified(
        NEUTRAL_MOUSE_POSITION[0],
        NEUTRAL_MOUSE_POSITION[1],
    )

    interruptible_sleep(
        0.10
    )


# ============================================================
# MAP REGISTRATION
# ============================================================

def find_reference_image() -> Path:
    if REFERENCE_IMAGE_EXACT.is_file():
        return REFERENCE_IMAGE_EXACT

    if REFERENCE_IMAGE_LEGACY.is_file():
        print(
            "WARNING: using local legacy registration image."
        )
        print(
            "Copy it to assets/registration_reference.png "
            "to make the reference location portable."
        )
        return REFERENCE_IMAGE_LEGACY

    search_root = (
        BASE_DIR
        / "test_output"
        / "recording_anchor_absolute"
    )

    matches = list(
        search_root.glob(
            "*/03_FINAL_RECORDING_ANCHOR.png"
        )
    )

    if not matches:
        raise FileNotFoundError(
            "Could not find a map registration reference. "
            "Copy the original "
            "03_FINAL_RECORDING_ANCHOR.png to "
            f"{REFERENCE_IMAGE_EXACT}. "
            "See assets/README.md."
        )

    fallback = max(
        matches,
        key=lambda path: path.stat().st_mtime,
    )

    print(
        "WARNING: exact reference was not found."
    )
    print(
        "Using newest anchor instead:"
    )
    print(
        fallback
    )

    return fallback


def registration_edges(
    image: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    return cv2.Canny(
        gray,
        50,
        150,
    )


@dataclass(frozen=True)
class RegistrationMatch:
    name: str

    ref_x1: int
    ref_y1: int
    ref_x2: int
    ref_y2: int

    dx: int
    dy: int

    confidence: float


def match_registration_patch(
    reference: np.ndarray,
    current: np.ndarray,
    patch,
) -> RegistrationMatch:
    (
        name,
        x1,
        y1,
        x2,
        y2,
    ) = patch

    height, width = (
        reference.shape[:2]
    )

    template = reference[
        y1:y2,
        x1:x2,
    ]

    template_edges = registration_edges(
        template
    )

    sx1 = max(
        0,
        x1 - REGISTRATION_SEARCH_MARGIN,
    )
    sy1 = max(
        0,
        y1 - REGISTRATION_SEARCH_MARGIN,
    )
    sx2 = min(
        width,
        x2 + REGISTRATION_SEARCH_MARGIN,
    )
    sy2 = min(
        height,
        y2 + REGISTRATION_SEARCH_MARGIN,
    )

    search = current[
        sy1:sy2,
        sx1:sx2,
    ]

    search_edges = registration_edges(
        search
    )

    result = cv2.matchTemplate(
        search_edges,
        template_edges,
        cv2.TM_CCOEFF_NORMED,
    )

    (
        _min_value,
        max_value,
        _min_location,
        max_location,
    ) = cv2.minMaxLoc(
        result
    )

    matched_x = (
        sx1
        + max_location[0]
    )
    matched_y = (
        sy1
        + max_location[1]
    )

    dx = matched_x - x1
    dy = matched_y - y1

    return RegistrationMatch(
        name=name,

        ref_x1=x1,
        ref_y1=y1,
        ref_x2=x2,
        ref_y2=y2,

        dx=int(dx),
        dy=int(dy),

        confidence=float(
            max_value
        ),
    )


def valid_registration_match(
    match: RegistrationMatch,
) -> bool:
    return (
        match.confidence
        >= REGISTRATION_MIN_PATCH_CONFIDENCE
        and abs(match.dx)
        <= REGISTRATION_MAX_SHIFT
        and abs(match.dy)
        <= REGISTRATION_MAX_SHIFT
    )


def median_int(values) -> int:
    values = sorted(
        int(value)
        for value in values
    )

    if not values:
        raise ValueError(
            "No values for median."
        )

    count = len(values)
    middle = count // 2

    if count % 2:
        return values[middle]

    return int(
        round(
            (
                values[middle - 1]
                + values[middle]
            )
            / 2.0
        )
    )


def save_registration_debug(
    current: np.ndarray,
    matches: list[RegistrationMatch],
    final_dx: int,
    final_dy: int,
    run_number: int,
) -> Path:
    debug = current.copy()

    for match in matches:
        accepted = (
            valid_registration_match(
                match
            )
        )

        cv2.rectangle(
            debug,
            (
                match.ref_x1,
                match.ref_y1,
            ),
            (
                match.ref_x2,
                match.ref_y2,
            ),
            (
                255,
                255,
                255,
            ),
            2,
        )

        color = (
            (0, 255, 0)
            if accepted
            else (0, 0, 255)
        )

        cv2.rectangle(
            debug,
            (
                match.ref_x1
                + match.dx,
                match.ref_y1
                + match.dy,
            ),
            (
                match.ref_x2
                + match.dx,
                match.ref_y2
                + match.dy,
            ),
            color,
            2,
        )

    cv2.putText(
        debug,
        (
            f"FINAL dx={final_dx:+d} "
            f"dy={final_dy:+d}"
        ),
        (
            40,
            75,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (
            255,
            255,
            255,
        ),
        2,
        cv2.LINE_AA,
    )

    timestamp = (
        datetime.now()
        .strftime(
            "%Y%m%d_%H%M%S"
        )
    )

    folder = (
        REGISTRATION_OUTPUT_ROOT
        / timestamp
    )

    folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        folder
        / (
            f"run_{run_number:04d}"
            f"_registration.png"
        )
    )

    cv2.imwrite(
        str(path),
        debug,
    )

    return path


def measure_anchor_shift(
    camera,
    run_number: int,
) -> tuple[int, int, float]:
    reference_path = (
        find_reference_image()
    )

    reference = cv2.imread(
        str(reference_path),
        cv2.IMREAD_COLOR,
    )

    if reference is None:
        raise RuntimeError(
            "Could not load registration reference."
        )

    interruptible_sleep(0.35)

    current = latest_frame(
        camera
    )

    if (
        reference.shape[:2]
        != current.shape[:2]
    ):
        raise RuntimeError(
            "Reference/current image dimensions differ."
        )

    matches: list[
        RegistrationMatch
    ] = []

    print()
    print(
        "Measuring map alignment..."
    )

    for patch in REGISTRATION_PATCHES:
        match = match_registration_patch(
            reference,
            current,
            patch,
        )

        matches.append(
            match
        )

        status = (
            "OK"
            if valid_registration_match(
                match
            )
            else "REJECT"
        )

        print(
            f"  {match.name:<22} "
            f"dx={match.dx:+3d} "
            f"dy={match.dy:+3d} "
            f"confidence="
            f"{match.confidence:.3f} "
            f"[{status}]"
        )

    valid = [
        match
        for match in matches
        if valid_registration_match(
            match
        )
    ]

    primary = next(
        (
            match
            for match in matches
            if match.name
            == "left_rock_primary"
        ),
        None,
    )

    final_matches: list[
        RegistrationMatch
    ] = []

    if (
        primary is not None
        and valid_registration_match(
            primary
        )
        and primary.confidence
        >= REGISTRATION_MIN_PRIMARY_CONFIDENCE
    ):
        final_matches = [
            match
            for match in valid
            if (
                abs(
                    match.dx
                    - primary.dx
                )
                <= REGISTRATION_INLIER_TOLERANCE
                and abs(
                    match.dy
                    - primary.dy
                )
                <= REGISTRATION_INLIER_TOLERANCE
            )
        ]

        if primary not in final_matches:
            final_matches.append(
                primary
            )

    else:
        if len(valid) < 2:
            raise RuntimeError(
                "Visual registration failed: "
                "not enough reliable terrain matches."
            )

        rough_dx = median_int(
            match.dx
            for match in valid
        )
        rough_dy = median_int(
            match.dy
            for match in valid
        )

        final_matches = [
            match
            for match in valid
            if (
                abs(
                    match.dx
                    - rough_dx
                )
                <= REGISTRATION_INLIER_TOLERANCE
                and abs(
                    match.dy
                    - rough_dy
                )
                <= REGISTRATION_INLIER_TOLERANCE
            )
        ]

        if len(final_matches) < 2:
            raise RuntimeError(
                "Visual registration failed: "
                "terrain patches disagreed."
            )

    final_dx = median_int(
        match.dx
        for match in final_matches
    )
    final_dy = median_int(
        match.dy
        for match in final_matches
    )

    confidence = sum(
        match.confidence
        for match in final_matches
    ) / len(final_matches)

    if (
        primary is None
        or primary not in final_matches
    ):
        if (
            confidence
            < REGISTRATION_MIN_CONSENSUS_CONFIDENCE
        ):
            raise RuntimeError(
                "Visual registration confidence was too low."
            )

    if (
        abs(final_dx)
        > REGISTRATION_MAX_SHIFT
        or abs(final_dy)
        > REGISTRATION_MAX_SHIFT
    ):
        raise RuntimeError(
            "Unsafe registration shift: "
            f"dx={final_dx}, dy={final_dy}"
        )

    debug_path = save_registration_debug(
        current,
        matches,
        final_dx,
        final_dy,
        run_number,
    )

    print()
    print(
        "MAP ALIGNMENT ACCEPTED"
    )
    print(
        f"  dx = {final_dx:+d} px"
    )
    print(
        f"  dy = {final_dy:+d} px"
    )
    print(
        f"  confidence = "
        f"{confidence:.3f}"
    )
    print(
        f"  supporting patches = "
        f"{len(final_matches)}"
    )
    print(
        f"  debug: {debug_path}"
    )
    print()

    return (
        final_dx,
        final_dy,
        confidence,
    )


# ============================================================
# ONE-TIME INITIAL SETUP
# ============================================================

def initial_anchor_setup(
    camera,
) -> None:
    print()
    print(
        "INITIAL SETUP"
    )

    print(
        "Verifying READY before moving..."
    )

    if ready_hit(camera) is None:
        raise RuntimeError(
            "READY was not detected at startup."
        )

    print(
        "READY FOUND."
    )
    print(
        "READY will not be clicked yet."
    )
    print()

    print(
        f"Holding A for "
        f"{A_HOLD_SECONDS:.3f}s..."
    )

    hold_key(
        "a",
        A_HOLD_SECONDS,
    )

    interruptible_sleep(
        AFTER_A_PAUSE_SECONDS
    )

    print(
        "Holding S + jump..."
    )

    hold_s_and_jump()

    interruptible_sleep(
        AFTER_S_PAUSE_SECONDS
    )

    print(
        f"Holding D for "
        f"{D_HOLD_SECONDS:.3f}s..."
    )

    hold_key(
        "d",
        D_HOLD_SECONDS,
    )

    print(
        f"Waiting "
        f"{CAMERA_RECOVERY_SECONDS:.1f}s "
        f"for camera recovery..."
    )

    interruptible_sleep(
        CAMERA_RECOVERY_SECONDS
    )

    force_max_zoom_out()
    force_max_topdown_pitch()

    print()
    print(
        "ABSOLUTE ANCHOR COMPLETE."
    )
    print(
        "Zoom: maximum out"
    )
    print(
        "Pitch: maximum top-down"
    )
    print(
        "I presses: 0"
    )
    print()


# ============================================================
# STRATEGY PARSER
# ============================================================

@dataclass(frozen=True)
class Action:
    kind: str
    args: tuple


SLEEP_RE = re.compile(
    r"^Sleep,\s*(\d+)\s*$",
    re.IGNORECASE,
)

SEND_RE = re.compile(
    r"^Send,\s*\{([^}]+)\}\s*$",
    re.IGNORECASE,
)

CLICK_RE = re.compile(
    r"^Click,\s*(\d+),\s*(\d+)\s+"
    r"(Left|Right),\s*(Down|Up)\s*$",
    re.IGNORECASE,
)


def parse_strategy(
    text: str,
) -> list[Action]:
    actions: list[Action] = []

    for (
        line_number,
        raw,
    ) in enumerate(
        text.splitlines(),
        start=1,
    ):
        line = raw.strip()

        if (
            not line
            or line.startswith(";")
        ):
            continue

        match = SLEEP_RE.match(
            line
        )

        if match:
            actions.append(
                Action(
                    "sleep",
                    (
                        int(
                            match.group(1)
                        ),
                    ),
                )
            )
            continue

        match = SEND_RE.match(
            line
        )

        if match:
            actions.append(
                Action(
                    "send",
                    (
                        match.group(1)
                        .lower(),
                    ),
                )
            )
            continue

        match = CLICK_RE.match(
            line
        )

        if match:
            actions.append(
                Action(
                    "mouse",
                    (
                        int(
                            match.group(1)
                        ),
                        int(
                            match.group(2)
                        ),
                        match.group(3)
                        .lower(),
                        match.group(4)
                        .lower(),
                    ),
                )
            )
            continue

        raise ValueError(
            f"Unsupported strategy line "
            f"{line_number}: {raw!r}"
        )

    return actions


# ============================================================
# PERSISTENT PLACEMENT OFFSETS
# ============================================================

def load_placement_offsets():
    offsets = {
        int(number): (
            int(value[0]),
            int(value[1]),
        )
        for (
            number,
            value,
        ) in SEED_PLACEMENT_OFFSETS.items()
    }

    if not PLACEMENT_OFFSETS_PATH.exists():
        return offsets

    try:
        data = json.loads(
            PLACEMENT_OFFSETS_PATH.read_text(
                encoding="utf-8"
            )
        )

        for (
            key,
            value,
        ) in data.items():
            number = int(key)

            if (
                isinstance(value, list)
                and len(value) == 2
            ):
                offsets[number] = (
                    int(value[0]),
                    int(value[1]),
                )

    except Exception as exc:
        print(
            "WARNING: Could not read "
            "placement_offsets.json:"
        )
        print(
            f"  {exc}"
        )

    return offsets


def save_placement_offsets(
    offsets,
) -> None:
    data = {
        str(number): [
            int(offset[0]),
            int(offset[1]),
        ]
        for (
            number,
            offset,
        ) in sorted(
            offsets.items()
        )
    }

    temporary = (
        PLACEMENT_OFFSETS_PATH
        .with_suffix(
            ".json.tmp"
        )
    )

    temporary.write_text(
        json.dumps(
            data,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        PLACEMENT_OFFSETS_PATH
    )


def placement_candidate_offsets(
    placement_number: int,
    learned_offsets,
):
    preferred = (
        learned_offsets.get(
            placement_number,
            (0, 0),
        )
    )

    candidates = [
        preferred,
        (0, 0),
        *PLACEMENT_RETRY_OFFSETS,
    ]

    unique = []

    for candidate in candidates:
        candidate = (
            int(candidate[0]),
            int(candidate[1]),
        )

        if candidate not in unique:
            unique.append(
                candidate
            )

    return unique


# ============================================================
# TOWER STATE
# ============================================================

@dataclass
class TowerState:
    placement_number: int
    hotkey: str
    name: str

    recorded_x: int
    recorded_y: int

    registered_x: int
    registered_y: int

    actual_x: int
    actual_y: int

    local_offset_x: int
    local_offset_y: int

    level: int | None = None


@dataclass
class PlacementAttempt:
    hotkey: str

    recorded_x: int
    recorded_y: int

    registered_x: int
    registered_y: int

    initial_offset_x: int
    initial_offset_y: int


def distance_to_tower(
    tower: TowerState,
    x: int,
    y: int,
) -> float:
    return math.hypot(
        tower.actual_x - x,
        tower.actual_y - y,
    )


def nearest_tower(
    towers: list[TowerState],
    x: int,
    y: int,
):
    if not towers:
        return None, None

    tower = min(
        towers,
        key=lambda item: (
            distance_to_tower(
                item,
                x,
                y,
            )
        ),
    )

    distance = (
        distance_to_tower(
            tower,
            x,
            y,
        )
    )

    if (
        distance
        > SELECTION_MATCH_DISTANCE
    ):
        return None, distance

    return tower, distance


# ============================================================
# PLACEMENT SCREENSHOTS
# ============================================================

def save_placement_frame(
    camera,
    run_number: int,
    placement_number: int,
    label: str,
) -> Path | None:
    try:
        frame = latest_frame(
            camera
        )

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        folder = (
            PLACEMENT_OUTPUT_ROOT
            / timestamp
        )

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            folder
            / (
                f"run_{run_number:04d}_"
                f"placement_{placement_number:02d}_"
                f"{label}.png"
            )
        )

        cv2.imwrite(
            str(path),
            frame,
        )

        return path

    except Exception:
        return None


# ============================================================
# VERIFIED PLACEMENT
# ============================================================

def verify_or_retry_placement(
    camera,
    attempt: PlacementAttempt,
    run_number: int,
    placement_number: int,
    learned_offsets,
):
    """
    The initial placement click has already occurred.

    If it did not open a tower panel, search nearby. The
    successful local offset is persisted so the next run starts
    with that location immediately.
    """

    initial_offset = (
        attempt.initial_offset_x,
        attempt.initial_offset_y,
    )

    print()
    print(
        f"PLACEMENT {placement_number}: "
        f"{TOWER_NAMES[attempt.hotkey]} "
        f"(key {attempt.hotkey})"
    )

    print(
        f"  recorded = "
        f"({attempt.recorded_x}, "
        f"{attempt.recorded_y})"
    )

    print(
        f"  registered = "
        f"({attempt.registered_x}, "
        f"{attempt.registered_y})"
    )

    print(
        f"  first local offset = "
        f"({initial_offset[0]:+d},"
        f"{initial_offset[1]:+d})"
    )

    if wait_for_tower_panel_present(
        camera
    ):
        successful_offset = (
            initial_offset
        )

        print(
            "  VERIFIED on first attempt."
        )

    else:
        failure_path = save_placement_frame(
            camera,
            run_number,
            placement_number,
            "first_attempt_failed",
        )

        print(
            "  First placement was NOT verified."
        )

        if failure_path is not None:
            print(
                f"  snapshot: {failure_path}"
            )

        successful_offset = None

        candidates = (
            placement_candidate_offsets(
                placement_number,
                learned_offsets,
            )
        )

        for retry_number, (
            offset_x,
            offset_y,
        ) in enumerate(
            candidates,
            start=1,
        ):
            if (
                offset_x,
                offset_y,
            ) == initial_offset:
                continue

            check_stop()

            retry_x = (
                attempt.registered_x
                + offset_x
            )

            retry_y = (
                attempt.registered_y
                + offset_y
            )

            print(
                f"  retry {retry_number}: "
                f"offset=({offset_x:+d},"
                f"{offset_y:+d}) "
                f"click=({retry_x},"
                f"{retry_y})"
            )

            deliberate_click(
                retry_x,
                retry_y,
                hold_seconds=(
                    PLACEMENT_RETRY_CLICK_HOLD
                ),
            )

            if wait_for_tower_panel_present(
                camera
            ):
                successful_offset = (
                    offset_x,
                    offset_y,
                )

                print(
                    "  VERIFIED with retry offset "
                    f"({offset_x:+d},"
                    f"{offset_y:+d})."
                )

                break

        if successful_offset is None:
            final_path = (
                save_placement_frame(
                    camera,
                    run_number,
                    placement_number,
                    "ALL_RETRIES_FAILED",
                )
            )

            raise RuntimeError(
                "Tower placement could not be verified. "
                f"Tower={TOWER_NAMES[attempt.hotkey]}, "
                f"placement={placement_number}, "
                f"registered position="
                f"({attempt.registered_x}, "
                f"{attempt.registered_y}), "
                f"failure screenshot="
                f"{final_path}"
            )

    learned_offsets[
        placement_number
    ] = successful_offset

    save_placement_offsets(
        learned_offsets
    )

    actual_x = (
        attempt.registered_x
        + successful_offset[0]
    )

    actual_y = (
        attempt.registered_y
        + successful_offset[1]
    )

    move_cursor_to_neutral()

    level = read_stable_tower_level(
        camera
    )

    print(
        f"  learned offset saved: "
        f"({successful_offset[0]:+d},"
        f"{successful_offset[1]:+d})"
    )

    if level is not None:
        print(
            f"  detected initial level: "
            f"{level}"
        )

    print()

    return (
        successful_offset,
        actual_x,
        actual_y,
        level,
    )


# ============================================================
# VERIFIED TOWER SELECTION
# ============================================================

def read_existing_selected_level(
    camera,
    timeout: float = 0.65,
):
    """
    Read the panel that is ALREADY open.

    Important: this function never clicks the tower. Re-clicking
    a tower whose panel is already visible was causing the GUI
    to close/reopen between upgrades.
    """

    move_cursor_to_neutral()

    return read_stable_tower_level(
        camera,
        timeout=timeout,
    )


def select_tower_for_upgrade(
    camera,
    tower: TowerState,
):
    """
    Select the intended remembered tower ONLY when there is no
    readable tower panel already open.

    The normal macro selection click usually leaves the correct
    panel open, so the common case performs no extra click.
    """

    existing_level = (
        read_existing_selected_level(
            camera,
            timeout=0.55,
        )
    )

    if existing_level is not None:
        return existing_level

    print(
        "  tower panel not open; "
        "re-selecting remembered tower..."
    )

    for (
        offset_x,
        offset_y,
    ) in SELECTION_RETRY_OFFSETS:
        select_x = (
            tower.actual_x
            + offset_x
        )
        select_y = (
            tower.actual_y
            + offset_y
        )

        deliberate_click(
            select_x,
            select_y,
            hold_seconds=0.090,
        )

        move_cursor_to_neutral()

        level = read_stable_tower_level(
            camera,
            timeout=0.80,
        )

        if level is not None:
            return level

    raise RuntimeError(
        "Could not select intended tower before upgrade: "
        f"{tower.name} #{tower.placement_number} "
        f"near ({tower.actual_x}, {tower.actual_y})."
    )


def reread_level_after_suspect_regression(
    camera,
    tower: TowerState,
    minimum_expected_level: int,
):
    """
    A tower's true level cannot spontaneously decrease during
    this strategy.

    During an upgrade animation the bright progression bars can
    briefly disappear/repaint. That previously produced a false
    Level 0 reading immediately after a real upgrade.

    Never accept a lower level from one read. Let the UI settle
    and retry several times. Re-select only if the panel becomes
    unreadable.
    """

    for check_number in range(
        1,
        UPGRADE_REGRESSION_RECHECKS + 1,
    ):
        interruptible_sleep(
            UPGRADE_TRANSIENT_SETTLE_SECONDS
        )

        level = read_existing_selected_level(
            camera,
            timeout=0.70,
        )

        if level is None:
            level = select_tower_for_upgrade(
                camera,
                tower,
            )

        if level >= minimum_expected_level:
            return level

        print(
            f"  ignoring transient lower-level read "
            f"{level} "
            f"(recheck "
            f"{check_number}/"
            f"{UPGRADE_REGRESSION_RECHECKS})"
        )

    return level


# ============================================================
# VERIFIED EXACTLY-ONCE UPGRADE
# ============================================================

def perform_verified_upgrade(
    camera,
    tower: TowerState,
):
    """
    Execute ONE recorded E safely.

    Normal flow:
        use the panel already open from the macro selection
        -> read current level
        -> E once
        -> wait for exactly +1 level

    If E failed because money was insufficient:
        panel remains at baseline
        -> wait
        -> re-read baseline
        -> E once again

    Before every retry, the level is read again. Therefore a
    delayed successful upgrade cannot cause a second purchase.

    Lower readings during GUI repaint are treated as transient;
    a tower is never assumed to have actually lost levels.
    """

    started = time.perf_counter()

    current_level = (
        select_tower_for_upgrade(
            camera,
            tower,
        )
    )

    # We previously verified this tower at a higher level.
    # A lower read is therefore UI transition noise until
    # repeated checks prove otherwise.
    if (
        tower.level is not None
        and current_level < tower.level
    ):
        current_level = (
            reread_level_after_suspect_regression(
                camera,
                tower,
                tower.level,
            )
        )

        if current_level < tower.level:
            raise RuntimeError(
                "Tower level repeatedly appeared to regress. "
                f"{tower.name} "
                f"#{tower.placement_number}: "
                f"previously verified={tower.level}, "
                f"now read={current_level}."
            )

    tower.level = current_level

    cap = STRATEGY_LEVEL_CAPS.get(
        tower.hotkey
    )

    print()
    print(
        "UPGRADE REQUEST:"
    )
    print(
        f"  tower = "
        f"{tower.name} "
        f"#{tower.placement_number}"
    )
    print(
        f"  current level = "
        f"{current_level}"
    )

    if (
        cap is not None
        and current_level >= cap
    ):
        print(
            f"  SKIPPED: strategy cap "
            f"for {tower.name} is "
            f"level {cap}."
        )
        print()

        return (
            time.perf_counter()
            - started,
            0,
            current_level,
            True,
        )

    baseline = current_level
    desired = baseline + 1

    deadline = (
        time.perf_counter()
        + UPGRADE_TOTAL_TIMEOUT
    )

    attempts = 0

    while (
        time.perf_counter()
        < deadline
    ):
        check_stop()

        # ----------------------------------------------------
        # SAFETY READ BEFORE ANY E
        # ----------------------------------------------------

        observed = (
            read_existing_selected_level(
                camera,
                timeout=0.70,
            )
        )

        if observed is None:
            observed = (
                select_tower_for_upgrade(
                    camera,
                    tower,
                )
            )

        if observed < baseline:
            observed = (
                reread_level_after_suspect_regression(
                    camera,
                    tower,
                    baseline,
                )
            )

        if observed == desired:
            tower.level = observed

            print(
                f"  VERIFIED level "
                f"{baseline} -> "
                f"{observed}"
            )
            print()

            return (
                time.perf_counter()
                - started,
                attempts,
                observed,
                False,
            )

        if observed > desired:
            raise RuntimeError(
                "Tower advanced farther than one level "
                "during a verified upgrade. "
                f"{tower.name} "
                f"#{tower.placement_number}: "
                f"baseline={baseline}, "
                f"observed={observed}."
            )

        if observed < baseline:
            raise RuntimeError(
                "Tower level stayed below its previously "
                "verified baseline after repeated checks. "
                f"{tower.name} "
                f"#{tower.placement_number}: "
                f"baseline={baseline}, "
                f"observed={observed}."
            )

        # Only baseline remains.
        #
        # First wait for the upgrade card to become green.
        # This avoids repeatedly pressing E while the economy
        # is short, and it works even though the level bars
        # themselves turn grey in the unaffordable state.
        affordable_result = (
            wait_until_upgrade_affordable(
                camera,
                tower,
                baseline,
            )
        )

        if affordable_result == desired:
            tower.level = desired

            print(
                f"  VERIFIED level "
                f"{baseline} -> "
                f"{desired}"
            )
            print()

            return (
                time.perf_counter()
                - started,
                attempts,
                desired,
                False,
            )

        if affordable_result > desired:
            raise RuntimeError(
                "Tower advanced farther than one level "
                "while waiting for affordability. "
                f"{tower.name} "
                f"#{tower.placement_number}: "
                f"baseline={baseline}, "
                f"observed={affordable_result}."
            )

        attempts += 1

        print(
            f"  E attempt {attempts}: "
            f"trying level "
            f"{baseline} -> "
            f"{desired}"
        )

        strategy_key_tap(
            "e"
        )

        # ----------------------------------------------------
        # WAIT FOR UI TO SHOW THE TARGET LEVEL
        # ----------------------------------------------------

        change_deadline = (
            time.perf_counter()
            + UPGRADE_LEVEL_CHANGE_TIMEOUT
        )

        while (
            time.perf_counter()
            < change_deadline
        ):
            check_stop()

            new_level = (
                read_existing_selected_level(
                    camera,
                    timeout=0.55,
                )
            )

            if new_level is None:
                # The panel may be repainting. Do not click it
                # immediately; give it a moment first.
                interruptible_sleep(
                    0.10
                )
                continue

            if new_level == desired:
                tower.level = new_level

                print(
                    f"  VERIFIED level "
                    f"{baseline} -> "
                    f"{new_level}"
                )
                print()

                return (
                    time.perf_counter()
                    - started,
                    attempts,
                    new_level,
                    False,
                )

            if new_level > desired:
                raise RuntimeError(
                    "Tower advanced farther than one "
                    "level after E. "
                    f"{tower.name} "
                    f"#{tower.placement_number}: "
                    f"baseline={baseline}, "
                    f"observed={new_level}."
                )

            if new_level < baseline:
                # Critical change from the previous build:
                # do NOT treat this as a real regression and
                # do NOT immediately retry E. Upgrade animation
                # can temporarily hide lit bars.
                print(
                    f"  transient bar read "
                    f"{new_level}; "
                    f"waiting for UI repaint..."
                )

                interruptible_sleep(
                    UPGRADE_TRANSIENT_SETTLE_SECONDS
                )

                continue

            # new_level == baseline:
            # keep watching until the change timeout expires.
            # The UI can take a little time to repaint.

        # ----------------------------------------------------
        # CHANGE TIMEOUT: DETERMINE WHETHER E ACTUALLY FAILED
        # ----------------------------------------------------

        settled = (
            read_existing_selected_level(
                camera,
                timeout=0.80,
            )
        )

        if settled is None:
            settled = (
                select_tower_for_upgrade(
                    camera,
                    tower,
                )
            )

        if settled < baseline:
            settled = (
                reread_level_after_suspect_regression(
                    camera,
                    tower,
                    baseline,
                )
            )

        if settled == desired:
            tower.level = settled

            print(
                f"  VERIFIED level "
                f"{baseline} -> "
                f"{settled}"
            )
            print()

            return (
                time.perf_counter()
                - started,
                attempts,
                settled,
                False,
            )

        if settled > desired:
            raise RuntimeError(
                "Tower advanced farther than expected "
                "after upgrade-settle check. "
                f"{tower.name} "
                f"#{tower.placement_number}: "
                f"baseline={baseline}, "
                f"observed={settled}."
            )

        if settled < baseline:
            raise RuntimeError(
                "Tower level repeatedly appeared lower "
                "than its verified baseline after E. "
                f"{tower.name} "
                f"#{tower.placement_number}: "
                f"baseline={baseline}, "
                f"observed={settled}."
            )

        # Still exactly baseline. Now we have evidence that the
        # E did not purchase the upgrade (usually insufficient
        # money). Wait, then loop. The next iteration re-reads
        # the level BEFORE another E.
        print(
            "  level did not change; "
            "waiting before safe retry..."
        )

        interruptible_sleep(
            UPGRADE_RETRY_DELAY
        )

    raise RuntimeError(
        "Timed out waiting for a verified upgrade. "
        f"{tower.name} "
        f"#{tower.placement_number}, "
        f"level {baseline} -> {desired}."
    )


# ============================================================
# STRATEGY PLAYER
# ============================================================

def play_strategy(
    camera,
    actions: list[Action],
    run_number: int,
    learned_offsets,
    initial_elapsed: float = 0.0,
    map_dx: int = 0,
    map_dy: int = 0,
) -> None:
    print(
        "Strategy started."
    )

    started = (
        time.perf_counter()
    )

    # Only normal detector overhead is allowed to shorten later
    # macro sleeps. Long retries/waits are intentionally not
    # fully erased.
    timing_credit = max(
        0.0,
        initial_elapsed,
    )

    placement_armed_key: str | None = None
    placement_in_progress: PlacementAttempt | None = None

    placement_number = 0

    active_mouse_x = None
    active_mouse_y = None

    mouse_button_held = False

    towers: list[
        TowerState
    ] = []

    selected_tower: TowerState | None = None

    for action in actions:
        check_stop()

        # ----------------------------------------------------
        # SLEEP
        # ----------------------------------------------------

        if action.kind == "sleep":
            (
                milliseconds,
            ) = action.args

            seconds = (
                milliseconds
                / 1000.0
            )

            if mouse_button_held:
                # Preserve recorded click hold exactly.
                interruptible_sleep(
                    seconds
                )

            else:
                consumed = min(
                    seconds,
                    timing_credit,
                )

                timing_credit -= (
                    consumed
                )

                interruptible_sleep(
                    seconds
                    - consumed
                )

        # ----------------------------------------------------
        # KEY
        # ----------------------------------------------------

        elif action.kind == "send":
            (
                key,
            ) = action.args

            # Tower hotkey: arm the next left click as a
            # placement.
            if key in PLACEMENT_HOTKEYS:
                if (
                    placement_armed_key
                    is not None
                ):
                    raise RuntimeError(
                        "A new tower hotkey was pressed "
                        "before the prior placement "
                        "was resolved."
                    )

                strategy_key_tap(
                    key
                )

                placement_armed_key = (
                    key
                )

            # Every E is now state-aware.
            elif key == "e":
                if selected_tower is None:
                    raise RuntimeError(
                        "Recorded E encountered without "
                        "an identifiable selected tower."
                    )

                (
                    elapsed,
                    attempts,
                    final_level,
                    skipped,
                ) = perform_verified_upgrade(
                    camera,
                    selected_tower,
                )

                # Compensate only normal visual-check overhead.
                # If money was late and multiple E attempts were
                # needed, do NOT erase that entire delay.
                timing_credit += min(
                    elapsed,
                    UPGRADE_TIMING_CREDIT_CAP,
                )

            else:
                strategy_key_tap(
                    key
                )

        # ----------------------------------------------------
        # MOUSE
        # ----------------------------------------------------

        elif action.kind == "mouse":
            (
                x,
                y,
                button,
                state,
            ) = action.args

            registered_x = (
                x
                + map_dx
            )

            registered_y = (
                y
                + map_dy
            )

            # ------------------------------------------------
            # MOUSE DOWN
            # ------------------------------------------------

            if state == "down":
                is_placement = (
                    button == "left"
                    and placement_armed_key
                    is not None
                )

                if is_placement:
                    # Make sure the previous tower panel closed
                    # and Roblox entered placement mode.
                    absence_started = (
                        time.perf_counter()
                    )

                    panel_absent = (
                        wait_for_tower_panel_absent(
                            camera
                        )
                    )

                    timing_credit += min(
                        time.perf_counter()
                        - absence_started,
                        PLACEMENT_TIMING_CREDIT_CAP,
                    )

                    if not panel_absent:
                        raise RuntimeError(
                            "Tower panel remained open "
                            "after pressing placement "
                            f"hotkey "
                            f"{placement_armed_key}."
                        )

                    placement_number += 1

                    candidates = (
                        placement_candidate_offsets(
                            placement_number,
                            learned_offsets,
                        )
                    )

                    (
                        first_offset_x,
                        first_offset_y,
                    ) = candidates[0]

                    active_mouse_x = (
                        registered_x
                        + first_offset_x
                    )

                    active_mouse_y = (
                        registered_y
                        + first_offset_y
                    )

                    placement_in_progress = (
                        PlacementAttempt(
                            hotkey=(
                                placement_armed_key
                            ),
                            recorded_x=x,
                            recorded_y=y,
                            registered_x=(
                                registered_x
                            ),
                            registered_y=(
                                registered_y
                            ),
                            initial_offset_x=(
                                first_offset_x
                            ),
                            initial_offset_y=(
                                first_offset_y
                            ),
                        )
                    )

                    print(
                        f"Placing tower "
                        f"{placement_number}/19 "
                        f"({TOWER_NAMES[placement_armed_key]}) "
                        f"first try at "
                        f"({active_mouse_x}, "
                        f"{active_mouse_y})..."
                    )

                else:
                    active_mouse_x = (
                        registered_x
                    )

                    active_mouse_y = (
                        registered_y
                    )

                prepare_roblox_click(
                    active_mouse_x,
                    active_mouse_y,
                )

                pydirectinput.mouseDown(
                    button=button
                )

                mouse_button_held = True

            # ------------------------------------------------
            # MOUSE UP
            # ------------------------------------------------

            else:
                if (
                    active_mouse_x
                    is None
                    or active_mouse_y
                    is None
                ):
                    active_mouse_x = (
                        registered_x
                    )

                    active_mouse_y = (
                        registered_y
                    )

                set_cursor_position_verified(
                    active_mouse_x,
                    active_mouse_y,
                )

                pydirectinput.mouseUp(
                    button=button
                )

                mouse_button_held = False

                # --------------------------------------------
                # FINISH PLACEMENT
                # --------------------------------------------

                if (
                    button == "left"
                    and placement_in_progress
                    is not None
                ):
                    verification_started = (
                        time.perf_counter()
                    )

                    (
                        successful_offset,
                        actual_x,
                        actual_y,
                        initial_level,
                    ) = verify_or_retry_placement(
                        camera,
                        placement_in_progress,
                        run_number,
                        placement_number,
                        learned_offsets,
                    )

                    verification_elapsed = (
                        time.perf_counter()
                        - verification_started
                    )

                    timing_credit += min(
                        verification_elapsed,
                        PLACEMENT_TIMING_CREDIT_CAP,
                    )

                    tower = TowerState(
                        placement_number=(
                            placement_number
                        ),
                        hotkey=(
                            placement_in_progress
                            .hotkey
                        ),
                        name=TOWER_NAMES[
                            placement_in_progress
                            .hotkey
                        ],
                        recorded_x=(
                            placement_in_progress
                            .recorded_x
                        ),
                        recorded_y=(
                            placement_in_progress
                            .recorded_y
                        ),
                        registered_x=(
                            placement_in_progress
                            .registered_x
                        ),
                        registered_y=(
                            placement_in_progress
                            .registered_y
                        ),
                        actual_x=(
                            actual_x
                        ),
                        actual_y=(
                            actual_y
                        ),
                        local_offset_x=(
                            successful_offset[0]
                        ),
                        local_offset_y=(
                            successful_offset[1]
                        ),
                        level=(
                            initial_level
                        ),
                    )

                    towers.append(
                        tower
                    )

                    selected_tower = (
                        tower
                    )

                    placement_in_progress = None
                    placement_armed_key = None

                # --------------------------------------------
                # NORMAL MAP CLICK: infer selected tower
                # --------------------------------------------

                elif button == "left":
                    (
                        candidate,
                        candidate_distance,
                    ) = nearest_tower(
                        towers,
                        registered_x,
                        registered_y,
                    )

                    if candidate is not None:
                        selected_tower = (
                            candidate
                        )

                        print(
                            "Selection candidate: "
                            f"{candidate.name} "
                            f"#{candidate.placement_number} "
                            f"(distance "
                            f"{candidate_distance:.1f}px)"
                        )

                    else:
                        selected_tower = None

                active_mouse_x = None
                active_mouse_y = None

        else:
            raise RuntimeError(
                f"Unknown action: "
                f"{action.kind}"
            )

    if placement_in_progress is not None:
        raise RuntimeError(
            "Strategy ended during a "
            "placement click."
        )

    if placement_armed_key is not None:
        raise RuntimeError(
            "Strategy ended with an "
            "unresolved tower placement."
        )

    if placement_number != 19:
        raise RuntimeError(
            "Strategy placement count mismatch: "
            f"expected 19, performed "
            f"{placement_number}."
        )

    elapsed = (
        time.perf_counter()
        - started
    )

    print(
        f"Strategy input finished "
        f"after {elapsed:.1f}s."
    )


# ============================================================
# FAILURE SCREENSHOT
# ============================================================

def save_failure_frame(
    camera,
    label: str,
) -> Path | None:
    try:
        frame = latest_frame(
            camera
        )

        timestamp = (
            datetime.now()
            .strftime(
                "%Y%m%d_%H%M%S"
            )
        )

        folder = (
            ERROR_OUTPUT_ROOT
            / timestamp
        )

        folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            folder
            / f"{label}.png"
        )

        cv2.imwrite(
            str(path),
            frame,
        )

        return path

    except Exception:
        return None


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    width, height = (
        current_screen_size()
    )

    print(
        "TDSBot - affordability-aware upgrade build"
    )
    print(
        f"Screen: "
        f"{width}x{height}"
    )
    print(
        "Runs: "
        + (
            "infinite"
            if MAX_RUNS is None
            else str(MAX_RUNS)
        )
    )
    print(
        "Emergency stop: F8"
    )

    if (
        width,
        height,
    ) != EXPECTED_SCREEN_SIZE:
        print()
        print(
            "ERROR: strategy coordinates "
            "require 1920x1080."
        )

        return 2

    actions = parse_strategy(
        STRATEGY_AHK
    )

    learned_offsets = (
        load_placement_offsets()
    )

    print(
        "Loaded placement offsets:"
    )

    for number in sorted(
        learned_offsets
    ):
        offset = (
            learned_offsets[
                number
            ]
        )

        print(
            f"  {number:02d}: "
            f"({offset[0]:+d},"
            f"{offset[1]:+d})"
        )

    print()


    tower_hotkeys = [
        action.args[0]
        for action in actions
        if (
            action.kind == "send"
            and action.args[0]
            in PLACEMENT_HOTKEYS
        )
    ]

    tower_counts = {
        key: tower_hotkeys.count(
            key
        )
        for key in sorted(
            PLACEMENT_HOTKEYS
        )
    }

    print(
        f"Parsed {len(actions)} "
        f"strategy actions."
    )
    print(
        "Expected placements: "
        f"{tower_counts}"
    )

    if tower_counts != {
        "1": 10,
        "2": 2,
        "3": 2,
        "4": 5,
    }:
        raise RuntimeError(
            "The strategy no longer contains "
            "the expected 19 tower hotkeys."
        )

    print()
    print(
        f"Starting in "
        f"{STARTUP_DELAY_SECONDS:.0f} "
        f"seconds..."
    )
    print(
        "Switch to Roblox and "
        "leave the mouse alone."
    )

    interruptible_sleep(
        STARTUP_DELAY_SECONDS
    )

    if activate_roblox():
        interruptible_sleep(
            0.5
        )
    else:
        print(
            "Warning: Roblox window "
            "was not found by title."
        )

    print(
        "Starting continuous "
        "DXcam WinRT capture..."
    )

    camera = dxcam.create(
        backend="winrt",
        output_color="BGR",
    )

    camera.start(
        target_fps=60
    )

    try:
        latest_frame(
            camera,
            timeout=3.0,
        )

        if RUN_INITIAL_SETUP:
            initial_anchor_setup(
                camera
            )
        else:
            print(
                "RUN_INITIAL_SETUP=False; "
                "assuming absolute anchor."
            )

        run_number = 0

        while (
            MAX_RUNS is None
            or run_number < MAX_RUNS
        ):
            check_stop()

            run_number += 1

            print()
            print(
                "=" * 60
            )
            print(
                f"RUN {run_number}"
            )
            print(
                "=" * 60
            )

            # Ensure we are at the READY state before trying
            # to register terrain.
            wait_for_button(
                camera,
                "READY",
                ready_hit,
                READY_WAIT_TIMEOUT,
            )

            (
                map_dx,
                map_dy,
                registration_confidence,
            ) = measure_anchor_shift(
                camera,
                run_number,
            )

            print(
                "Strategy map correction:"
            )
            print(
                f"  x += {map_dx:+d}"
            )
            print(
                f"  y += {map_dy:+d}"
            )
            print()

            ready_elapsed = (
                click_ready(
                    camera
                )
            )

            play_strategy(
                camera=camera,
                actions=actions,
                run_number=run_number,
                learned_offsets=(
                    learned_offsets
                ),
                initial_elapsed=(
                    ready_elapsed
                ),
                map_dx=map_dx,
                map_dy=map_dy,
            )

            click_restart(
                camera
            )

            print()
            print(
                f"Run {run_number} complete."
            )
            print(
                "Waiting for the next READY."
            )
            print(
                "Camera/player setup will not repeat."
            )

        print()
        print(
            "Configured run count completed."
        )

        return 0

    except StopRequested:
        print()
        print(
            "F8 detected. "
            "Stopping safely..."
        )

        return 0

    except KeyboardInterrupt:
        print()
        print(
            "Ctrl+C detected. "
            "Stopping safely..."
        )

        return 0

    except Exception as exc:
        print()
        print(
            f"ERROR: "
            f"{type(exc).__name__}: "
            f"{exc}"
        )

        saved = save_failure_frame(
            camera,
            "failure_state",
        )

        if saved is not None:
            print(
                "Saved failure screenshot:"
            )
            print(
                saved
            )

        return 1

    finally:
        safe_release_all_inputs()

        try:
            camera.stop()
        except Exception:
            pass

        print(
            "Inputs released. "
            "DXcam stopped."
        )


if __name__ == "__main__":
    sys.exit(
        main()
    )
