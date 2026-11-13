from __future__ import annotations

import ctypes
from datetime import datetime
from pathlib import Path
import time

import cv2
import dxcam
import numpy as np


# ============================================================
# CONFIG
# ============================================================

CAPTURE_FPS = 60
POLL_SECONDS = 0.04

# F8 emergency stop
STOP_VK = 0x77

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR
    / "test_output"
    / "tower_level_detector"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


# ============================================================
# PANEL / BAR REGIONS
#
# The supplied Farm screenshots were 1917x1077 rather than
# exactly 1920x1080, so these are stored as normalized regions
# and automatically scaled to the current capture size.
# ============================================================

# Blue ? button used to establish that a selected-tower panel
# is actually open.
QUESTION_ROI_NORM = (
    1200 / 1917,
    510 / 1077,
    1335 / 1917,
    650 / 1077,
)

# Horizontal level/progression bars.
#
# In the supplied Farm examples the five Farm slots are around:
# x = 953..1253
# y = 672..684
#
# Level 0: 0 lit
# Level 1: 1 lit
# Level 2: 2 lit
# Level 3: 3 lit
LEVEL_BAR_ROI_NORM = (
    930 / 1917,
    660 / 1077,
    1280 / 1917,
    695 / 1077,
)


# ============================================================
# COLOR / COMPONENT THRESHOLDS
# ============================================================

# Blue/cyan ? button.
QUESTION_HSV_LOW = np.array(
    [85, 100, 100],
    dtype=np.uint8,
)
QUESTION_HSV_HIGH = np.array(
    [115, 255, 255],
    dtype=np.uint8,
)

QUESTION_MIN_AREA = 850
QUESTION_MIN_WIDTH = 34
QUESTION_MIN_HEIGHT = 34


# Lit level segment.
#
# Measured from the supplied screenshots:
#
# lit segment center:
#     RGB ~ (57, 255, 87)
#     HSV ~ (65, 198, 255)
#
# unlit segment center:
#     RGB ~ (28, 54, 36)
#     HSV ~ (69, 123, 54)
#
# Brightness and saturation separate them extremely well.
LIT_BAR_HSV_LOW = np.array(
    [50, 155, 180],
    dtype=np.uint8,
)
LIT_BAR_HSV_HIGH = np.array(
    [82, 255, 255],
    dtype=np.uint8,
)

LIT_SEGMENT_MIN_WIDTH = 35
LIT_SEGMENT_MAX_WIDTH = 75
LIT_SEGMENT_MIN_HEIGHT = 7
LIT_SEGMENT_MAX_HEIGHT = 22
LIT_SEGMENT_MIN_AREA = 350


# Require the same level in several fresh frames before
# reporting it as stable.
STABLE_FRAMES_REQUIRED = 3


# ============================================================
# WINDOWS
# ============================================================

user32 = ctypes.WinDLL(
    "user32",
    use_last_error=True,
)


def stop_requested() -> bool:
    return bool(
        user32.GetAsyncKeyState(STOP_VK)
        & 0x8000
    )


# ============================================================
# ROI HELPERS
# ============================================================

def scaled_roi(
    frame: np.ndarray,
    normalized_roi,
):
    height, width = frame.shape[:2]

    x1 = int(
        round(
            normalized_roi[0]
            * width
        )
    )
    y1 = int(
        round(
            normalized_roi[1]
            * height
        )
    )
    x2 = int(
        round(
            normalized_roi[2]
            * width
        )
    )
    y2 = int(
        round(
            normalized_roi[3]
            * height
        )
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


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

    return tuple(
        int(value)
        for value in stats[best_index]
    )


# ============================================================
# PANEL DETECTOR
# ============================================================

def question_button_visible(
    frame: np.ndarray,
) -> bool:
    (
        x1,
        y1,
        x2,
        y2,
    ) = scaled_roi(
        frame,
        QUESTION_ROI_NORM,
    )

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
        QUESTION_HSV_LOW,
        QUESTION_HSV_HIGH,
    )

    component = largest_component(
        mask
    )

    if component is None:
        return False

    (
        _x,
        _y,
        width,
        height,
        area,
    ) = component

    return (
        area >= QUESTION_MIN_AREA
        and width >= QUESTION_MIN_WIDTH
        and height >= QUESTION_MIN_HEIGHT
    )


# ============================================================
# LEVEL BAR DETECTOR
# ============================================================

def detect_lit_segments(
    frame: np.ndarray,
):
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

    mask = cv2.inRange(
        hsv,
        LIT_BAR_HSV_LOW,
        LIT_BAR_HSV_HIGH,
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
            LIT_SEGMENT_MIN_WIDTH
            <= width
            <= LIT_SEGMENT_MAX_WIDTH
        ):
            continue

        if not (
            LIT_SEGMENT_MIN_HEIGHT
            <= height
            <= LIT_SEGMENT_MAX_HEIGHT
        ):
            continue

        if (
            area
            < LIT_SEGMENT_MIN_AREA
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

    # Keep left-to-right ordering.
    segments.sort(
        key=lambda item: item[0]
    )

    return segments


def detect_tower_level(
    frame: np.ndarray,
):
    """
    For the TDS panel, each current level lights one additional
    left-to-right progression segment.

    The number of TOTAL slots may differ by tower.
    That does not matter here: current level = number of LIT
    segments.

    Returns None when no selected-tower panel is visible.
    """

    if not question_button_visible(
        frame
    ):
        return None, []

    segments = detect_lit_segments(
        frame
    )

    return (
        len(segments),
        segments,
    )


# ============================================================
# DEBUG SCREENSHOT
# ============================================================

def save_debug(
    frame: np.ndarray,
    level,
    segments,
    number: int,
):
    debug = frame.copy()

    (
        qx1,
        qy1,
        qx2,
        qy2,
    ) = scaled_roi(
        frame,
        QUESTION_ROI_NORM,
    )

    (
        bx1,
        by1,
        bx2,
        by2,
    ) = scaled_roi(
        frame,
        LEVEL_BAR_ROI_NORM,
    )

    cv2.rectangle(
        debug,
        (
            qx1,
            qy1,
        ),
        (
            qx2,
            qy2,
        ),
        (
            255,
            255,
            255,
        ),
        2,
    )

    cv2.rectangle(
        debug,
        (
            bx1,
            by1,
        ),
        (
            bx2,
            by2,
        ),
        (
            255,
            255,
            255,
        ),
        2,
    )

    for (
        x,
        y,
        width,
        height,
        _area,
    ) in segments:
        cv2.rectangle(
            debug,
            (
                x,
                y,
            ),
            (
                x + width,
                y + height,
            ),
            (
                0,
                255,
                0,
            ),
            2,
        )

    if level is None:
        text = (
            "TOWER PANEL: ABSENT"
        )
    else:
        text = (
            f"TOWER LEVEL: {level}"
        )

    cv2.putText(
        debug,
        text,
        (
            40,
            80,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (
            255,
            255,
            255,
        ),
        3,
        cv2.LINE_AA,
    )

    path = (
        OUTPUT_DIR
        / (
            f"{number:02d}_"
            + (
                "absent.png"
                if level is None
                else f"level_{level}.png"
            )
        )
    )

    cv2.imwrite(
        str(path),
        debug,
    )

    return path


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "TDS Tower Level Detector Test"
    )

    print()
    print(
        "F8 = stop"
    )

    print()
    print(
        "Select a Farm and upgrade it through levels 0-3."
    )

    print(
        "The script should report:"
    )

    print(
        "  TOWER LEVEL: 0"
    )
    print(
        "  TOWER LEVEL: 1"
    )
    print(
        "  TOWER LEVEL: 2"
    )
    print(
        "  TOWER LEVEL: 3"
    )

    print()
    print(
        "Different tower maximum levels are fine:"
    )

    print(
        "the detector counts only the lit segments."
    )

    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera = dxcam.create(
        backend="winrt",
        output_color="BGR",
    )

    camera.start(
        target_fps=CAPTURE_FPS
    )

    last_timestamp = None

    candidate_level = object()
    candidate_streak = 0

    reported_level = object()

    debug_number = 0

    try:
        time.sleep(0.5)

        while True:
            if stop_requested():
                print()
                print(
                    "F8 detected. Stopping."
                )
                break

            result = (
                camera.get_latest_frame(
                    with_timestamp=True
                )
            )

            if result is None:
                time.sleep(
                    POLL_SECONDS
                )
                continue

            frame, timestamp = (
                result
            )

            if (
                frame is None
                or timestamp
                == last_timestamp
            ):
                time.sleep(
                    POLL_SECONDS
                )
                continue

            last_timestamp = (
                timestamp
            )

            (
                level,
                segments,
            ) = detect_tower_level(
                frame
            )

            if level == candidate_level:
                candidate_streak += 1
            else:
                candidate_level = level
                candidate_streak = 1

            if (
                candidate_streak
                >= STABLE_FRAMES_REQUIRED
                and candidate_level
                != reported_level
            ):
                reported_level = (
                    candidate_level
                )

                debug_number += 1

                if reported_level is None:
                    print()
                    print(
                        "TOWER PANEL: ABSENT"
                    )
                else:
                    print()
                    print(
                        f"TOWER LEVEL: "
                        f"{reported_level}"
                    )

                    print(
                        "  lit segments:"
                    )

                    for segment in segments:
                        print(
                            f"    {segment}"
                        )

                path = save_debug(
                    frame,
                    reported_level,
                    segments,
                    debug_number,
                )

                print(
                    f"  saved: {path}"
                )

            time.sleep(
                POLL_SECONDS
            )

    finally:
        try:
            camera.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
