import ctypes
import time
from datetime import datetime
from pathlib import Path

import cv2
import dxcam
import numpy as np


# ============================================================
# CONFIG
# ============================================================

CAPTURE_FPS = 60

# F8
STOP_VK = 0x77

POLL_SECONDS = 0.10

BASE = Path(__file__).resolve().parent

OUTPUT_FOLDER = (
    BASE
    / "test_output"
    / "tower_panel_detector"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)


# ============================================================
# DETECTOR REGIONS
#
# These are deliberately much smaller than the whole panel.
#
# Blue question-mark button:
# roughly x=1241..1296, y=533..589
#
# Red Sell button:
# roughly x=685..915, y=845..891
#
# ROIs include some safety margin.
# ============================================================

QUESTION_ROI = (
    1200,
    520,
    1330,
    650,
)

SELL_ROI = (
    630,
    820,
    960,
    940,
)


# ============================================================
# DETECTOR THRESHOLDS
# ============================================================

# From the supplied selected-tower screenshot, the blue
# question-mark component was approximately:
#
#     width  = 55
#     height = 56
#     area   = 1925
#
# We deliberately set the thresholds substantially lower to
# tolerate minor rendering differences.

QUESTION_MIN_AREA = 900
QUESTION_MIN_WIDTH = 35
QUESTION_MIN_HEIGHT = 35


# The red Sell button produced a component approximately:
#
#     width = 220
#     height = 31
#     area = 5465
#
SELL_MIN_AREA = 2500
SELL_MIN_WIDTH = 150
SELL_MIN_HEIGHT = 20


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
# COMPONENT HELPER
# ============================================================

def largest_component(mask):
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


# ============================================================
# QUESTION MARK DETECTOR
# ============================================================

def detect_question_button(frame):
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

    # Cyan/blue used by the ? button.
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


# ============================================================
# SELL BUTTON DETECTOR
# ============================================================

def detect_sell_button(frame):
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

    # Red wraps around HSV hue 0.
    lower_red = cv2.inRange(
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

    upper_red = cv2.inRange(
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
        lower_red,
        upper_red,
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


# ============================================================
# COMPLETE PANEL DETECTOR
# ============================================================

def tower_panel_visible(frame):
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

    # Require BOTH independent GUI features.
    visible = (
        question_found
        and sell_found
    )

    return (
        visible,
        question_component,
        sell_component,
    )


# ============================================================
# DEBUG IMAGE
# ============================================================

def draw_debug(
    frame,
    visible,
    question_component,
    sell_component,
):
    debug = frame.copy()

    # Question ROI
    cv2.rectangle(
        debug,
        (
            QUESTION_ROI[0],
            QUESTION_ROI[1],
        ),
        (
            QUESTION_ROI[2],
            QUESTION_ROI[3],
        ),
        (
            255,
            255,
            255,
        ),
        2,
    )

    # Sell ROI
    cv2.rectangle(
        debug,
        (
            SELL_ROI[0],
            SELL_ROI[1],
        ),
        (
            SELL_ROI[2],
            SELL_ROI[3],
        ),
        (
            255,
            255,
            255,
        ),
        2,
    )

    if question_component:
        (
            x,
            y,
            width,
            height,
            _area,
        ) = question_component

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

    if sell_component:
        (
            x,
            y,
            width,
            height,
            _area,
        ) = sell_component

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

    text = (
        "TOWER PANEL: PRESENT"
        if visible
        else "TOWER PANEL: ABSENT"
    )

    cv2.putText(
        debug,
        text,
        (
            40,
            80,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (
            255,
            255,
            255,
        ),
        3,
        cv2.LINE_AA,
    )

    return debug


# ============================================================
# MAIN
# ============================================================

def main():
    print(
        "Tower Panel Detector Test"
    )

    print()
    print(
        "F8 = stop"
    )

    print()
    print(
        "Test both states:"
    )

    print(
        "  1. Nothing selected"
    )

    print(
        "  2. Tower selected / panel open"
    )

    print()

    OUTPUT_FOLDER.mkdir(
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

    previous_state = None
    screenshot_number = 0

    try:
        time.sleep(0.5)

        while True:
            if stop_requested():
                print()
                print(
                    "F8 detected. Stopping."
                )
                break

            frame = (
                camera.get_latest_frame()
            )

            if frame is None:
                time.sleep(
                    POLL_SECONDS
                )
                continue

            (
                visible,
                question_component,
                sell_component,
            ) = tower_panel_visible(
                frame
            )

            if (
                visible
                != previous_state
            ):
                screenshot_number += 1

                print()
                print(
                    "TOWER PANEL:",
                    (
                        "PRESENT"
                        if visible
                        else "ABSENT"
                    ),
                )

                print(
                    "  question:",
                    question_component,
                )

                print(
                    "  sell:",
                    sell_component,
                )

                debug = draw_debug(
                    frame,
                    visible,
                    question_component,
                    sell_component,
                )

                path = (
                    OUTPUT_FOLDER
                    / (
                        f"{screenshot_number:02d}_"
                        + (
                            "present.png"
                            if visible
                            else "absent.png"
                        )
                    )
                )

                cv2.imwrite(
                    str(path),
                    debug,
                )

                print(
                    "  saved:",
                    path,
                )

                previous_state = (
                    visible
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