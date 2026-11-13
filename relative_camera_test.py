import ctypes
import time
from datetime import datetime
from pathlib import Path

import cv2
import dxcam
import numpy as np


# ==================================================
# OUTPUT FOLDER
# ==================================================

BASE_FOLDER = Path(__file__).parent

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_FOLDER = (
    BASE_FOLDER
    / "test_output"
    / "relative_camera_fresh"
    / RUN_TIMESTAMP
)


# ==================================================
# READY DETECTOR
# ==================================================

READY_X1 = 930
READY_Y1 = 160
READY_X2 = 1120
READY_Y2 = 290


# ==================================================
# CAMERA TEST
# ==================================================

RELATIVE_DX = 0
RELATIVE_DY = 60

CAPTURE_FPS = 60

# Wait this long after the mouse movement before
# selecting the final frame.
POST_ACTION_WAIT = 1.5


# ==================================================
# WINDOWS MOUSE
# ==================================================

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

user32 = ctypes.WinDLL(
    "user32",
    use_last_error=True
)


def set_cursor_position(x, y):
    result = user32.SetCursorPos(
        int(x),
        int(y)
    )

    if result == 0:
        raise ctypes.WinError(
            ctypes.get_last_error()
        )


def relative_mouse_move(dx, dy):
    user32.mouse_event(
        MOUSEEVENTF_MOVE,
        int(dx),
        int(dy),
        0,
        0
    )


def right_mouse_down():
    user32.mouse_event(
        MOUSEEVENTF_RIGHTDOWN,
        0,
        0,
        0,
        0
    )


def right_mouse_up():
    user32.mouse_event(
        MOUSEEVENTF_RIGHTUP,
        0,
        0,
        0,
        0
    )


# ==================================================
# VISION
# ==================================================

def ready_visible(frame):
    roi = frame[
        READY_Y1:READY_Y2,
        READY_X1:READY_X2
    ]

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )

    lower_green = np.array(
        [45, 120, 80],
        dtype=np.uint8
    )

    upper_green = np.array(
        [75, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    for i in range(1, num_labels):
        x, y, width, height, area = stats[i]

        if (
            area >= 4000
            and width >= 120
            and height >= 35
        ):
            return True

    return False


def save_frame(frame, filename):
    path = RUN_FOLDER / filename

    if not cv2.imwrite(
        str(path),
        frame
    ):
        raise RuntimeError(
            f"Could not save screenshot: {path}"
        )

    print(f"Saved: {path}")


# ==================================================
# CAMERA INPUT
# ==================================================

def relative_right_drag(dx, dy):
    """
    Perform one genuine relative Roblox camera movement.
    """

    # Absolute positioning is okay BEFORE RMB is held.
    set_cursor_position(
        1200,
        500
    )

    time.sleep(0.30)

    # Small refresh nudge.
    relative_mouse_move(1, 0)
    time.sleep(0.05)

    relative_mouse_move(-1, 0)
    time.sleep(0.10)

    # Begin Roblox camera control.
    right_mouse_down()

    try:
        time.sleep(0.20)

        # One true relative mouse delta.
        relative_mouse_move(
            dx,
            dy
        )

        time.sleep(0.20)

    finally:
        right_mouse_up()


# ==================================================
# MAIN
# ==================================================

def main():
    print(
        "Prototype 5A-9: Relative camera + fresh capture test"
    )

    print()
    print("Start from a fresh INITIAL spawn.")
    print("Do not manually move, zoom, or rotate.")
    print("READY should be visible.")

    print()
    print(
        f"Camera input: dx={RELATIVE_DX}, "
        f"dy={RELATIVE_DY}"
    )

    print()
    print(
        f"DXcam continuous capture: {CAPTURE_FPS} FPS"
    )

    print()
    print("Output folder:")
    print(RUN_FOLDER)

    print()
    print("Starting in 5 seconds...")

    time.sleep(5)

    try:
        RUN_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        with dxcam.create(
            backend="winrt",
            output_color="BGR",
        ) as camera:

            # ------------------------------------------
            # START CONTINUOUS CAPTURE
            # ------------------------------------------

            print()
            print("Starting continuous DXcam capture...")

            camera.start(
                target_fps=CAPTURE_FPS
            )

            # Let the capture thread warm up.
            time.sleep(0.75)

            # ------------------------------------------
            # PRECONDITION
            # ------------------------------------------

            before_frame, before_timestamp = (
                camera.get_latest_frame(
                    with_timestamp=True
                )
            )

            if before_frame is None:
                print(
                    "ERROR: Could not get starting frame."
                )
                return

            if not ready_visible(before_frame):
                print()
                print("ABORTED")
                print("READY was not detected.")
                print("No mouse input was sent.")
                return

            print()
            print("READY FOUND")
            print("Starting-state check passed.")

            print(
                f"Before frame timestamp: "
                f"{before_timestamp:.6f}"
            )

            save_frame(
                before_frame,
                "00_before_relative_drag.png"
            )

            # ------------------------------------------
            # CAMERA ACTION
            # ------------------------------------------

            print()
            print(
                "Sending genuine relative camera movement..."
            )

            relative_right_drag(
                RELATIVE_DX,
                RELATIVE_DY
            )

            # ------------------------------------------
            # WAIT FOR CAMERA / CAPTURE TO SETTLE
            # ------------------------------------------

            print(
                f"Waiting {POST_ACTION_WAIT:.1f} seconds "
                "for fresh rendered frames..."
            )

            time.sleep(
                POST_ACTION_WAIT
            )

            # Get the most recent frame from the
            # continuously running capture thread.
            after_frame, after_timestamp = (
                camera.get_latest_frame(
                    with_timestamp=True
                )
            )

            if after_frame is None:
                print(
                    "ERROR: Could not get final frame."
                )
                return

            print(
                f"After frame timestamp:  "
                f"{after_timestamp:.6f}"
            )

            print(
                f"Timestamp difference:   "
                f"{after_timestamp - before_timestamp:.3f} s"
            )

            save_frame(
                after_frame,
                "01_after_relative_drag.png"
            )

            # ------------------------------------------
            # SIMPLE IMAGE DIFFERENCE
            # ------------------------------------------

            if before_frame.shape == after_frame.shape:

                difference = cv2.absdiff(
                    before_frame,
                    after_frame
                )

                mean_difference = float(
                    difference.mean()
                )

                print()
                print(
                    f"Mean image difference: "
                    f"{mean_difference:.2f}"
                )

            # ------------------------------------------
            # STOP CAPTURE
            # ------------------------------------------

            camera.stop()

            print()
            print("TEST COMPLETE")

            print()
            print("Compare:")
            print("  00_before_relative_drag.png")
            print("  01_after_relative_drag.png")

            print()
            print(
                "No player movement, zoom, READY click, "
                "or tower placement was sent."
            )

    except Exception as exc:

        try:
            right_mouse_up()
        except Exception:
            pass

        print()
        print("ERROR")
        print(
            f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    main()