import ctypes
import time
from datetime import datetime
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pydirectinput


# ==================================================
# OUTPUT
# ==================================================

BASE_FOLDER = Path(__file__).parent

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_FOLDER = (
    BASE_FOLDER
    / "test_output"
    / "navigation_reconstruction"
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
# NAVIGATION SETTINGS
# ==================================================

# First Pulover camera movement was approximately:
#
# (1320,525) -> (1326,585)
#
# So:
# dx = +6
# dy = +60
FIRST_CAMERA_DX = 6
FIRST_CAMERA_DY = 60


# Second recorded camera movement:
#
# (1326,575) -> (1326,576)
#
# It is tiny, but we'll preserve it during this first
# reconstruction attempt.
SECOND_CAMERA_DX = 0
SECOND_CAMERA_DY = 1


# The repeated S events in Pulover spanned about
# 3.0 seconds.
MOVE_S_TIME = 3.0


# Continuous DXcam capture.
CAPTURE_FPS = 60


# ==================================================
# WINDOWS RAW MOUSE
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
    """
    Return True when the READY button is visible.
    """

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


def get_fresh_frame(camera, settle_time=0.75):
    """
    Wait for fresh rendered frames, then return the latest one.
    """

    time.sleep(settle_time)

    frame, timestamp = camera.get_latest_frame(
        with_timestamp=True
    )

    if frame is None:
        raise RuntimeError(
            "Could not obtain a fresh DXcam frame."
        )

    return frame, timestamp


# ==================================================
# ROBLOX CAMERA INPUT
# ==================================================

def raw_relative_right_drag(dx, dy, hold_before=0.20):
    """
    Perform one true relative right-mouse camera movement.

    No absolute mouse positioning occurs while RMB is held.
    """

    # Put cursor somewhere safe before Roblox captures it.
    set_cursor_position(
        1200,
        500
    )

    time.sleep(0.20)

    # Tiny refresh nudge.
    relative_mouse_move(1, 0)
    time.sleep(0.05)

    relative_mouse_move(-1, 0)
    time.sleep(0.10)

    right_mouse_down()

    try:
        time.sleep(hold_before)

        relative_mouse_move(
            dx,
            dy
        )

        time.sleep(0.20)

    finally:
        right_mouse_up()

    time.sleep(0.50)


# ==================================================
# PLAYER MOVEMENT
# ==================================================

def hold_key(key, duration):
    """
    Hold a keyboard key for a precise duration.
    """

    pydirectinput.keyDown(key)

    try:
        time.sleep(duration)

    finally:
        pydirectinput.keyUp(key)


# ==================================================
# MAIN
# ==================================================

def main():
    print(
        "Prototype 5A-10: Navigation reconstruction test"
    )

    print()
    print("Start from a fresh INITIAL spawn.")
    print("Use the normal untouched starting camera.")
    print("READY should still be visible.")

    print()
    print("This test will reproduce:")
    print(
        f"  1. Relative camera drag "
        f"({FIRST_CAMERA_DX}, {FIRST_CAMERA_DY})"
    )
    print(
        f"  2. Relative camera drag "
        f"({SECOND_CAMERA_DX}, {SECOND_CAMERA_DY})"
    )
    print(
        f"  3. Hold S for {MOVE_S_TIME:.1f} seconds"
    )

    print()
    print("It will NOT:")
    print("  - click READY")
    print("  - place a tower")
    print("  - upgrade anything")
    print("  - press I or O")

    print()
    print("Output folder:")
    print(RUN_FOLDER)

    print()
    print("Starting in 5 seconds...")

    time.sleep(5)

    try:
        pydirectinput.PAUSE = 0

        RUN_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        with dxcam.create(
            backend="winrt",
            output_color="BGR",
        ) as camera:

            # ------------------------------------------
            # CONTINUOUS CAPTURE
            # ------------------------------------------

            print()
            print("Starting continuous DXcam capture...")

            camera.start(
                target_fps=CAPTURE_FPS
            )

            time.sleep(0.75)

            # ------------------------------------------
            # PRECONDITION
            # ------------------------------------------

            start_frame, start_timestamp = (
                camera.get_latest_frame(
                    with_timestamp=True
                )
            )

            if start_frame is None:
                print(
                    "ERROR: Could not obtain starting frame."
                )
                return

            if not ready_visible(start_frame):

                print()
                print("ABORTED")
                print("READY was not detected.")
                print("No navigation input was sent.")

                return

            print()
            print("READY FOUND")
            print("Starting-state check passed.")

            save_frame(
                start_frame,
                "00_initial_spawn.png"
            )

            # ------------------------------------------
            # FIRST CAMERA MOVEMENT
            # ------------------------------------------

            print()
            print(
                "First raw relative camera movement:"
            )

            print(
                f"dx={FIRST_CAMERA_DX}, "
                f"dy={FIRST_CAMERA_DY}"
            )

            raw_relative_right_drag(
                FIRST_CAMERA_DX,
                FIRST_CAMERA_DY
            )

            frame, timestamp = get_fresh_frame(
                camera
            )

            save_frame(
                frame,
                "01_after_first_camera_drag.png"
            )

            # ------------------------------------------
            # ORIGINAL MACRO DELAY
            # ------------------------------------------

            time.sleep(0.890)

            # ------------------------------------------
            # SECOND CAMERA MOVEMENT
            # ------------------------------------------

            print()
            print(
                "Second raw relative camera movement:"
            )

            print(
                f"dx={SECOND_CAMERA_DX}, "
                f"dy={SECOND_CAMERA_DY}"
            )

            raw_relative_right_drag(
                SECOND_CAMERA_DX,
                SECOND_CAMERA_DY
            )

            frame, timestamp = get_fresh_frame(
                camera
            )

            save_frame(
                frame,
                "02_after_second_camera_drag.png"
            )

            # Original delay before movement.
            time.sleep(0.343)

            # ------------------------------------------
            # PLAYER MOVEMENT
            # ------------------------------------------

            print()
            print(
                f"Holding S for "
                f"{MOVE_S_TIME:.1f} seconds..."
            )

            hold_key(
                "s",
                MOVE_S_TIME
            )

            print("S released.")

            # Give player/camera time to settle.
            frame, final_timestamp = get_fresh_frame(
                camera,
                settle_time=1.5
            )

            save_frame(
                frame,
                "03_after_s_movement.png"
            )

            # ------------------------------------------
            # COMPLETE
            # ------------------------------------------

            camera.stop()

            print()
            print("NAVIGATION TEST COMPLETE")

            print()
            print("Important files:")
            print("  00_initial_spawn.png")
            print("  01_after_first_camera_drag.png")
            print("  02_after_second_camera_drag.png")
            print("  03_after_s_movement.png")

            print()
            print(
                "No READY click or gameplay action was sent."
            )

    except Exception as exc:

        # Safety releases.
        try:
            pydirectinput.keyUp("s")
        except Exception:
            pass

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