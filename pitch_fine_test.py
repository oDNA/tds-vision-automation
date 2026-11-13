import time
from datetime import datetime
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pydirectinput


# --------------------------------------------------
# Folder structure
# --------------------------------------------------

BASE_FOLDER = Path(__file__).parent

# Future test screenshots will go under:
#
# C:\TDSBot\test_output\
#
TEST_OUTPUT_FOLDER = BASE_FOLDER / "test_output"

# Each run gets its own folder so screenshots from
# different attempts do not overwrite each other.
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_FOLDER = (
    TEST_OUTPUT_FOLDER
    / "pitch_fine"
    / RUN_TIMESTAMP
)


# --------------------------------------------------
# READY detector
# --------------------------------------------------

READY_X1 = 930
READY_Y1 = 160
READY_X2 = 1120
READY_Y2 = 290


# --------------------------------------------------
# Calibration settings
# --------------------------------------------------

# Move away from the spawn obstruction.
MOVE_TIME = 2.0

# Force known maximum zoom-out.
ZOOM_OUT_PRESSES = 20

# This reproduces the camera base used in the previous test.
#
# IMPORTANT:
# We are NOT claiming this is a physical camera limit.
# It is simply a repeatable relative camera movement from
# the deterministic starting view.
BASE_DOWNWARD_DRAG = 300

# Fine pitch steps.
FINE_STEP_PIXELS = 5

# Produce:
# +5
# +10
# +15
# +20
NUMBER_OF_FINE_STEPS = 4


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


def press_key_once(key, hold_time=0.08):
    """
    Send one deliberate keyboard press.
    """

    pydirectinput.keyDown(key)

    try:
        time.sleep(hold_time)
    finally:
        pydirectinput.keyUp(key)


def capture_and_save(camera, filename):
    """
    Capture the current screen and save it in this
    test run's folder.
    """

    frame = camera.grab(
        new_frame_only=False
    )

    if frame is None:
        raise RuntimeError(
            "DXcam failed to capture a frame."
        )

    path = RUN_FOLDER / filename

    if not cv2.imwrite(
        str(path),
        frame
    ):
        raise RuntimeError(
            f"Could not save screenshot: {path}"
        )

    print(f"Saved: {path}")


def prepare_mouse():
    """
    Move the mouse to a safe region and force Roblox
    to refresh its internal mouse position.
    """

    pydirectinput.moveTo(
        1200,
        500
    )

    pydirectinput.moveRel(1, 0)
    time.sleep(0.05)

    pydirectinput.moveRel(-1, 0)
    time.sleep(0.10)


def camera_drag_vertical(pixels):
    """
    Right-click drag vertically.

    Positive:
        drag mouse downward

    Negative:
        drag mouse upward
    """

    prepare_mouse()

    direction = 1 if pixels > 0 else -1
    distance = abs(pixels)

    pydirectinput.mouseDown(
        button="right"
    )

    try:
        for _ in range(distance):

            pydirectinput.moveRel(
                0,
                direction
            )

            time.sleep(0.010)

    finally:
        pydirectinput.mouseUp(
            button="right"
        )

    time.sleep(0.40)


def main():
    print("Prototype 5A-6: Fine camera pitch calibration")

    print()
    print("Start from a fresh INITIAL spawn.")
    print("Do not manually move, zoom, or rotate.")
    print("READY should be visible.")

    print()
    print("This test will:")
    print("  1. Hold W for 2.0 seconds")
    print("  2. Press O 20 times")
    print("  3. Drag camera downward 300 px")
    print("  4. Save the base camera")
    print("  5. Move upward in 5 px increments")
    print("  6. Stop after 20 px")

    print()
    print("No READY click or tower input will occur.")

    print()
    print(f"Images will be stored in:")
    print(RUN_FOLDER)

    print()
    print("Starting in 5 seconds...")

    time.sleep(5)

    try:
        pydirectinput.PAUSE = 0

        # Create organized folders.
        RUN_FOLDER.mkdir(
            parents=True,
            exist_ok=True
        )

        with dxcam.create(
            backend="winrt",
            output_color="BGR",
        ) as camera:

            # ------------------------------------------
            # PRECONDITION
            # ------------------------------------------

            frame = camera.grab(
                new_frame_only=False
            )

            if frame is None:
                print()
                print("ERROR: Could not capture starting screen.")
                return

            if not ready_visible(frame):

                print()
                print("ABORTED")
                print("READY was not detected.")
                print("No movement was sent.")

                return

            print()
            print("READY FOUND")
            print("Starting-state check passed.")

            capture_and_save(
                camera,
                "00_start.png"
            )

            # ------------------------------------------
            # MOVE AWAY FROM SPAWN OBSTRUCTION
            # ------------------------------------------

            print()
            print(
                f"Holding W for "
                f"{MOVE_TIME:.1f} seconds..."
            )

            pydirectinput.keyDown("w")

            try:
                time.sleep(MOVE_TIME)

            finally:
                pydirectinput.keyUp("w")

            print("W released.")

            time.sleep(0.75)

            capture_and_save(
                camera,
                "01_after_move.png"
            )

            # ------------------------------------------
            # NORMALIZE ZOOM
            # ------------------------------------------

            print()
            print(
                f"Pressing O "
                f"{ZOOM_OUT_PRESSES} times..."
            )

            for _ in range(
                ZOOM_OUT_PRESSES
            ):
                press_key_once("o")

                time.sleep(0.07)

            print("Zoom-out finished.")

            time.sleep(0.75)

            capture_and_save(
                camera,
                "02_max_zoom_out.png"
            )

            # ------------------------------------------
            # REPRODUCE CAMERA BASE
            # ------------------------------------------

            print()
            print(
                f"Dragging camera downward "
                f"{BASE_DOWNWARD_DRAG} px..."
            )

            camera_drag_vertical(
                BASE_DOWNWARD_DRAG
            )

            capture_and_save(
                camera,
                "03_pitch_base.png"
            )

            # ------------------------------------------
            # FINE UPWARD STEPS
            # ------------------------------------------

            for step in range(
                1,
                NUMBER_OF_FINE_STEPS + 1
            ):

                total_up = (
                    step * FINE_STEP_PIXELS
                )

                print()
                print(
                    f"Dragging upward "
                    f"{FINE_STEP_PIXELS} px..."
                )

                camera_drag_vertical(
                    -FINE_STEP_PIXELS
                )

                capture_and_save(
                    camera,
                    f"04_pitch_up_{total_up:02d}.png"
                )

                print(
                    f"Total upward movement from base: "
                    f"{total_up} px"
                )

            # ------------------------------------------
            # COMPLETE
            # ------------------------------------------

            print()
            print("FINE PITCH TEST COMPLETE")

            print()
            print("Look inside:")
            print(RUN_FOLDER)

            print()
            print("Important comparison images:")
            print("  03_pitch_base.png")
            print("  04_pitch_up_05.png")
            print("  04_pitch_up_10.png")
            print("  04_pitch_up_15.png")
            print("  04_pitch_up_20.png")

            print()
            print(
                "No READY click or tower input was sent."
            )

    except Exception as exc:

        # Safety release.
        try:
            pydirectinput.keyUp("w")
            pydirectinput.keyUp("o")

            pydirectinput.mouseUp(
                button="right"
            )

        except Exception:
            pass

        print()
        print("ERROR")
        print(
            f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    main()