import time
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pydirectinput


# --------------------------------------------------
# Output files
# --------------------------------------------------

BASE_FOLDER = Path(__file__).parent

START_FILE = BASE_FOLDER / "pitch_start.png"
ZOOMED_FILE = BASE_FOLDER / "pitch_zoomed.png"

PITCH_FILES = [
    BASE_FOLDER / "pitch_60.png",
    BASE_FOLDER / "pitch_120.png",
    BASE_FOLDER / "pitch_180.png",
    BASE_FOLDER / "pitch_240.png",
]


# --------------------------------------------------
# READY detector
# --------------------------------------------------

READY_X1 = 930
READY_Y1 = 160
READY_X2 = 1120
READY_Y2 = 290


# --------------------------------------------------
# Test configuration
# --------------------------------------------------

MOVE_TIME = 2.0
ZOOM_OUT_PRESSES = 20

# Each camera test moves vertically downward by 60 pixels.
PITCH_STEP_PIXELS = 60

# Four steps:
# 60, 120, 180, 240 total pixels of downward drag.
NUMBER_OF_PITCH_STEPS = 4


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
    Perform one deliberate keyboard press.
    """

    pydirectinput.keyDown(key)

    try:
        time.sleep(hold_time)
    finally:
        pydirectinput.keyUp(key)


def save_frame(camera, path):
    """
    Capture the current screen and save it.
    """

    frame = camera.grab(
        new_frame_only=False
    )

    if frame is None:
        raise RuntimeError(
            "DXcam could not capture the screen."
        )

    if not cv2.imwrite(
        str(path),
        frame
    ):
        raise RuntimeError(
            f"Could not save screenshot: {path}"
        )


def camera_drag_down(pixels):
    """
    Right-click drag downward by a controlled number
    of pixels.

    Only vertical movement is used so we do not
    intentionally change camera yaw.
    """

    # Start safely near the middle of the screen.
    start_x = 1200
    start_y = 400

    pydirectinput.moveTo(
        start_x,
        start_y
    )

    # Same tiny mouse-position refresh that helped
    # with our Roblox click tests.
    pydirectinput.moveRel(1, 0)
    time.sleep(0.05)

    pydirectinput.moveRel(-1, 0)
    time.sleep(0.10)

    # Begin camera drag.
    pydirectinput.mouseDown(
        button="right"
    )

    try:
        # Move one pixel at a time.
        # This gives Roblox a sequence of genuine
        # relative mouse updates instead of one
        # instantaneous jump.
        for _ in range(pixels):

            pydirectinput.moveRel(
                0,
                1
            )

            time.sleep(0.010)

    finally:
        pydirectinput.mouseUp(
            button="right"
        )

    # Give the camera time to settle.
    time.sleep(0.40)


def main():
    print(
        "Prototype 5A-4: Camera pitch step test"
    )

    print()
    print("Start from a fresh INITIAL spawn.")
    print(
        "Do not manually move, zoom, or rotate "
        "the camera."
    )
    print("READY should be visible.")

    print()
    print("This test will:")
    print("  1. Hold W for 2 seconds")
    print("  2. Press O 20 times")
    print("  3. Right-drag downward 60 px")
    print("  4. Repeat that drag four times")
    print("  5. Save a screenshot after each step")

    print()
    print("No READY click or tower input will occur.")

    print()
    print("Starting in 5 seconds...")

    time.sleep(5)

    try:
        pydirectinput.PAUSE = 0

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
                print(
                    "ERROR: Could not capture "
                    "starting screen."
                )
                return

            if not ready_visible(frame):

                print()
                print("ABORTED")
                print(
                    "READY was not detected."
                )
                print(
                    "No movement was sent."
                )

                return

            print()
            print("READY FOUND")
            print(
                "Starting-state check passed."
            )

            cv2.imwrite(
                str(START_FILE),
                frame
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

            print("Zoom sequence finished.")

            time.sleep(0.75)

            save_frame(
                camera,
                ZOOMED_FILE
            )

            print(
                f"Saved: {ZOOMED_FILE}"
            )

            # ------------------------------------------
            # CAMERA PITCH TEST
            # ------------------------------------------

            for step in range(
                NUMBER_OF_PITCH_STEPS
            ):

                total_pixels = (
                    (step + 1)
                    * PITCH_STEP_PIXELS
                )

                print()
                print(
                    "Right-dragging camera "
                    f"downward by "
                    f"{PITCH_STEP_PIXELS} px..."
                )

                camera_drag_down(
                    PITCH_STEP_PIXELS
                )

                output_file = (
                    PITCH_FILES[step]
                )

                save_frame(
                    camera,
                    output_file
                )

                print(
                    f"Total downward drag: "
                    f"{total_pixels} px"
                )

                print(
                    f"Saved: {output_file}"
                )

            # ------------------------------------------
            # FINISHED
            # ------------------------------------------

            print()
            print("PITCH TEST COMPLETE")

            print()
            print("Created:")
            print(ZOOMED_FILE)

            for path in PITCH_FILES:
                print(path)

            print()
            print(
                "No READY click or tower input "
                "was sent."
            )

    except Exception as exc:

        # Safety releases.
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
            f"{type(exc).__name__}: "
            f"{exc}"
        )


if __name__ == "__main__":
    main()