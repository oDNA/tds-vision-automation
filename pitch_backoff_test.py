import time
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pydirectinput


# --------------------------------------------------
# Output folder / files
# --------------------------------------------------

BASE_FOLDER = Path(__file__).parent

START_FILE = BASE_FOLDER / "backoff_start.png"
ZOOMED_FILE = BASE_FOLDER / "backoff_zoomed.png"
MAX_TOPDOWN_FILE = BASE_FOLDER / "backoff_max_topdown.png"


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

# Move away from the spawn obstruction first.
MOVE_TIME = 2.0

# Force camera to maximum zoom-out.
ZOOM_OUT_PRESSES = 20

# Deliberately overshoot the downward camera movement
# so Roblox reaches its maximum top-down pitch.
MAX_TOPDOWN_DRAG = 300

# Then move upward from that known limit.
PITCH_BACKOFF_STEP = 20

# This produces:
# 20, 40, 60, 80, 100, 120, 140, 160 pixels
# upward from maximum top-down.
NUMBER_OF_BACKOFF_STEPS = 8


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
    Send one deliberate key press.
    """

    pydirectinput.keyDown(key)

    try:
        time.sleep(hold_time)
    finally:
        pydirectinput.keyUp(key)


def capture_and_save(camera, path):
    """
    Capture the current screen and save it as a PNG.
    """

    frame = camera.grab(
        new_frame_only=False
    )

    if frame is None:
        raise RuntimeError(
            "DXcam failed to capture a frame."
        )

    if not cv2.imwrite(
        str(path),
        frame
    ):
        raise RuntimeError(
            f"Could not save screenshot: {path}"
        )


def prepare_mouse():
    """
    Put the mouse in a safe central area and perform the
    tiny relative nudge that has worked reliably in Roblox.
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
    Right-click drag the camera vertically.

    Positive pixels:
        drag downward

    Negative pixels:
        drag upward
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

            # Slow enough that Roblox sees the individual
            # relative mouse movements.
            time.sleep(0.010)

    finally:
        pydirectinput.mouseUp(
            button="right"
        )

    time.sleep(0.40)


def main():
    print("Prototype 5A-5: Pitch backoff calibration")
    print()
    print("Start from a fresh INITIAL spawn.")
    print("Do not manually move, zoom, or rotate the camera.")
    print("READY should be visible.")
    print()
    print("This test will:")
    print("  1. Hold W for 2.0 seconds")
    print("  2. Press O 20 times")
    print("  3. Drag downward 300 px to force max top-down")
    print("  4. Save that view")
    print("  5. Drag upward in 20 px increments")
    print("  6. Save a screenshot after every increment")
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
            # STARTING CHECK
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

            cv2.imwrite(
                str(START_FILE),
                frame
            )

            # ------------------------------------------
            # MOVE AWAY FROM SPAWN OBSTRUCTION
            # ------------------------------------------

            print()
            print(
                f"Holding W for {MOVE_TIME:.1f} seconds..."
            )

            pydirectinput.keyDown("w")

            try:
                time.sleep(MOVE_TIME)
            finally:
                pydirectinput.keyUp("w")

            print("W released.")

            time.sleep(0.75)

            # ------------------------------------------
            # MAXIMUM ZOOM-OUT
            # ------------------------------------------

            print()
            print(
                f"Pressing O {ZOOM_OUT_PRESSES} times..."
            )

            for _ in range(
                ZOOM_OUT_PRESSES
            ):
                press_key_once("o")
                time.sleep(0.07)

            print("Zoom normalization finished.")

            time.sleep(0.75)

            capture_and_save(
                camera,
                ZOOMED_FILE
            )

            print(
                f"Saved: {ZOOMED_FILE}"
            )

            # ------------------------------------------
            # FORCE MAXIMUM TOP-DOWN PITCH
            # ------------------------------------------

            print()
            print(
                f"Dragging downward {MAX_TOPDOWN_DRAG} px "
                "to force maximum top-down..."
            )

            camera_drag_vertical(
                MAX_TOPDOWN_DRAG
            )

            capture_and_save(
                camera,
                MAX_TOPDOWN_FILE
            )

            print(
                f"Saved: {MAX_TOPDOWN_FILE}"
            )

            # ------------------------------------------
            # BACK AWAY FROM MAXIMUM TOP-DOWN
            # ------------------------------------------

            for step in range(
                1,
                NUMBER_OF_BACKOFF_STEPS + 1
            ):

                total_backoff = (
                    step * PITCH_BACKOFF_STEP
                )

                print()
                print(
                    f"Dragging upward "
                    f"{PITCH_BACKOFF_STEP} px..."
                )

                camera_drag_vertical(
                    -PITCH_BACKOFF_STEP
                )

                output_file = (
                    BASE_FOLDER
                    / f"backoff_up_{total_backoff}.png"
                )

                capture_and_save(
                    camera,
                    output_file
                )

                print(
                    f"Total upward backoff from max: "
                    f"{total_backoff} px"
                )

                print(
                    f"Saved: {output_file}"
                )

            # ------------------------------------------
            # FINISHED
            # ------------------------------------------

            print()
            print("PITCH BACKOFF TEST COMPLETE")
            print()
            print("Created:")
            print(MAX_TOPDOWN_FILE)

            for step in range(
                1,
                NUMBER_OF_BACKOFF_STEPS + 1
            ):
                amount = (
                    step * PITCH_BACKOFF_STEP
                )

                print(
                    BASE_FOLDER
                    / f"backoff_up_{amount}.png"
                )

            print()
            print(
                "No READY click or tower input was sent."
            )

    except Exception as exc:

        # Safety release in case something goes wrong
        # while an input is being held.
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