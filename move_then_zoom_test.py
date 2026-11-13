import time
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pydirectinput


# --------------------------------------------------
# Output screenshots
# --------------------------------------------------

START_FILE = Path(__file__).with_name("mtz_start.png")
AFTER_MOVE_FILE = Path(__file__).with_name("mtz_after_move.png")
AFTER_ZOOM_FILE = Path(__file__).with_name("mtz_after_zoom.png")


# --------------------------------------------------
# READY detector configuration
# --------------------------------------------------

READY_X1 = 930
READY_Y1 = 160
READY_X2 = 1120
READY_Y2 = 290


# --------------------------------------------------
# Test configuration
# --------------------------------------------------

# We are NOT trying to reach the final anchor yet.
# This movement is only intended to get away from the
# object blocking the camera at spawn.
MOVE_TIME = 2.0

# Number of O presses used to test maximum zoom-out.
ZOOM_OUT_PRESSES = 20


def ready_visible(frame):
    """
    Return True when the green READY button is visible.
    """

    # Crop the expected READY region.
    roi = frame[
        READY_Y1:READY_Y2,
        READY_X1:READY_X2
    ]

    # Convert from OpenCV's BGR format to HSV.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Green range that worked in our previous READY tests.
    lower_green = np.array([45, 120, 80], dtype=np.uint8)
    upper_green = np.array([75, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Find connected green regions.
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    # Label 0 is the background.
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


def main():
    print("Prototype 5A-3: Move forward, then zoom test")
    print()
    print("Start from a fresh INITIAL spawn.")
    print("Do not manually move, zoom, or rotate the camera.")
    print("READY should be visible.")
    print()
    print("This test will:")
    print("  1. Hold W for 2.0 seconds")
    print("  2. Stop")
    print("  3. Press O 20 times")
    print()
    print("Starting in 5 seconds...")

    time.sleep(5)

    try:
        # Remove PyDirectInput's default pause between commands.
        pydirectinput.PAUSE = 0

        with dxcam.create(
            backend="winrt",
            output_color="BGR",
        ) as camera:

            # ------------------------------------------
            # STARTING PRECONDITION
            # ------------------------------------------

            frame = camera.grab(new_frame_only=False)

            if frame is None:
                print()
                print("ERROR: Could not capture starting screen.")
                return

            if not ready_visible(frame):
                print()
                print("ABORTED")
                print("READY was not detected.")
                print("No movement or zoom input was sent.")
                return

            print()
            print("READY FOUND")
            print("Starting-state check passed.")

            cv2.imwrite(
                str(START_FILE),
                frame
            )

            # ------------------------------------------
            # MOVE AWAY FROM THE SPAWN OBSTRUCTION
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
            print("Movement finished.")

            # Let the game/camera settle.
            time.sleep(0.75)

            frame = camera.grab(new_frame_only=False)

            if frame is None:
                print()
                print("ERROR: Could not capture after movement.")
                return

            cv2.imwrite(
                str(AFTER_MOVE_FILE),
                frame
            )

            # ------------------------------------------
            # TEST ZOOM-OUT
            # ------------------------------------------

            print()
            print(
                f"Pressing O {ZOOM_OUT_PRESSES} times..."
            )

            for press_number in range(
                1,
                ZOOM_OUT_PRESSES + 1
            ):
                press_key_once("o")

                # Small delay so Roblox has time to process
                # each individual zoom step.
                time.sleep(0.07)

            print("Zoom sequence finished.")

            # Let the camera settle again.
            time.sleep(0.75)

            # ------------------------------------------
            # CAPTURE FINAL RESULT
            # ------------------------------------------

            frame = camera.grab(new_frame_only=False)

            if frame is None:
                print()
                print("ERROR: Could not capture after zoom.")
                return

            if not cv2.imwrite(
                str(AFTER_ZOOM_FILE),
                frame
            ):
                print()
                print("ERROR: Could not save final screenshot.")
                return

            print()
            print("TEST COMPLETE")
            print()
            print("Starting view:")
            print(START_FILE)

            print()
            print("After holding W for 2.0 seconds:")
            print(AFTER_MOVE_FILE)

            print()
            print("After movement + 20 O presses:")
            print(AFTER_ZOOM_FILE)

            print()
            print(
                "No camera drag, READY click, or tower "
                "input was sent."
            )

    except Exception as exc:
        # Best-effort safety release in case an exception
        # happens while a key is held.
        try:
            pydirectinput.keyUp("w")
            pydirectinput.keyUp("o")
        except Exception:
            pass

        print()
        print("ERROR")
        print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()