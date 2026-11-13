import time
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pydirectinput


BEFORE_FILE = Path(__file__).with_name("zoom_before.png")
AFTER_FILE = Path(__file__).with_name("zoom_after.png")


# READY detector region.
READY_X1 = 930
READY_Y1 = 160
READY_X2 = 1120
READY_Y2 = 290


def ready_visible(frame):
    """
    Return True when READY is visible.
    """

    roi = frame[
        READY_Y1:READY_Y2,
        READY_X1:READY_X2
    ]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_green = np.array([45, 120, 80], dtype=np.uint8)
    upper_green = np.array([75, 255, 255], dtype=np.uint8)

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


def press_roblox_key(key, hold_time=0.08):
    """
    Send one deliberate key press with a short key-down time.
    """

    pydirectinput.keyDown(key)
    time.sleep(hold_time)
    pydirectinput.keyUp(key)


def main():
    print("Prototype 5A-2: Zoom normalization test")
    print()
    print("Start from the normal INITIAL spawn camera.")
    print("Do not manually zoom or rotate the camera.")
    print("READY should be visible.")
    print()
    print("Starting in 5 seconds...")

    time.sleep(5)

    try:
        pydirectinput.PAUSE = 0

        with dxcam.create(
            backend="winrt",
            output_color="BGR",
        ) as camera:

            # -------------------------
            # PRECONDITION
            # -------------------------

            frame = camera.grab(new_frame_only=False)

            if frame is None:
                print("ERROR: Could not capture screen.")
                return

            if not ready_visible(frame):
                print()
                print("ABORTED")
                print("READY was not detected.")
                print("No keyboard input was sent.")
                return

            print()
            print("READY FOUND")
            print("Starting-state check passed.")

            cv2.imwrite(str(BEFORE_FILE), frame)

            # -------------------------
            # ZOOM OUT
            # -------------------------

            print()
            print("Pressing O 20 times to reach maximum zoom-out...")

            for press_number in range(1, 21):
                press_roblox_key("o")
                time.sleep(0.07)

            print("Zoom sequence finished.")

            # Give Roblox a moment to settle.
            time.sleep(0.75)

            # -------------------------
            # CAPTURE RESULT
            # -------------------------

            frame = camera.grab(new_frame_only=False)

            if frame is None:
                print("ERROR: Could not capture final screen.")
                return

            if not cv2.imwrite(str(AFTER_FILE), frame):
                print("ERROR: Could not save zoom_after.png.")
                return

            print()
            print("ZOOM TEST COMPLETE")
            print(f"Before: {BEFORE_FILE}")
            print(f"After:  {AFTER_FILE}")
            print()
            print("No movement, camera drag, READY click,")
            print("or tower input was sent.")

    except Exception as exc:
        try:
            pydirectinput.keyUp("o")
        except Exception:
            pass

        print()
        print("ERROR")
        print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()