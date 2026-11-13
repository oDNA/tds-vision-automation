import time
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pydirectinput


READY_X1 = 930
READY_Y1 = 160
READY_X2 = 1120
READY_Y2 = 290

READY_CLICK_X = 1030
READY_CLICK_Y = 224

POST_CLICK_TIMEOUT = 5.0

DEBUG_FOLDER = Path(__file__).with_name("debug")


def ready_visible(frame):
    roi = frame[
        READY_Y1:READY_Y2,
        READY_X1:READY_X2
    ]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_green = np.array([45, 120, 80], dtype=np.uint8)
    upper_green = np.array([75, 255, 255], dtype=np.uint8)

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            green_mask,
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


def save_debug_image(frame, filename):
    DEBUG_FOLDER.mkdir(exist_ok=True)

    path = DEBUG_FOLDER / filename
    cv2.imwrite(str(path), frame)

    print(f"Debug screenshot saved to: {path}")


def click_once_roblox(x, y):
    """
    Move to the intended location, force a tiny relative
    mouse-position update, then perform exactly one click.
    """

    # Move to target.
    pydirectinput.moveTo(x, y)

    # Give Roblox time to process the new cursor position.
    time.sleep(0.20)

    # Tiny relative nudge and return to the same pixel.
    # This is intended to make Roblox refresh its internal
    # mouse position.
    pydirectinput.moveRel(1, 0)
    time.sleep(0.05)

    pydirectinput.moveRel(-1, 0)
    time.sleep(0.15)

    # Exactly ONE physical-style click:
    # one button-down followed by one button-up.
    pydirectinput.mouseDown(button="left")
    time.sleep(0.10)
    pydirectinput.mouseUp(button="left")


def main():
    print("Prototype 3: READY click test - attempt 2")
    print()
    print("Switch to Roblox.")
    print("Make sure the READY button is visible.")
    print()
    print("Starting in 5 seconds...")

    time.sleep(5)

    try:
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
                print("No mouse input was sent.")

                save_debug_image(
                    frame,
                    "ready_not_found_before_click.png"
                )
                return

            print()
            print("READY FOUND")
            print("Precondition passed.")

            # -------------------------
            # ACTION
            # -------------------------

            print(
                f"Moving to READY at "
                f"({READY_CLICK_X}, {READY_CLICK_Y})..."
            )

            print("Sending exactly one click...")

            click_once_roblox(
                READY_CLICK_X,
                READY_CLICK_Y
            )

            print("Click sequence finished.")

            # -------------------------
            # POSTCONDITION
            # -------------------------

            print("Waiting for READY to disappear...")

            start_time = time.monotonic()
            consecutive_not_found = 0

            while time.monotonic() - start_time < POST_CLICK_TIMEOUT:

                frame = camera.grab(new_frame_only=False)

                if frame is None:
                    time.sleep(0.1)
                    continue

                if ready_visible(frame):
                    consecutive_not_found = 0
                else:
                    consecutive_not_found += 1

                if consecutive_not_found >= 3:
                    print()
                    print("SUCCESS")
                    print("READY disappeared after one click.")
                    print("Prototype 3 passed.")
                    return

                time.sleep(0.1)

            # -------------------------
            # FAILURE
            # -------------------------

            print()
            print("FAILED")
            print(
                "READY was still detected after the timeout."
            )
            print("The program will NOT click again.")

            save_debug_image(
                frame,
                "ready_still_visible_after_click.png"
            )

    except Exception as exc:
        print()
        print("ERROR")
        print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()