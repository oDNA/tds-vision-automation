import time
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pydirectinput


# --------------------------------------------------
# RESTART detector configuration - 1920x1080
# --------------------------------------------------

RESTART_X1 = 680
RESTART_Y1 = 730
RESTART_X2 = 990
RESTART_Y2 = 840

# Approximate center of the green "Restart Match (0/1)" button.
RESTART_CLICK_X = 835
RESTART_CLICK_Y = 795

# Maximum time to wait for the Restart Match button to disappear.
POST_CLICK_TIMEOUT = 10.0

DEBUG_FOLDER = Path(__file__).with_name("debug")


def restart_visible(frame):
    """
    Return True when the green Restart Match button
    is visible in its expected region.
    """

    roi = frame[
        RESTART_Y1:RESTART_Y2,
        RESTART_X1:RESTART_X2
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
            and width >= 140
            and height >= 35
        ):
            return True

    return False


def save_debug_image(frame, filename):
    """
    Save a screenshot when a test fails.
    """

    DEBUG_FOLDER.mkdir(exist_ok=True)

    path = DEBUG_FOLDER / filename
    cv2.imwrite(str(path), frame)

    print(f"Debug screenshot saved to: {path}")


def click_once_roblox(x, y):
    """
    Perform the Roblox-compatible click sequence that worked
    during Prototype 3.

    This still produces exactly one left mouse click.
    """

    # Move to the target.
    pydirectinput.moveTo(x, y)

    # Give Roblox time to register the cursor position.
    time.sleep(0.20)

    # Tiny relative movement to make Roblox refresh its
    # internal mouse position.
    pydirectinput.moveRel(1, 0)
    time.sleep(0.05)

    pydirectinput.moveRel(-1, 0)
    time.sleep(0.15)

    # Exactly one click.
    pydirectinput.mouseDown(button="left")
    time.sleep(0.10)
    pydirectinput.mouseUp(button="left")


def main():
    print("Prototype 4B: RESTART click test")
    print()
    print("Switch to Roblox.")
    print("Make sure the game-over screen and")
    print("'Restart Match (0/1)' button are visible.")
    print()
    print("Starting in 5 seconds...")

    time.sleep(5)

    try:
        with dxcam.create(
            backend="winrt",
            output_color="BGR",
        ) as camera:

            # --------------------------------
            # PRECONDITION
            # --------------------------------

            frame = camera.grab(new_frame_only=False)

            if frame is None:
                print()
                print("ERROR: Could not capture screen.")
                return

            if not restart_visible(frame):
                print()
                print("ABORTED")
                print("RESTART was not detected.")
                print("No mouse input was sent.")

                save_debug_image(
                    frame,
                    "restart_not_found_before_click.png"
                )
                return

            print()
            print("RESTART FOUND")
            print("Precondition passed.")

            # --------------------------------
            # ACTION
            # --------------------------------

            print(
                f"Moving to Restart Match at "
                f"({RESTART_CLICK_X}, {RESTART_CLICK_Y})..."
            )

            print("Sending exactly one click...")

            click_once_roblox(
                RESTART_CLICK_X,
                RESTART_CLICK_Y
            )

            print("Click sequence finished.")

            # --------------------------------
            # POSTCONDITION
            # --------------------------------

            print("Waiting for RESTART to disappear...")

            start_time = time.monotonic()
            consecutive_not_found = 0
            last_frame = frame

            while time.monotonic() - start_time < POST_CLICK_TIMEOUT:

                frame = camera.grab(new_frame_only=False)

                if frame is None:
                    time.sleep(0.1)
                    continue

                last_frame = frame

                if restart_visible(frame):
                    consecutive_not_found = 0
                else:
                    consecutive_not_found += 1

                # Require three consecutive frames without
                # the Restart Match button.
                if consecutive_not_found >= 3:
                    print()
                    print("SUCCESS")
                    print("RESTART disappeared after one click.")
                    print("Prototype 4 passed.")
                    return

                time.sleep(0.1)

            # --------------------------------
            # FAILURE
            # --------------------------------

            print()
            print("FAILED")
            print("RESTART was still detected after the timeout.")
            print("The program will NOT click again.")

            save_debug_image(
                last_frame,
                "restart_still_visible_after_click.png"
            )

    except Exception as exc:
        print()
        print("ERROR")
        print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()