import time

import cv2
import dxcam
import numpy as np


# RESTART search region for a 1920x1080 screen.
# Calibrated from your game-over screenshots.
RESTART_X1 = 680
RESTART_Y1 = 730
RESTART_X2 = 990
RESTART_Y2 = 840


def restart_visible(frame):
    """
    Return True if a large green Restart Match button-like
    region is visible inside the expected RESTART area.
    """

    # Crop only the area where Restart Match appears.
    roi = frame[
        RESTART_Y1:RESTART_Y2,
        RESTART_X1:RESTART_X2
    ]

    # Convert OpenCV BGR image to HSV.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Bright-green range.
    # This is the same general range that worked for READY.
    lower_green = np.array([45, 120, 80], dtype=np.uint8)
    upper_green = np.array([75, 255, 255], dtype=np.uint8)

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Find connected green areas.
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            green_mask,
            connectivity=8
        )
    )

    # Component 0 is the background, so skip it.
    for i in range(1, num_labels):
        x, y, width, height, area = stats[i]

        # The actual Restart Match button is roughly:
        # width  ~220 px
        # height ~50 px
        #
        # These thresholds intentionally leave tolerance.
        if (
            area >= 4000
            and width >= 140
            and height >= 35
        ):
            return True

    return False


def main():
    print("Prototype 4A: RESTART detector")
    print()
    print("Capturing screen in 5 seconds...")
    print("Switch to Roblox.")

    time.sleep(5)

    try:
        with dxcam.create(
            backend="winrt",
            output_color="BGR",
        ) as camera:

            frame = camera.grab(new_frame_only=False)

        if frame is None:
            print("ERROR: Could not capture screen.")
            return

        if restart_visible(frame):
            print("RESTART FOUND")
        else:
            print("RESTART NOT FOUND")

    except Exception as exc:
        print()
        print("ERROR")
        print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()