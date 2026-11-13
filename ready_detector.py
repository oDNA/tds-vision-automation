import time

import cv2
import dxcam
import numpy as np


# READY search region on a 1920x1080 screen.
READY_X1 = 930
READY_Y1 = 160
READY_X2 = 1120
READY_Y2 = 290


def ready_visible(frame):
    """
    Returns True if a large green READY-button-like region
    is visible inside the READY ROI.
    """

    # Crop only the part of the screen where READY can appear.
    roi = frame[
        READY_Y1:READY_Y2,
        READY_X1:READY_X2
    ]

    # Convert from OpenCV BGR colors to HSV.
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Detect the bright green used by the READY button.
    lower_green = np.array([45, 120, 80], dtype=np.uint8)
    upper_green = np.array([75, 255, 255], dtype=np.uint8)

    green_mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    # Find connected green regions.
    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            green_mask,
            connectivity=8
        )
    )

    # Skip component 0 because it represents the background.
    for i in range(1, num_labels):
        x, y, width, height, area = stats[i]

        # The READY button in your screenshot is approximately:
        # width  = 158 px
        # height = 55 px
        # green area = 7161 px
        #
        # These thresholds deliberately leave some tolerance.
        if (
            area >= 4000
            and width >= 120
            and height >= 35
        ):
            return True

    return False


def main():
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

        if ready_visible(frame):
            print("READY FOUND")
        else:
            print("READY NOT FOUND")

    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()