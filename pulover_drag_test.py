import time
from datetime import datetime
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pydirectinput


# --------------------------------------------------
# Output folder
# --------------------------------------------------

BASE_FOLDER = Path(__file__).parent

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_FOLDER = (
    BASE_FOLDER
    / "test_output"
    / "pulover_drag"
    / RUN_TIMESTAMP
)


# --------------------------------------------------
# READY detector
# --------------------------------------------------

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


def capture_and_save(camera, filename):
    """
    Capture the screen and save it into this run's folder.
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


def refresh_mouse_position():
    """
    Tiny nudge before using the mouse.

    This is the same Roblox compatibility trick that
    made our READY/RESTART clicks reliable.
    """

    pydirectinput.moveRel(1, 0)
    time.sleep(0.05)

    pydirectinput.moveRel(-1, 0)
    time.sleep(0.10)


def right_drag_absolute(
    start_x,
    start_y,
    end_x,
    end_y,
    hold_time
):
    """
    Reproduce a Pulover-style right mouse drag using
    absolute start/end screen coordinates.

    IMPORTANT:
    Unlike our previous tests, this does NOT repeatedly
    call moveRel() while the right button is held.
    """

    # Move to the exact recorded starting position.
    pydirectinput.moveTo(
        start_x,
        start_y
    )

    time.sleep(0.15)

    refresh_mouse_position()

    # Begin right mouse drag.
    pydirectinput.mouseDown(
        button="right"
    )

    try:
        # Approximate the time between Pulover's
        # Right Down and Right Up events.
        time.sleep(hold_time)

        # Move directly to the recorded endpoint.
        pydirectinput.moveTo(
            end_x,
            end_y
        )

        # Give Roblox a moment to process that movement
        # before releasing the button.
        time.sleep(0.15)

    finally:
        pydirectinput.mouseUp(
            button="right"
        )

    time.sleep(0.75)


def main():
    print("Prototype 5A-7: Pulover camera drag reproduction")

    print()
    print("Start from a fresh INITIAL spawn.")
    print("Do not manually move, zoom, or rotate the camera.")
    print("READY should be visible.")

    print()
    print("This test will reproduce ONLY the two")
    print("right-mouse camera actions from your Pulover macro.")

    print()
    print("It will NOT:")
    print("  - move the player")
    print("  - zoom")
    print("  - click READY")
    print("  - place towers")

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
                print("No input was sent.")
                return

            print()
            print("READY FOUND")
            print("Starting-state check passed.")

            capture_and_save(
                camera,
                "00_before_drag.png"
            )

            # ------------------------------------------
            # FIRST PULOVER DRAG
            #
            # Recorded:
            #
            # Click, 1320, 525 Right, Down
            # Sleep, 875
            # Click, 1326, 585 Right, Up
            # ------------------------------------------

            print()
            print("Performing first Pulover camera drag:")
            print("  (1320,525) -> (1326,585)")

            right_drag_absolute(
                start_x=1320,
                start_y=525,
                end_x=1326,
                end_y=585,
                hold_time=0.875
            )

            capture_and_save(
                camera,
                "01_after_first_drag.png"
            )

            # Original Pulover delay.
            time.sleep(0.890)

            # ------------------------------------------
            # SECOND PULOVER DRAG
            #
            # Recorded:
            #
            # Click, 1326, 575 Right, Down
            # Sleep, 1407
            # Click, 1326, 576 Right, Up
            # ------------------------------------------

            print()
            print("Performing second Pulover camera drag:")
            print("  (1326,575) -> (1326,576)")

            right_drag_absolute(
                start_x=1326,
                start_y=575,
                end_x=1326,
                end_y=576,
                hold_time=1.407
            )

            capture_and_save(
                camera,
                "02_after_second_drag.png"
            )

            # ------------------------------------------
            # FINISHED
            # ------------------------------------------

            print()
            print("PULOVER DRAG TEST COMPLETE")

            print()
            print("Compare:")
            print("  00_before_drag.png")
            print("  01_after_first_drag.png")
            print("  02_after_second_drag.png")

            print()
            print("No player movement, zoom, READY click,")
            print("or tower input was sent.")

    except Exception as exc:

        try:
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