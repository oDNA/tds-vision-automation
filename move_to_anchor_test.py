import time
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pydirectinput


OUTPUT_FILE = Path(__file__).with_name("anchor_candidate.png")
BEFORE_FILE = Path(__file__).with_name("anchor_before_move.png")


# READY detector
READY_X1 = 930
READY_Y1 = 160
READY_X2 = 1120
READY_Y2 = 290


def ready_visible(frame):
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


def roblox_camera_drag():
    """
    Reproduce the camera movement from the Pulover recording.
    """

    # Disable PyDirectInput's automatic delay between calls.
    pydirectinput.PAUSE = 0

    # Put cursor at the approximate original starting point.
    pydirectinput.moveTo(1320, 525)

    # Tiny nudge so Roblox updates its internal mouse position.
    pydirectinput.moveRel(1, 0)
    time.sleep(0.05)
    pydirectinput.moveRel(-1, 0)
    time.sleep(0.10)

    # --------------------------------------------------
    # First recorded right-drag:
    #
    # (1320,525) -> (1326,585)
    # approximately +6 X and +60 Y over 0.875 seconds.
    # --------------------------------------------------

    pydirectinput.mouseDown(button="right")

    steps = 60
    step_delay = 0.875 / steps

    for step in range(1, steps + 1):

        # Move downward by one pixel each step.
        dx = 0
        dy = 1

        # Add six total horizontal pixels across the drag.
        if step % 10 == 0:
            dx = 1

        pydirectinput.moveRel(dx, dy)
        time.sleep(step_delay)

    pydirectinput.mouseUp(button="right")

    # Delay present in the Pulover recording.
    time.sleep(0.890)

    # --------------------------------------------------
    # Second recorded right-mouse action.
    #
    # This only moved about 1 pixel, but we're preserving
    # it during the first reproduction test.
    # --------------------------------------------------

    pydirectinput.moveTo(1326, 575)

    time.sleep(0.10)

    pydirectinput.mouseDown(button="right")

    time.sleep(1.407)

    pydirectinput.moveRel(0, 1)

    pydirectinput.mouseUp(button="right")

    time.sleep(0.343)


def move_player_to_anchor():
    """
    The Pulover keyboard-repeat sequence corresponds to
    holding S for approximately 3.0 seconds.
    """

    pydirectinput.keyDown("s")

    try:
        time.sleep(3.0)
    finally:
        # Always release S even if something unexpected happens.
        pydirectinput.keyUp("s")


def main():
    print("Prototype 5A-1: Anchor movement reproduction")
    print()
    print("Before starting:")
    print("  - Be at the INITIAL spawn position")
    print("  - Do not manually rotate the camera")
    print("  - READY should still be visible")
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
                print("READY was not detected at the starting position.")
                print("No movement was sent.")
                return

            print()
            print("READY FOUND")
            print("Starting-state check passed.")

            cv2.imwrite(str(BEFORE_FILE), frame)

            # -------------------------
            # CAMERA
            # -------------------------

            print()
            print("Reproducing recorded camera movement...")

            roblox_camera_drag()

            print("Camera sequence finished.")

            # -------------------------
            # PLAYER MOVEMENT
            # -------------------------

            print("Holding S for 3.0 seconds...")

            move_player_to_anchor()

            print("Player movement finished.")

            # Give the camera/game a moment to settle.
            time.sleep(0.75)

            # -------------------------
            # CAPTURE RESULT
            # -------------------------

            frame = camera.grab(new_frame_only=False)

            if frame is None:
                print("ERROR: Could not capture final view.")
                return

            success = cv2.imwrite(
                str(OUTPUT_FILE),
                frame
            )

            if not success:
                print("ERROR: Could not save anchor screenshot.")
                return

            print()
            print("MOVEMENT TEST COMPLETE")
            print(f"Saved starting view to:")
            print(BEFORE_FILE)
            print()
            print(f"Saved resulting anchor view to:")
            print(OUTPUT_FILE)
            print()
            print("No READY click or tower input was sent.")

    except Exception as exc:
        # Best effort to ensure keys/buttons aren't left held.
        try:
            pydirectinput.keyUp("s")
            pydirectinput.mouseUp(button="right")
        except Exception:
            pass

        print()
        print("ERROR")
        print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()