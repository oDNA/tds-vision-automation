import time
from datetime import datetime
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pydirectinput


# ==================================================
# OUTPUT
# ==================================================

BASE_FOLDER = Path(__file__).parent

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

RUN_FOLDER = (
    BASE_FOLDER
    / "test_output"
    / "route_hold"
    / RUN_TIMESTAMP
)


# ==================================================
# READY DETECTOR
# ==================================================

READY_X1 = 930
READY_Y1 = 160
READY_X2 = 1120
READY_Y2 = 290

CAPTURE_FPS = 60


# ==================================================
# ROUTE TIMINGS
# ==================================================

# Estimated from Pulover keyboard-repeat timing.

A_HOLD_TIME = 0.625
AFTER_A_PAUSE = 0.547

S_HOLD_TIME = 2.000

# Space was pressed 140 ms after S started.
JUMP_AFTER_S_START = 0.140

# Therefore:
#
# 2.000 - 0.140 = 1.860 seconds
#
# remaining after the jump event.
S_AFTER_JUMP_TIME = 1.860

AFTER_S_PAUSE = 0.422

D_HOLD_TIME = 0.968

AFTER_D_PAUSE = 0.813


# ==================================================
# VISION
# ==================================================

def ready_visible(frame):
    """
    Return True if READY is visible.
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


def save_fresh_frame(camera, filename, wait=0.75):
    """
    Wait for fresh rendered frames, then save the latest.
    """

    time.sleep(wait)

    frame, timestamp = camera.get_latest_frame(
        with_timestamp=True
    )

    if frame is None:
        raise RuntimeError(
            "Could not obtain a fresh DXcam frame."
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

    return frame


# ==================================================
# INPUT HELPERS
# ==================================================

def hold_key(key, duration):
    """
    Hold one movement key continuously.
    """

    pydirectinput.keyDown(key)

    try:
        time.sleep(duration)

    finally:
        pydirectinput.keyUp(key)


def tap_key(key, duration=0.08):
    """
    Send one deliberate key press.
    """

    pydirectinput.keyDown(key)

    try:
        time.sleep(duration)

    finally:
        pydirectinput.keyUp(key)


def hold_s_and_jump():
    """
    Reproduce what appears to have been:

        hold S
        wait 140 ms
        jump while still holding S
        continue holding S
        release after total ~2.0 s
    """

    pydirectinput.keyDown("s")

    try:

        # Move backward briefly before jumping.
        time.sleep(
            JUMP_AFTER_S_START
        )

        # Jump WITHOUT releasing S.
        pydirectinput.keyDown("space")

        try:
            time.sleep(0.08)

        finally:
            pydirectinput.keyUp("space")

        # Continue moving backward.
        #
        # Subtract the 80 ms used to hold Space so the
        # overall S duration stays close to the original
        # 2.0 seconds.
        remaining = (
            S_AFTER_JUMP_TIME - 0.08
        )

        time.sleep(
            remaining
        )

    finally:
        pydirectinput.keyUp("s")


# ==================================================
# MAIN
# ==================================================

def main():
    print(
        "Prototype 5A-12: Held-key route replay"
    )

    print()
    print("Start from a fresh INITIAL spawn.")
    print("Use the normal untouched starting camera.")
    print("READY should be visible.")

    print()
    print("Reconstructed physical input:")
    print(
        f"  A held for {A_HOLD_TIME:.3f} s"
    )
    print(
        f"  pause {AFTER_A_PAUSE:.3f} s"
    )
    print(
        f"  S held for ~{S_HOLD_TIME:.3f} s"
    )
    print(
        f"    Space tapped after "
        f"{JUMP_AFTER_S_START:.3f} s"
    )
    print(
        f"  pause {AFTER_S_PAUSE:.3f} s"
    )
    print(
        f"  D held for {D_HOLD_TIME:.3f} s"
    )

    print()
    print("This test will NOT:")
    print("  - adjust camera pitch")
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
            # CONTINUOUS CAPTURE
            # ------------------------------------------

            camera.start(
                target_fps=CAPTURE_FPS
            )

            time.sleep(0.75)

            # ------------------------------------------
            # PRECONDITION
            # ------------------------------------------

            start_frame, timestamp = (
                camera.get_latest_frame(
                    with_timestamp=True
                )
            )

            if start_frame is None:
                print()
                print(
                    "ERROR: Could not obtain starting frame."
                )
                return

            if not ready_visible(start_frame):

                print()
                print("ABORTED")
                print("READY was not detected.")
                print("No movement was sent.")

                return

            print()
            print("READY FOUND")
            print("Starting-state check passed.")

            start_path = (
                RUN_FOLDER / "00_start.png"
            )

            cv2.imwrite(
                str(start_path),
                start_frame
            )

            print(f"Saved: {start_path}")

            # ==========================================
            # A MOVEMENT
            # ==========================================

            print()
            print(
                f"Holding A for "
                f"{A_HOLD_TIME:.3f} seconds..."
            )

            hold_key(
                "a",
                A_HOLD_TIME
            )

            print("A released.")

            save_fresh_frame(
                camera,
                "01_after_a_hold.png"
            )

            print(
                f"Waiting {AFTER_A_PAUSE:.3f} seconds..."
            )

            time.sleep(
                AFTER_A_PAUSE
            )

            # ==========================================
            # S + JUMP
            # ==========================================

            print()
            print(
                "Holding S and jumping..."
            )

            hold_s_and_jump()

            print("S released.")

            save_fresh_frame(
                camera,
                "02_after_s_jump.png"
            )

            print(
                f"Waiting {AFTER_S_PAUSE:.3f} seconds..."
            )

            time.sleep(
                AFTER_S_PAUSE
            )

            # ==========================================
            # D MOVEMENT
            # ==========================================

            print()
            print(
                f"Holding D for "
                f"{D_HOLD_TIME:.3f} seconds..."
            )

            hold_key(
                "d",
                D_HOLD_TIME
            )

            print("D released.")

            # Match the recorded pause before the
            # camera adjustment began.
            time.sleep(
                AFTER_D_PAUSE
            )

            save_fresh_frame(
                camera,
                "03_final_position_before_camera.png",
                wait=1.0
            )

            # ==========================================
            # FINISHED
            # ==========================================

            camera.stop()

            print()
            print("HELD-KEY ROUTE TEST COMPLETE")

            print()
            print("Compare:")
            print("  00_start.png")
            print("  01_after_a_hold.png")
            print("  02_after_s_jump.png")
            print(
                "  03_final_position_before_camera.png"
            )

            print()
            print(
                "No camera adjustment, READY click, "
                "or gameplay action was sent."
            )

    except Exception as exc:

        # Make absolutely sure no key remains held.
        for key in (
            "a",
            "s",
            "d",
            "space"
        ):
            try:
                pydirectinput.keyUp(key)
            except Exception:
                pass

        print()
        print("ERROR")
        print(
            f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    main()