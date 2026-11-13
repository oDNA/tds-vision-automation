import ctypes
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
    / "anchor_setup"
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

# These are the timings that worked well in Prototype 5A-12.

A_HOLD_TIME = 0.625
AFTER_A_PAUSE = 0.547

S_HOLD_TIME = 2.000
JUMP_AFTER_S_START = 0.140

AFTER_S_PAUSE = 0.422

D_HOLD_TIME = 0.968

AFTER_D_PAUSE = 0.813


# ==================================================
# CAMERA SETTINGS
# ==================================================

# Pure vertical relative camera movement.
#
# We already confirmed that +60 produces a substantial
# downward/top-down pitch change without rotating sideways.

CAMERA_DX = 0
CAMERA_DY = 60


# ==================================================
# WINDOWS RAW MOUSE
# ==================================================

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

user32 = ctypes.WinDLL(
    "user32",
    use_last_error=True
)


def set_cursor_position(x, y):
    """
    Move the cursor before Roblox captures the mouse.
    """

    result = user32.SetCursorPos(
        int(x),
        int(y)
    )

    if result == 0:
        raise ctypes.WinError(
            ctypes.get_last_error()
        )


def relative_mouse_move(dx, dy):
    """
    Send genuine relative Windows mouse movement.
    """

    user32.mouse_event(
        MOUSEEVENTF_MOVE,
        int(dx),
        int(dy),
        0,
        0
    )


def right_mouse_down():
    user32.mouse_event(
        MOUSEEVENTF_RIGHTDOWN,
        0,
        0,
        0,
        0
    )


def right_mouse_up():
    user32.mouse_event(
        MOUSEEVENTF_RIGHTUP,
        0,
        0,
        0,
        0
    )


# ==================================================
# VISION
# ==================================================

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


def save_fresh_frame(camera, filename, wait=0.75):
    """
    Wait for newly rendered frames and save the latest one.
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

    return frame, timestamp


# ==================================================
# PLAYER INPUT
# ==================================================

def hold_key(key, duration):
    """
    Hold one keyboard key continuously.
    """

    pydirectinput.keyDown(key)

    try:
        time.sleep(duration)

    finally:
        pydirectinput.keyUp(key)


def hold_s_and_jump():
    """
    Hold S for about two seconds.

    Tap Space 140 ms after S begins while continuing
    to hold S.
    """

    pydirectinput.keyDown("s")

    start_time = time.perf_counter()

    try:
        # First part of backward movement.
        time.sleep(
            JUMP_AFTER_S_START
        )

        # Jump without releasing S.
        pydirectinput.keyDown("space")

        try:
            time.sleep(0.08)

        finally:
            pydirectinput.keyUp("space")

        # Keep S held until approximately two seconds
        # have elapsed in total.
        elapsed = (
            time.perf_counter()
            - start_time
        )

        remaining = (
            S_HOLD_TIME
            - elapsed
        )

        if remaining > 0:
            time.sleep(
                remaining
            )

    finally:
        pydirectinput.keyUp("s")


# ==================================================
# CAMERA INPUT
# ==================================================

def raw_relative_right_drag(dx, dy):
    """
    Perform one genuine relative camera movement.

    No absolute cursor positioning occurs while
    right mouse is held.
    """

    # Put cursor safely inside Roblox first.
    set_cursor_position(
        1200,
        500
    )

    time.sleep(0.25)

    # Tiny refresh nudge.
    relative_mouse_move(1, 0)
    time.sleep(0.05)

    relative_mouse_move(-1, 0)
    time.sleep(0.10)

    # Begin camera control.
    right_mouse_down()

    try:
        time.sleep(0.20)

        relative_mouse_move(
            dx,
            dy
        )

        time.sleep(0.20)

    finally:
        right_mouse_up()

    # Let Roblox finish updating the camera.
    time.sleep(0.75)


# ==================================================
# MAIN
# ==================================================

def main():
    print(
        "Prototype 5A-13: Route + anchor camera setup"
    )

    print()
    print("Start from a fresh INITIAL spawn.")
    print("Use the normal untouched starting camera.")
    print("READY should be visible.")

    print()
    print("This test will:")
    print(
        f"  1. Hold A for {A_HOLD_TIME:.3f} s"
    )
    print(
        f"  2. Hold S for ~{S_HOLD_TIME:.3f} s "
        "and jump"
    )
    print(
        f"  3. Hold D for {D_HOLD_TIME:.3f} s"
    )
    print("  4. Allow the camera zoom to recover naturally")
    print(
        f"  5. Apply camera delta "
        f"dx={CAMERA_DX}, dy={CAMERA_DY}"
    )
    print("  6. Save the resulting anchor view")

    print()
    print("It will NOT:")
    print("  - press I or O")
    print("  - click READY")
    print("  - place towers")
    print("  - upgrade towers")

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
            # STARTING CHECK
            # ------------------------------------------

            start_frame, start_timestamp = (
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
            # A
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

            time.sleep(
                AFTER_A_PAUSE
            )

            # ==========================================
            # S + JUMP
            # ==========================================

            print()
            print("Holding S and jumping...")

            hold_s_and_jump()

            print("S released.")

            time.sleep(
                AFTER_S_PAUSE
            )

            # ==========================================
            # D
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

            time.sleep(
                AFTER_D_PAUSE
            )

            # ==========================================
            # SAVE POSITION BEFORE CAMERA ADJUSTMENT
            # ==========================================

            before_camera, before_timestamp = (
                save_fresh_frame(
                    camera,
                    "01_position_before_camera.png",
                    wait=1.0
                )
            )

            # ==========================================
            # CAMERA PITCH
            # ==========================================

            print()
            print(
                "Applying raw relative camera adjustment..."
            )

            print(
                f"dx={CAMERA_DX}, "
                f"dy={CAMERA_DY}"
            )

            raw_relative_right_drag(
                CAMERA_DX,
                CAMERA_DY
            )

            # ==========================================
            # SAVE FINAL ANCHOR
            # ==========================================

            after_camera, after_timestamp = (
                save_fresh_frame(
                    camera,
                    "02_anchor_candidate.png",
                    wait=1.5
                )
            )

            # Simple confirmation that a new frame
            # really reflects a changed image.
            if (
                before_camera.shape
                == after_camera.shape
            ):
                difference = cv2.absdiff(
                    before_camera,
                    after_camera
                )

                mean_difference = float(
                    difference.mean()
                )

                print()
                print(
                    f"Mean camera-change image difference: "
                    f"{mean_difference:.2f}"
                )

            print(
                f"Frame timestamp difference: "
                f"{after_timestamp - before_timestamp:.3f} s"
            )

            camera.stop()

            # ==========================================
            # FINISHED
            # ==========================================

            print()
            print("ANCHOR SETUP TEST COMPLETE")

            print()
            print("Compare:")
            print(
                "  01_position_before_camera.png"
            )
            print(
                "  02_anchor_candidate.png"
            )

            print()
            print(
                "No zoom command, READY click, "
                "or tower input was sent."
            )

    except Exception as exc:

        # Safety releases.
        for key in (
            "a",
            "s",
            "d",
            "space"
        ):
            try:
                pydirectinput.keyUp(
                    key
                )
            except Exception:
                pass

        try:
            right_mouse_up()
        except Exception:
            pass

        print()
        print("ERROR")
        print(
            f"{type(exc).__name__}: {exc}"
        )


if __name__ == "__main__":
    main()