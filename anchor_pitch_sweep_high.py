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
    / "anchor_pitch_sweep_high"
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
# SUCCESSFUL ROUTE TIMINGS
# ==================================================

A_HOLD_TIME = 0.625
AFTER_A_PAUSE = 0.547

S_HOLD_TIME = 2.000
JUMP_AFTER_S_START = 0.140

AFTER_S_PAUSE = 0.422

D_HOLD_TIME = 0.968
AFTER_D_PAUSE = 0.813


# ==================================================
# HIGH-PITCH CAMERA SWEEP
# ==================================================
#
# Previous test proved that 120 is still not enough.
#
# This test starts by applying 120 in one movement,
# then adds 30 at a time:
#
#   120
#   150
#   180
#   210
#   240
#   270
#   300
#
# ==================================================

CAMERA_DX = 0

INITIAL_CAMERA_DY = 120
ADDITIONAL_CAMERA_DY = 30

PITCH_LEVELS = [
    120,
    150,
    180,
    210,
    240,
    270,
    300,
]


# ==================================================
# WINDOWS RAW MOUSE INPUT
# ==================================================

MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010

user32 = ctypes.WinDLL(
    "user32",
    use_last_error=True
)


def set_cursor_position(x, y):
    result = user32.SetCursorPos(
        int(x),
        int(y)
    )

    if result == 0:
        raise ctypes.WinError(
            ctypes.get_last_error()
        )


def relative_mouse_move(dx, dy):
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
# READY DETECTOR
# ==================================================

def ready_visible(frame):
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


# ==================================================
# SCREENSHOT
# ==================================================

def save_fresh_frame(camera, filename, wait=1.0):
    """
    Wait for Roblox to render the new camera position,
    then save the latest DXcam frame.
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

    success = cv2.imwrite(
        str(path),
        frame
    )

    if not success:
        raise RuntimeError(
            f"Could not save screenshot: {path}"
        )

    print(f"Saved: {path}")

    return frame, timestamp


# ==================================================
# KEYBOARD INPUT
# ==================================================

def hold_key(key, duration):
    pydirectinput.keyDown(key)

    try:
        time.sleep(duration)

    finally:
        pydirectinput.keyUp(key)


def hold_s_and_jump():
    pydirectinput.keyDown("s")

    start_time = time.perf_counter()

    try:
        # Hold S briefly before jumping.
        time.sleep(
            JUMP_AFTER_S_START
        )

        # Jump while S remains held.
        pydirectinput.keyDown("space")

        try:
            time.sleep(0.08)

        finally:
            pydirectinput.keyUp("space")

        # Continue holding S until about two seconds
        # total have elapsed.
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
    Hold Roblox right-mouse camera control and send
    one genuine relative mouse movement.
    """

    # Place the mouse safely inside the Roblox viewport
    # BEFORE beginning right-mouse camera control.
    set_cursor_position(
        1200,
        500
    )

    time.sleep(0.25)

    # Tiny nudge first.
    relative_mouse_move(
        1,
        0
    )

    time.sleep(0.05)

    relative_mouse_move(
        -1,
        0
    )

    time.sleep(0.10)

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

    # Give the Roblox camera time to settle.
    time.sleep(0.75)


# ==================================================
# MAIN
# ==================================================

def main():
    print(
        "Prototype 5A-15: High camera pitch sweep"
    )

    print()
    print("Start from a fresh INITIAL spawn.")
    print("Use the normal untouched starting camera.")
    print("READY should be visible.")

    print()
    print("This test will:")
    print(
        f"  1. Hold A for {A_HOLD_TIME:.3f} seconds"
    )
    print(
        f"  2. Hold S for ~{S_HOLD_TIME:.3f} seconds "
        "and jump"
    )
    print(
        f"  3. Hold D for {D_HOLD_TIME:.3f} seconds"
    )
    print("  4. Let camera zoom recover naturally")
    print("  5. Test these cumulative pitch values:")

    print(
        "     "
        + ", ".join(
            str(value)
            for value in PITCH_LEVELS
        )
    )

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

            # ==========================================
            # START CONTINUOUS CAPTURE
            # ==========================================

            print()
            print("Starting continuous DXcam capture...")

            camera.start(
                target_fps=CAPTURE_FPS
            )

            time.sleep(0.75)

            # ==========================================
            # VERIFY STARTING STATE
            # ==========================================

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
                RUN_FOLDER
                / "00_start.png"
            )

            if not cv2.imwrite(
                str(start_path),
                start_frame
            ):
                raise RuntimeError(
                    f"Could not save screenshot: {start_path}"
                )

            print(f"Saved: {start_path}")

            # ==========================================
            # ROUTE: A
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
            # ROUTE: S + JUMP
            # ==========================================

            print()
            print("Holding S and jumping...")

            hold_s_and_jump()

            print("S released.")

            time.sleep(
                AFTER_S_PAUSE
            )

            # ==========================================
            # ROUTE: D
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
            # BEFORE CAMERA
            # ==========================================

            print()
            print(
                "Saving final position before camera pitch..."
            )

            save_fresh_frame(
                camera,
                "01_position_before_camera.png",
                wait=1.0
            )

            # ==========================================
            # INITIAL CAMERA PITCH: 120
            # ==========================================

            current_pitch = INITIAL_CAMERA_DY

            print()
            print(
                f"Applying initial camera pitch: "
                f"dy={INITIAL_CAMERA_DY}"
            )

            raw_relative_right_drag(
                CAMERA_DX,
                INITIAL_CAMERA_DY
            )

            save_fresh_frame(
                camera,
                f"02_pitch_{current_pitch:03d}.png",
                wait=1.25
            )

            # ==========================================
            # ADDITIONAL +30 STEPS
            # ==========================================

            for target_pitch in PITCH_LEVELS[1:]:

                print()
                print(
                    f"Increasing cumulative pitch "
                    f"from {current_pitch} "
                    f"to {target_pitch}..."
                )

                raw_relative_right_drag(
                    CAMERA_DX,
                    ADDITIONAL_CAMERA_DY
                )

                current_pitch = target_pitch

                save_fresh_frame(
                    camera,
                    f"02_pitch_{current_pitch:03d}.png",
                    wait=1.25
                )

            # ==========================================
            # DONE
            # ==========================================

            camera.stop()

            print()
            print("HIGH-PITCH SWEEP COMPLETE")

            print()
            print("Created:")
            print("  01_position_before_camera.png")

            for pitch in PITCH_LEVELS:
                print(
                    f"  02_pitch_{pitch:03d}.png"
                )

            print()
            print(
                "No zoom, READY click, tower placement, "
                "or upgrade input was sent."
            )

    except Exception as exc:

        # Release anything that might possibly have
        # remained held after an exception.
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