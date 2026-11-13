import ctypes
import time
from datetime import datetime
from pathlib import Path

import cv2
import dxcam
import numpy as np
import pydirectinput


# ============================================================
# CONFIG
# ============================================================

CAPTURE_FPS = 60


# ============================================================
# SUCCESSFUL PLAYER ROUTE
# ============================================================

A_HOLD_SECONDS = 0.625
AFTER_A_PAUSE_SECONDS = 0.547

S_HOLD_SECONDS = 2.000
JUMP_AFTER_S_SECONDS = 0.140
SPACE_HOLD_SECONDS = 0.080

AFTER_S_PAUSE_SECONDS = 0.422

D_HOLD_SECONDS = 0.968

CAMERA_RECOVERY_SECONDS = 1.5


# ============================================================
# ABSOLUTE ZOOM NORMALIZATION
# ============================================================
#
# We deliberately send far more O presses than should be
# necessary.
#
# Once Roblox reaches its zoom-out limit, additional O presses
# should have no further effect.
# ============================================================

MAX_ZOOM_OUT_PRESSES = 20

ZOOM_KEY_HOLD_SECONDS = 0.080
ZOOM_BETWEEN_PRESSES_SECONDS = 0.120
ZOOM_SETTLE_SECONDS = 1.0


# ============================================================
# ABSOLUTE PITCH NORMALIZATION
# ============================================================
#
# Positive DY was proven to move the camera toward top-down.
#
# Instead of assuming a starting pitch, repeatedly push against
# Roblox's top-down pitch limit.
#
# 8 × 120 = 960 cumulative relative mouse units.
#
# Once the pitch clamp is reached, later movements should simply
# have no additional effect.
# ============================================================

PITCH_STEP_DY = 120
PITCH_SATURATION_STEPS = 8

PITCH_BETWEEN_STEPS_SECONDS = 0.30
PITCH_SETTLE_SECONDS = 1.0


# ============================================================
# ZOOM-IN CALIBRATION
# ============================================================
#
# After establishing MAXIMUM zoom-out, press I one step at a
# time and save a screenshot after each.
#
# 00 = maximum zoom-out
# 01 = max zoom-out + I once
# 02 = max zoom-out + I twice
# ...
# ============================================================

ZOOM_IN_TEST_STEPS = 10

ZOOM_IN_BETWEEN_SECONDS = 0.50
SCREENSHOT_SETTLE_SECONDS = 0.75


# ============================================================
# READY DETECTOR
# ============================================================

READY_X1 = 930
READY_Y1 = 160
READY_X2 = 1120
READY_Y2 = 290


# ============================================================
# OUTPUT
# ============================================================

BASE_FOLDER = Path(__file__).resolve().parent

RUN_TIMESTAMP = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

RUN_FOLDER = (
    BASE_FOLDER
    / "test_output"
    / "absolute_camera_calibration"
    / RUN_TIMESTAMP
)


# ============================================================
# INPUT SETUP
# ============================================================

pydirectinput.PAUSE = 0

user32 = ctypes.WinDLL(
    "user32",
    use_last_error=True,
)


MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_RIGHTDOWN = 0x0008
MOUSEEVENTF_RIGHTUP = 0x0010


# ============================================================
# RAW WINDOWS MOUSE
# ============================================================

def set_cursor_position(
    x: int,
    y: int,
) -> None:
    result = user32.SetCursorPos(
        int(x),
        int(y),
    )

    if result == 0:
        raise ctypes.WinError(
            ctypes.get_last_error()
        )


def relative_mouse_move(
    dx: int,
    dy: int,
) -> None:
    user32.mouse_event(
        MOUSEEVENTF_MOVE,
        int(dx),
        int(dy),
        0,
        0,
    )


def right_mouse_down() -> None:
    user32.mouse_event(
        MOUSEEVENTF_RIGHTDOWN,
        0,
        0,
        0,
        0,
    )


def right_mouse_up() -> None:
    user32.mouse_event(
        MOUSEEVENTF_RIGHTUP,
        0,
        0,
        0,
        0,
    )


# ============================================================
# READY DETECTOR
# ============================================================

def ready_visible(
    frame: np.ndarray,
) -> bool:
    roi = frame[
        READY_Y1:READY_Y2,
        READY_X1:READY_X2,
    ]

    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV,
    )

    lower_green = np.array(
        [45, 120, 80],
        dtype=np.uint8,
    )

    upper_green = np.array(
        [75, 255, 255],
        dtype=np.uint8,
    )

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green,
    )

    (
        num_labels,
        _labels,
        stats,
        _centroids,
    ) = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    for i in range(
        1,
        num_labels,
    ):
        (
            _x,
            _y,
            width,
            height,
            area,
        ) = stats[i]

        if (
            area >= 4000
            and width >= 120
            and height >= 35
        ):
            return True

    return False


# ============================================================
# SCREENSHOTS
# ============================================================

def save_fresh_frame(
    camera,
    filename: str,
    wait: float = SCREENSHOT_SETTLE_SECONDS,
):
    time.sleep(wait)

    frame, timestamp = (
        camera.get_latest_frame(
            with_timestamp=True
        )
    )

    if frame is None:
        raise RuntimeError(
            "DXcam did not provide a fresh frame."
        )

    path = RUN_FOLDER / filename

    success = cv2.imwrite(
        str(path),
        frame,
    )

    if not success:
        raise RuntimeError(
            f"Could not save screenshot: {path}"
        )

    print(
        f"Saved: {path}"
    )

    return frame, timestamp


# ============================================================
# KEYBOARD INPUT
# ============================================================

def hold_key(
    key: str,
    seconds: float,
) -> None:
    pydirectinput.keyDown(key)

    try:
        time.sleep(seconds)

    finally:
        pydirectinput.keyUp(key)


def tap_key(
    key: str,
    hold_seconds: float = 0.080,
) -> None:
    pydirectinput.keyDown(key)

    try:
        time.sleep(
            hold_seconds
        )

    finally:
        pydirectinput.keyUp(key)


def hold_s_and_jump() -> None:
    pydirectinput.keyDown("s")

    started = time.perf_counter()

    try:
        time.sleep(
            JUMP_AFTER_S_SECONDS
        )

        pydirectinput.keyDown(
            "space"
        )

        try:
            time.sleep(
                SPACE_HOLD_SECONDS
            )

        finally:
            pydirectinput.keyUp(
                "space"
            )

        elapsed = (
            time.perf_counter()
            - started
        )

        remaining = (
            S_HOLD_SECONDS
            - elapsed
        )

        if remaining > 0:
            time.sleep(
                remaining
            )

    finally:
        pydirectinput.keyUp(
            "space"
        )

        pydirectinput.keyUp(
            "s"
        )


# ============================================================
# CAMERA INPUT
# ============================================================

def raw_relative_right_drag(
    dx: int,
    dy: int,
) -> None:
    """
    Genuine relative Roblox camera movement using the method
    that worked in our earlier pitch calibration.
    """

    set_cursor_position(
        1200,
        500,
    )

    time.sleep(0.20)

    # Small nudge before RMB capture.
    relative_mouse_move(
        1,
        0,
    )

    time.sleep(0.05)

    relative_mouse_move(
        -1,
        0,
    )

    time.sleep(0.10)

    right_mouse_down()

    try:
        time.sleep(0.20)

        relative_mouse_move(
            dx,
            dy,
        )

        time.sleep(0.20)

    finally:
        right_mouse_up()

    time.sleep(
        PITCH_BETWEEN_STEPS_SECONDS
    )


# ============================================================
# ABSOLUTE ZOOM
# ============================================================

def force_max_zoom_out() -> None:
    print()
    print(
        "Forcing camera to MAXIMUM zoom-out..."
    )

    for press in range(
        1,
        MAX_ZOOM_OUT_PRESSES + 1,
    ):
        tap_key(
            "o",
            ZOOM_KEY_HOLD_SECONDS,
        )

        print(
            f"  O press "
            f"{press}/"
            f"{MAX_ZOOM_OUT_PRESSES}"
        )

        time.sleep(
            ZOOM_BETWEEN_PRESSES_SECONDS
        )

    time.sleep(
        ZOOM_SETTLE_SECONDS
    )

    print(
        "Maximum zoom-out normalization finished."
    )


# ============================================================
# ABSOLUTE PITCH
# ============================================================

def force_max_topdown_pitch() -> None:
    print()
    print(
        "Forcing camera to MAXIMUM top-down pitch..."
    )

    cumulative = 0

    for step in range(
        1,
        PITCH_SATURATION_STEPS + 1,
    ):
        raw_relative_right_drag(
            0,
            PITCH_STEP_DY,
        )

        cumulative += PITCH_STEP_DY

        print(
            f"  pitch push "
            f"{step}/"
            f"{PITCH_SATURATION_STEPS} "
            f"(cumulative dy={cumulative})"
        )

    time.sleep(
        PITCH_SETTLE_SECONDS
    )

    print(
        "Top-down pitch normalization finished."
    )


# ============================================================
# SAFETY
# ============================================================

def release_inputs() -> None:
    for key in [
        "a",
        "s",
        "d",
        "space",
        "o",
        "i",
    ]:
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "Prototype 5A-16: Absolute camera calibration"
    )

    print()
    print(
        "Start from a fresh Wretched Front spawn."
    )

    print(
        "READY must be visible."
    )

    print()
    print(
        "This test will:"
    )

    print(
        f"  1. Hold A for "
        f"{A_HOLD_SECONDS:.3f}s"
    )

    print(
        f"  2. Hold S for "
        f"{S_HOLD_SECONDS:.3f}s + jump"
    )

    print(
        f"  3. Hold D for "
        f"{D_HOLD_SECONDS:.3f}s"
    )

    print(
        f"  4. Press O "
        f"{MAX_ZOOM_OUT_PRESSES} times"
    )

    print(
        f"  5. Push camera downward "
        f"{PITCH_SATURATION_STEPS} × "
        f"{PITCH_STEP_DY}"
    )

    print(
        "  6. Save maximum-zoom / "
        "maximum-pitch reference"
    )

    print(
        f"  7. Press I one at a time "
        f"through I × {ZOOM_IN_TEST_STEPS}"
    )

    print(
        "  8. Save a screenshot after each zoom level"
    )

    print()
    print(
        "It will NOT click READY."
    )

    print(
        "It will NOT place towers."
    )

    print()
    print(
        "Output folder:"
    )

    print(
        RUN_FOLDER
    )

    print()
    print(
        "Starting in 5 seconds..."
    )

    time.sleep(5)

    RUN_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with dxcam.create(
            backend="winrt",
            output_color="BGR",
        ) as camera:

            camera.start(
                target_fps=CAPTURE_FPS
            )

            time.sleep(0.75)

            # ==========================================
            # START CHECK
            # ==========================================

            frame, _timestamp = (
                camera.get_latest_frame(
                    with_timestamp=True
                )
            )

            if frame is None:
                raise RuntimeError(
                    "Could not capture Roblox."
                )

            if not ready_visible(frame):
                raise RuntimeError(
                    "READY was not detected. "
                    "Start from a fresh spawn."
                )

            print()
            print(
                "READY FOUND."
            )

            print(
                "READY will remain untouched."
            )

            start_path = (
                RUN_FOLDER
                / "00_start.png"
            )

            cv2.imwrite(
                str(start_path),
                frame,
            )

            print(
                f"Saved: {start_path}"
            )

            # ==========================================
            # PLAYER ROUTE
            # ==========================================

            print()
            print(
                f"Holding A for "
                f"{A_HOLD_SECONDS:.3f}s..."
            )

            hold_key(
                "a",
                A_HOLD_SECONDS,
            )

            time.sleep(
                AFTER_A_PAUSE_SECONDS
            )

            print(
                "Holding S and jumping..."
            )

            hold_s_and_jump()

            time.sleep(
                AFTER_S_PAUSE_SECONDS
            )

            print(
                f"Holding D for "
                f"{D_HOLD_SECONDS:.3f}s..."
            )

            hold_key(
                "d",
                D_HOLD_SECONDS,
            )

            print(
                f"Waiting "
                f"{CAMERA_RECOVERY_SECONDS:.1f}s "
                f"for camera recovery..."
            )

            time.sleep(
                CAMERA_RECOVERY_SECONDS
            )

            save_fresh_frame(
                camera,
                "01_after_route.png",
                wait=0.75,
            )

            # ==========================================
            # FORCE ABSOLUTE MAX ZOOM
            # ==========================================

            force_max_zoom_out()

            save_fresh_frame(
                camera,
                "02_max_zoom_before_pitch.png",
                wait=1.0,
            )

            # ==========================================
            # FORCE ABSOLUTE MAX PITCH
            # ==========================================

            force_max_topdown_pitch()

            save_fresh_frame(
                camera,
                "03_zoom_00_MAX_OUT_max_pitch.png",
                wait=1.0,
            )

            # ==========================================
            # WALK INWARD ONE ZOOM STEP AT A TIME
            # ==========================================

            print()
            print(
                "Beginning I zoom-in calibration..."
            )

            for level in range(
                1,
                ZOOM_IN_TEST_STEPS + 1,
            ):
                tap_key(
                    "i",
                    ZOOM_KEY_HOLD_SECONDS,
                )

                time.sleep(
                    ZOOM_IN_BETWEEN_SECONDS
                )

                filename = (
                    f"03_zoom_{level:02d}"
                    f"_I_x{level:02d}.png"
                )

                save_fresh_frame(
                    camera,
                    filename,
                    wait=SCREENSHOT_SETTLE_SECONDS,
                )

                print(
                    f"  Completed I × {level}"
                )

            camera.stop()

            print()
            print(
                "=" * 65
            )

            print(
                "ABSOLUTE CAMERA CALIBRATION COMPLETE"
            )

            print(
                "=" * 65
            )

            print()
            print(
                "Compare these screenshots:"
            )

            print()
            print(
                "  03_zoom_00_MAX_OUT_max_pitch.png"
            )

            for level in range(
                1,
                ZOOM_IN_TEST_STEPS + 1,
            ):
                print(
                    f"  03_zoom_{level:02d}"
                    f"_I_x{level:02d}.png"
                )

            print()
            print(
                "Choose the view that is easiest "
                "for your tower placements."
            )

            print()
            print(
                "For example, if I × 4 looks best, "
                "our permanent camera setup becomes:"
            )

            print()
            print(
                f"  O × {MAX_ZOOM_OUT_PRESSES}"
            )

            print(
                f"  top-down pitch saturation "
                f"({PITCH_SATURATION_STEPS} pushes)"
            )

            print(
                "  I × 4"
            )

            print()
            print(
                "That state should be reproducible "
                "regardless of Roblox's starting camera."
            )

    finally:
        release_inputs()


if __name__ == "__main__":
    main()