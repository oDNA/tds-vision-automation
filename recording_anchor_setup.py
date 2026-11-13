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

# Successful player route.
A_HOLD_SECONDS = 0.625
AFTER_A_PAUSE_SECONDS = 0.547

S_HOLD_SECONDS = 2.000
JUMP_AFTER_S_SECONDS = 0.140
SPACE_HOLD_SECONDS = 0.080

AFTER_S_PAUSE_SECONDS = 0.422

D_HOLD_SECONDS = 0.968

CAMERA_RECOVERY_SECONDS = 1.5


# ============================================================
# ABSOLUTE CAMERA NORMALIZATION
# ============================================================

# Force camera to maximum zoom-out.
MAX_ZOOM_OUT_PRESSES = 20

ZOOM_KEY_HOLD_SECONDS = 0.080
ZOOM_BETWEEN_PRESSES_SECONDS = 0.120
ZOOM_SETTLE_SECONDS = 1.0


# Force camera against the maximum top-down pitch limit.
PITCH_STEP_DY = 120
PITCH_SATURATION_STEPS = 8

PITCH_BETWEEN_STEPS_SECONDS = 0.30
PITCH_SETTLE_SECONDS = 1.0


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
    / "recording_anchor_absolute"
    / RUN_TIMESTAMP
)


# ============================================================
# INPUT
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
# RAW MOUSE
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
# READY DETECTION
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

    mask = cv2.inRange(
        hsv,
        np.array(
            [45, 120, 80],
            dtype=np.uint8,
        ),
        np.array(
            [75, 255, 255],
            dtype=np.uint8,
        ),
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
    wait: float = 0.75,
) -> None:
    time.sleep(wait)

    frame, _timestamp = (
        camera.get_latest_frame(
            with_timestamp=True
        )
    )

    if frame is None:
        raise RuntimeError(
            "DXcam did not provide a fresh frame."
        )

    path = (
        RUN_FOLDER
        / filename
    )

    if not cv2.imwrite(
        str(path),
        frame,
    ):
        raise RuntimeError(
            f"Could not save screenshot: {path}"
        )

    print(
        f"Saved: {path}"
    )


# ============================================================
# KEYBOARD
# ============================================================

def hold_key(
    key: str,
    duration: float,
) -> None:
    pydirectinput.keyDown(key)

    try:
        time.sleep(duration)

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
# CAMERA
# ============================================================

def raw_relative_right_drag(
    dx: int,
    dy: int,
) -> None:
    """
    Genuine relative Roblox camera movement using the
    previously tested raw Windows mouse method.
    """

    set_cursor_position(
        1200,
        500,
    )

    time.sleep(0.20)

    # Tiny activation wiggle.
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


def force_max_zoom_out() -> None:
    print()
    print(
        "Forcing maximum zoom-out..."
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
        "Maximum zoom-out reached."
    )


def force_max_topdown_pitch() -> None:
    print()
    print(
        "Forcing maximum top-down pitch..."
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
        "Maximum top-down pitch reached."
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
        "Absolute Recording Anchor Setup"
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
        "This script will:"
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
        f"  5. Push pitch +"
        f"{PITCH_STEP_DY} "
        f"{PITCH_SATURATION_STEPS} times"
    )

    print()
    print(
        "It will NOT press I."
    )

    print(
        "It will NOT click READY."
    )

    print(
        "It will NOT place towers."
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
            # VERIFY FRESH SPAWN
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
                    "Start from a fresh spawn "
                    "with READY visible."
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
            # ABSOLUTE ZOOM
            # ==========================================

            force_max_zoom_out()

            save_fresh_frame(
                camera,
                "02_max_zoom_out.png",
                wait=1.0,
            )

            # ==========================================
            # ABSOLUTE PITCH
            # ==========================================

            force_max_topdown_pitch()

            save_fresh_frame(
                camera,
                "03_FINAL_RECORDING_ANCHOR.png",
                wait=1.0,
            )

            camera.stop()

            # ==========================================
            # FINISHED
            # ==========================================

            print()
            print(
                "=" * 65
            )

            print(
                "ABSOLUTE RECORDING ANCHOR READY"
            )

            print(
                "=" * 65
            )

            print()
            print(
                "Do NOT:"
            )

            print(
                "  move the character"
            )

            print(
                "  move the camera"
            )

            print(
                "  press I or O"
            )

            print(
                "  scroll"
            )

            print()
            print(
                "Start Pulover recording now."
            )

            print()
            print(
                "Final reference screenshot:"
            )

            print(
                RUN_FOLDER
                / "03_FINAL_RECORDING_ANCHOR.png"
            )

    finally:
        release_inputs()


if __name__ == "__main__":
    main()