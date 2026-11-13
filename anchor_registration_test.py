from __future__ import annotations

import ctypes
from datetime import datetime
from pathlib import Path
import time

import cv2
import dxcam
import numpy as np
import pydirectinput


# ============================================================
# CONFIG
# ============================================================

BASE = Path(__file__).resolve().parent

STARTUP_DELAY_SECONDS = 5.0

# Successful player route.
A_HOLD_SECONDS = 0.625
AFTER_A_PAUSE_SECONDS = 0.547

S_HOLD_SECONDS = 2.000
JUMP_AFTER_S_SECONDS = 0.140
SPACE_HOLD_SECONDS = 0.080

AFTER_S_PAUSE_SECONDS = 0.422

D_HOLD_SECONDS = 0.968
CAMERA_RECOVERY_SECONDS = 1.5


# Absolute camera normalization.
MAX_ZOOM_OUT_PRESSES = 20
ZOOM_KEY_HOLD_SECONDS = 0.080
ZOOM_BETWEEN_PRESSES_SECONDS = 0.120
ZOOM_SETTLE_SECONDS = 1.0

PITCH_STEP_DY = 120
PITCH_SATURATION_STEPS = 8
PITCH_BETWEEN_STEPS_SECONDS = 0.30
PITCH_SETTLE_SECONDS = 1.0


# ============================================================
# IMAGE REGISTRATION
# ============================================================
#
# This patch is deliberately on static map geometry:
#
# - no READY button
# - no player
# - no tower UI
# - avoids most animated path arrows
#
# The bot searches +/- 80 pixels around the reference position.
# ============================================================

TEMPLATE_X1 = 430
TEMPLATE_Y1 = 180
TEMPLATE_X2 = 760
TEMPLATE_Y2 = 500

SEARCH_MARGIN = 80

MIN_CONFIDENCE = 0.45
MAX_ALLOWED_SHIFT = 80


# ============================================================
# OUTPUT
# ============================================================

timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

OUTPUT = (
    BASE
    / "test_output"
    / "anchor_registration"
    / timestamp
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
# FIND REFERENCE IMAGE
# ============================================================

def find_reference_image() -> Path:
    """
    Prefer a copy placed directly in C:\\TDSBot.

    Otherwise use the newest reference produced by
    recording_anchor_setup.py.
    """

    direct = (
        BASE
        / "03_FINAL_RECORDING_ANCHOR.png"
    )

    if direct.exists():
        return direct

    search_root = (
        BASE
        / "test_output"
        / "recording_anchor_absolute"
    )

    matches = list(
        search_root.glob(
            "*/03_FINAL_RECORDING_ANCHOR.png"
        )
    )

    if not matches:
        raise FileNotFoundError(
            "Could not find "
            "03_FINAL_RECORDING_ANCHOR.png.\n\n"
            "Either copy it to:\n"
            "C:\\TDSBot\\03_FINAL_RECORDING_ANCHOR.png\n\n"
            "or leave the original under:\n"
            "C:\\TDSBot\\test_output\\"
            "recording_anchor_absolute\\..."
        )

    return max(
        matches,
        key=lambda p: p.stat().st_mtime,
    )


# ============================================================
# KEYBOARD
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
    seconds: float = 0.080,
) -> None:
    pydirectinput.keyDown(key)

    try:
        time.sleep(seconds)

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

        remaining = (
            S_HOLD_SECONDS
            - (
                time.perf_counter()
                - started
            )
        )

        if remaining > 0:
            time.sleep(remaining)

    finally:
        pydirectinput.keyUp(
            "space"
        )

        pydirectinput.keyUp(
            "s"
        )


# ============================================================
# RAW CAMERA INPUT
# ============================================================

def set_cursor(
    x: int,
    y: int,
) -> None:
    if not user32.SetCursorPos(
        int(x),
        int(y),
    ):
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


def right_down() -> None:
    user32.mouse_event(
        MOUSEEVENTF_RIGHTDOWN,
        0,
        0,
        0,
        0,
    )


def right_up() -> None:
    user32.mouse_event(
        MOUSEEVENTF_RIGHTUP,
        0,
        0,
        0,
        0,
    )


def camera_drag(
    dx: int,
    dy: int,
) -> None:
    set_cursor(
        1200,
        500,
    )

    time.sleep(0.20)

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

    right_down()

    try:
        time.sleep(0.20)

        relative_mouse_move(
            dx,
            dy,
        )

        time.sleep(0.20)

    finally:
        right_up()

    time.sleep(
        PITCH_BETWEEN_STEPS_SECONDS
    )


# ============================================================
# ABSOLUTE CAMERA NORMALIZATION
# ============================================================

def force_max_zoom_out() -> None:
    print()
    print(
        "Forcing maximum zoom-out..."
    )

    for i in range(
        MAX_ZOOM_OUT_PRESSES
    ):
        tap_key(
            "o",
            ZOOM_KEY_HOLD_SECONDS,
        )

        print(
            f"  O {i + 1}/"
            f"{MAX_ZOOM_OUT_PRESSES}"
        )

        time.sleep(
            ZOOM_BETWEEN_PRESSES_SECONDS
        )

    time.sleep(
        ZOOM_SETTLE_SECONDS
    )


def force_max_pitch() -> None:
    print()
    print(
        "Forcing maximum top-down pitch..."
    )

    for i in range(
        PITCH_SATURATION_STEPS
    ):
        camera_drag(
            0,
            PITCH_STEP_DY,
        )

        print(
            f"  pitch {i + 1}/"
            f"{PITCH_SATURATION_STEPS}"
        )

    time.sleep(
        PITCH_SETTLE_SECONDS
    )


# ============================================================
# IMAGE PROCESSING
# ============================================================

def make_edges(
    image: np.ndarray,
) -> np.ndarray:
    """
    Compare geometry rather than purple lighting.

    Canny edges make the registration much less sensitive to
    animated lighting / brightness changes.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    return cv2.Canny(
        gray,
        50,
        150,
    )


def register_anchor(
    reference: np.ndarray,
    current: np.ndarray,
) -> tuple[int, int, float]:
    if (
        reference.shape[:2]
        != current.shape[:2]
    ):
        raise RuntimeError(
            "Reference and current screenshots "
            "have different dimensions.\n"
            f"Reference: {reference.shape[1]}x"
            f"{reference.shape[0]}\n"
            f"Current:   {current.shape[1]}x"
            f"{current.shape[0]}"
        )

    height, width = (
        reference.shape[:2]
    )

    # Reference patch.
    template = reference[
        TEMPLATE_Y1:TEMPLATE_Y2,
        TEMPLATE_X1:TEMPLATE_X2,
    ]

    template_edges = make_edges(
        template
    )

    # Search slightly around that same position.
    sx1 = max(
        0,
        TEMPLATE_X1
        - SEARCH_MARGIN,
    )

    sy1 = max(
        0,
        TEMPLATE_Y1
        - SEARCH_MARGIN,
    )

    sx2 = min(
        width,
        TEMPLATE_X2
        + SEARCH_MARGIN,
    )

    sy2 = min(
        height,
        TEMPLATE_Y2
        + SEARCH_MARGIN,
    )

    search = current[
        sy1:sy2,
        sx1:sx2,
    ]

    search_edges = make_edges(
        search
    )

    result = cv2.matchTemplate(
        search_edges,
        template_edges,
        cv2.TM_CCOEFF_NORMED,
    )

    (
        _min_value,
        max_value,
        _min_location,
        max_location,
    ) = cv2.minMaxLoc(
        result
    )

    matched_x = (
        sx1
        + max_location[0]
    )

    matched_y = (
        sy1
        + max_location[1]
    )

    # This is how far the MAP moved on screen compared with
    # the exact frame used for recording.
    dx = (
        matched_x
        - TEMPLATE_X1
    )

    dy = (
        matched_y
        - TEMPLATE_Y1
    )

    return (
        int(dx),
        int(dy),
        float(max_value),
    )


# ============================================================
# DEBUG IMAGE
# ============================================================

def save_debug_image(
    current: np.ndarray,
    dx: int,
    dy: int,
    confidence: float,
) -> Path:
    debug = current.copy()

    # Reference location.
    cv2.rectangle(
        debug,
        (
            TEMPLATE_X1,
            TEMPLATE_Y1,
        ),
        (
            TEMPLATE_X2,
            TEMPLATE_Y2,
        ),
        (255, 255, 255),
        2,
    )

    # Detected current location.
    cv2.rectangle(
        debug,
        (
            TEMPLATE_X1 + dx,
            TEMPLATE_Y1 + dy,
        ),
        (
            TEMPLATE_X2 + dx,
            TEMPLATE_Y2 + dy,
        ),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        debug,
        (
            f"dx={dx} dy={dy} "
            f"confidence={confidence:.3f}"
        ),
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    path = (
        OUTPUT
        / "registration_debug.png"
    )

    cv2.imwrite(
        str(path),
        debug,
    )

    return path


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
            pydirectinput.keyUp(key)
        except Exception:
            pass

    try:
        right_up()
    except Exception:
        pass


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    reference_path = (
        find_reference_image()
    )

    print(
        "Anchor Registration Test"
    )

    print()
    print(
        "Reference:"
    )

    print(
        reference_path
    )

    reference = cv2.imread(
        str(reference_path),
        cv2.IMREAD_COLOR,
    )

    if reference is None:
        raise RuntimeError(
            "Could not read the reference image."
        )

    print()
    print(
        "Start from a fresh spawn "
        "with READY visible."
    )

    print()
    print(
        "This test WILL NOT click READY."
    )

    print(
        "It WILL NOT run the strategy."
    )

    print()
    print(
        f"Starting in "
        f"{STARTUP_DELAY_SECONDS:.0f} seconds..."
    )

    time.sleep(
        STARTUP_DELAY_SECONDS
    )

    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        with dxcam.create(
            backend="winrt",
            output_color="BGR",
        ) as camera:

            camera.start(
                target_fps=60
            )

            time.sleep(0.75)

            # ========================================
            # SAME ROUTE AS BOT
            # ========================================

            print()
            print(
                f"Holding A "
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
                "Holding S + jump..."
            )

            hold_s_and_jump()

            time.sleep(
                AFTER_S_PAUSE_SECONDS
            )

            print(
                f"Holding D "
                f"{D_HOLD_SECONDS:.3f}s..."
            )

            hold_key(
                "d",
                D_HOLD_SECONDS,
            )

            time.sleep(
                CAMERA_RECOVERY_SECONDS
            )

            # ========================================
            # SAME ABSOLUTE CAMERA
            # ========================================

            force_max_zoom_out()

            force_max_pitch()

            time.sleep(1.0)

            current = (
                camera.get_latest_frame()
            )

            if current is None:
                raise RuntimeError(
                    "DXcam returned no frame."
                )

            current_path = (
                OUTPUT
                / "current_anchor.png"
            )

            cv2.imwrite(
                str(current_path),
                current,
            )

            # ========================================
            # REGISTRATION
            # ========================================

            (
                dx,
                dy,
                confidence,
            ) = register_anchor(
                reference,
                current,
            )

            debug_path = save_debug_image(
                current,
                dx,
                dy,
                confidence,
            )

            print()
            print(
                "=" * 60
            )

            print(
                "ANCHOR REGISTRATION RESULT"
            )

            print(
                "=" * 60
            )

            print()
            print(
                f"Map shift dx: {dx:+d} px"
            )

            print(
                f"Map shift dy: {dy:+d} px"
            )

            print(
                f"Confidence:   "
                f"{confidence:.3f}"
            )

            print()

            if (
                abs(dx)
                > MAX_ALLOWED_SHIFT
                or abs(dy)
                > MAX_ALLOWED_SHIFT
            ):
                print(
                    "FAIL: displacement is "
                    "larger than expected."
                )

            elif (
                confidence
                < MIN_CONFIDENCE
            ):
                print(
                    "FAIL: image-match "
                    "confidence is too low."
                )

            else:
                print(
                    "Registration accepted."
                )

                print()
                print(
                    "The strategy should use:"
                )

                print(
                    f"    click_x += {dx:+d}"
                )

                print(
                    f"    click_y += {dy:+d}"
                )

                print()
                print(
                    "Example:"
                )

                print(
                    "recorded (600, 498)"
                )

                print(
                    f"corrected "
                    f"({600 + dx}, "
                    f"{498 + dy})"
                )

            print()
            print(
                "Current screenshot:"
            )

            print(
                current_path
            )

            print()
            print(
                "Registration debug:"
            )

            print(
                debug_path
            )

            camera.stop()

    finally:
        release_inputs()


if __name__ == "__main__":
    main()