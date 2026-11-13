from __future__ import annotations

import ctypes
from datetime import datetime
from pathlib import Path
import time

import cv2
import dxcam
import numpy as np


# ============================================================
# CONFIG
# ============================================================

CAPTURE_FPS = 60
SAMPLE_INTERVAL = 0.08

# F6 = calibrate the CURRENT visible wave as Wave 0
CALIBRATE_VK = 0x75

# F7 = save debug sample
DEBUG_VK = 0x76

# F8 = stop
STOP_VK = 0x77

# Tight ROI around ONLY the current-wave numeral(s).
#
# From the supplied 1917x1077 screenshot:
#   current "3" ~= x572..594
#   slash begins ~= x605
#
# The wider left edge allows two-digit waves such as 10..45.
WAVE_NUMBER_ROI_NORM = (
    525 / 1917,
    100 / 1077,
    602 / 1917,
    160 / 1077,
)

# White text mask. The black outline is deliberately ignored.
WHITE_HSV_LOW = np.array(
    [0, 0, 155],
    dtype=np.uint8,
)

WHITE_HSV_HIGH = np.array(
    [179, 115, 255],
    dtype=np.uint8,
)

# Remove tiny unrelated bright pixels.
MIN_GLYPH_COMPONENT_AREA = 45
MIN_TOTAL_WHITE_AREA = 100
MAX_TOTAL_WHITE_AREA = 1500

# The accepted wave image must differ substantially before a transition
# is even considered.
CHANGE_DIFF_THRESHOLD = 0.14

# Candidate frames must look nearly identical to each other before the
# new wave is accepted. This waits out any transition animation/shake.
CANDIDATE_STABILITY_THRESHOLD = 0.045

# Consecutive stable frames required for calibration/new-wave acceptance.
CALIBRATION_STABLE_FRAMES = 5
NEW_WAVE_STABLE_FRAMES = 5

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    BASE_DIR
    / "test_output"
    / "visual_wave_tracker"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
)

LOG_PATH = (
    OUTPUT_DIR
    / "wave_log.txt"
)


# ============================================================
# WINDOWS
# ============================================================

user32 = ctypes.WinDLL(
    "user32",
    use_last_error=True,
)


def key_down(vk: int) -> bool:
    return bool(
        user32.GetAsyncKeyState(vk)
        & 0x8000
    )


# ============================================================
# IMAGE HELPERS
# ============================================================

def crop_normalized(
    frame: np.ndarray,
    roi,
) -> np.ndarray:
    height, width = frame.shape[:2]

    x1 = int(round(roi[0] * width))
    y1 = int(round(roi[1] * height))
    x2 = int(round(roi[2] * width))
    y2 = int(round(roi[3] * height))

    return frame[
        y1:y2,
        x1:x2,
    ].copy()


def extract_wave_mask(
    frame: np.ndarray,
):
    """
    Extract only the white current-wave numeral(s).

    Returns:
        crop,
        cleaned binary mask,
        valid
    """

    crop = crop_normalized(
        frame,
        WAVE_NUMBER_ROI_NORM,
    )

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    raw_mask = cv2.inRange(
        hsv,
        WHITE_HSV_LOW,
        WHITE_HSV_HIGH,
    )

    (
        count,
        labels,
        stats,
        _centroids,
    ) = cv2.connectedComponentsWithStats(
        raw_mask,
        connectivity=8,
    )

    clean = np.zeros_like(
        raw_mask
    )

    glyph_count = 0
    total_area = 0

    for index in range(
        1,
        count,
    ):
        area = int(
            stats[
                index,
                cv2.CC_STAT_AREA,
            ]
        )

        height = int(
            stats[
                index,
                cv2.CC_STAT_HEIGHT,
            ]
        )

        # Wave digits are tall. This also discards small sparkles/noise.
        if (
            area >= MIN_GLYPH_COMPONENT_AREA
            and height >= 18
        ):
            clean[
                labels == index
            ] = 255

            glyph_count += 1
            total_area += area

    # Current wave 0-9 -> one main glyph.
    # Current wave 10-45 -> two main glyphs.
    valid = (
        glyph_count in (1, 2)
        and MIN_TOTAL_WHITE_AREA
        <= total_area
        <= MAX_TOTAL_WHITE_AREA
    )

    return (
        crop,
        clean,
        valid,
    )


def mask_difference(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Symmetric binary difference normalized by the union.

    0.0 = same visual numeral
    1.0 = completely different
    """

    if a.shape != b.shape:
        return 1.0

    a_on = a > 0
    b_on = b > 0

    union = np.logical_or(
        a_on,
        b_on,
    )

    union_count = int(
        np.count_nonzero(
            union
        )
    )

    if union_count == 0:
        return 1.0

    xor = np.logical_xor(
        a_on,
        b_on,
    )

    return float(
        np.count_nonzero(
            xor
        )
        / union_count
    )


# ============================================================
# LOG / DEBUG
# ============================================================

def append_log(
    message: str,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%H:%M:%S.%f"
    )[:-3]

    with LOG_PATH.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            f"{timestamp} | {message}\n"
        )


def save_state_images(
    *,
    frame: np.ndarray,
    crop: np.ndarray,
    mask: np.ndarray,
    label: str,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now().strftime(
        "%H%M%S_%f"
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{label}_{stamp}_screen.png"
        ),
        frame,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{label}_{stamp}_crop.png"
        ),
        crop,
    )

    cv2.imwrite(
        str(
            OUTPUT_DIR
            / f"{label}_{stamp}_mask.png"
        ),
        mask,
    )


# ============================================================
# TRACKER
# ============================================================

class VisualWaveTracker:
    """
    The tracker does NOT OCR the number.

    At calibration we know the visible state is Wave 0.

    Afterwards, TDS guarantees sequential progression:
        0 -> 1 -> 2 -> ... -> 45

    Therefore the only question is:
        "Has the visual wave numeral changed to a new stable shape?"

    Each accepted stable visual change increments the internal wave by 1.
    """

    def __init__(self):
        self.wave = None

        self.accepted_mask = None

        self.calibration_candidate = None
        self.calibration_streak = 0

        self.transition_candidate = None
        self.transition_streak = 0

    @property
    def calibrated(self) -> bool:
        return (
            self.wave is not None
            and self.accepted_mask is not None
        )

    def reset_calibration_candidate(
        self,
    ) -> None:
        self.calibration_candidate = None
        self.calibration_streak = 0

    def reset_transition_candidate(
        self,
    ) -> None:
        self.transition_candidate = None
        self.transition_streak = 0

    def feed_calibration_frame(
        self,
        mask: np.ndarray,
    ) -> bool:
        """
        Build a stable Wave-0 baseline after F6.
        """

        if self.calibration_candidate is None:
            self.calibration_candidate = (
                mask.copy()
            )
            self.calibration_streak = 1

            return False

        diff = mask_difference(
            mask,
            self.calibration_candidate,
        )

        if (
            diff
            <= CANDIDATE_STABILITY_THRESHOLD
        ):
            self.calibration_streak += 1
        else:
            self.calibration_candidate = (
                mask.copy()
            )
            self.calibration_streak = 1

        if (
            self.calibration_streak
            >= CALIBRATION_STABLE_FRAMES
        ):
            self.wave = 0
            self.accepted_mask = (
                mask.copy()
            )

            self.reset_calibration_candidate()
            self.reset_transition_candidate()

            return True

        return False

    def update(
        self,
        mask: np.ndarray,
    ):
        """
        Return a newly accepted wave number, else None.
        """

        if not self.calibrated:
            return None

        difference_from_current = (
            mask_difference(
                mask,
                self.accepted_mask,
            )
        )

        # Still the same wave.
        if (
            difference_from_current
            < CHANGE_DIFF_THRESHOLD
        ):
            self.reset_transition_candidate()
            return None

        # Visually different. Require the NEW appearance itself to settle.
        if self.transition_candidate is None:
            self.transition_candidate = (
                mask.copy()
            )
            self.transition_streak = 1

            return None

        candidate_diff = mask_difference(
            mask,
            self.transition_candidate,
        )

        if (
            candidate_diff
            <= CANDIDATE_STABILITY_THRESHOLD
        ):
            self.transition_streak += 1
        else:
            self.transition_candidate = (
                mask.copy()
            )
            self.transition_streak = 1

            return None

        if (
            self.transition_streak
            < NEW_WAVE_STABLE_FRAMES
        ):
            return None

        # Final safety check: the stable candidate must still truly differ
        # from the currently accepted wave.
        final_diff = mask_difference(
            self.transition_candidate,
            self.accepted_mask,
        )

        if (
            final_diff
            < CHANGE_DIFF_THRESHOLD
        ):
            self.reset_transition_candidate()
            return None

        if self.wave >= 45:
            self.reset_transition_candidate()
            return None

        self.wave += 1

        self.accepted_mask = (
            self.transition_candidate.copy()
        )

        self.reset_transition_candidate()

        return self.wave


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "TDS Visual Wave Tracker - no OCR"
    )

    print()
    print(
        "IMPORTANT:"
    )
    print(
        "  1. Start this test."
    )
    print(
        "  2. Switch to Roblox."
    )
    print(
        "  3. While the display genuinely says Wave 0, press F6."
    )
    print(
        "  4. Leave it running."
    )

    print()
    print(
        "F6 = calibrate visible state as Wave 0"
    )
    print(
        "F7 = save debug image"
    )
    print(
        "F8 = stop"
    )

    print()
    print(
        "Sell condition for future bot:"
    )
    print(
        "  tracked wave >= 30"
    )

    print()
    print(
        f"Automatic log:"
    )
    print(
        f"  {LOG_PATH}"
    )
    print()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    camera = dxcam.create(
        backend="winrt",
        output_color="BGR",
    )

    camera.start(
        target_fps=CAPTURE_FPS
    )

    tracker = VisualWaveTracker()

    calibrating = False

    previous_calibrate_down = False
    previous_debug_down = False

    try:
        time.sleep(
            0.75
        )

        while True:
            if key_down(
                STOP_VK
            ):
                print()
                print(
                    "F8 detected. Stopping."
                )

                append_log(
                    "STOP"
                )

                break

            frame = (
                camera.get_latest_frame()
            )

            if frame is None:
                time.sleep(
                    0.02
                )
                continue

            (
                crop,
                mask,
                valid,
            ) = extract_wave_mask(
                frame
            )

            calibrate_down = key_down(
                CALIBRATE_VK
            )

            if (
                calibrate_down
                and not previous_calibrate_down
            ):
                calibrating = True

                tracker = (
                    VisualWaveTracker()
                )

                print()
                print(
                    "F6 detected: calibrating current "
                    "visual state as Wave 0..."
                )

                append_log(
                    "F6 pressed; beginning Wave-0 calibration"
                )

            previous_calibrate_down = (
                calibrate_down
            )

            if valid:
                if (
                    calibrating
                    and not tracker.calibrated
                ):
                    if tracker.feed_calibration_frame(
                        mask
                    ):
                        calibrating = False

                        print(
                            "WAVE: 0 "
                            "(visual baseline calibrated)"
                        )

                        append_log(
                            "ACCEPTED WAVE 0"
                        )

                        save_state_images(
                            frame=frame,
                            crop=crop,
                            mask=mask,
                            label="wave_00",
                        )

                elif tracker.calibrated:
                    new_wave = (
                        tracker.update(
                            mask
                        )
                    )

                    if new_wave is not None:
                        print(
                            f"WAVE: {new_wave}"
                        )

                        append_log(
                            f"ACCEPTED WAVE {new_wave}"
                        )

                        save_state_images(
                            frame=frame,
                            crop=crop,
                            mask=mask,
                            label=(
                                f"wave_{new_wave:02d}"
                            ),
                        )

                        if (
                            new_wave >= 30
                        ):
                            print(
                                "  SELL CONDITION: TRUE "
                                "(wave >= 30)"
                            )

                            append_log(
                                "SELL CONDITION TRUE"
                            )

            debug_down = key_down(
                DEBUG_VK
            )

            if (
                debug_down
                and not previous_debug_down
            ):
                print(
                    "F7: debug image saved."
                )

                append_log(
                    "manual debug sample"
                )

                save_state_images(
                    frame=frame,
                    crop=crop,
                    mask=mask,
                    label="manual",
                )

            previous_debug_down = (
                debug_down
            )

            time.sleep(
                SAMPLE_INTERVAL
            )

    finally:
        try:
            camera.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
