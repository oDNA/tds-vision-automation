from __future__ import annotations

import asyncio
import ctypes
from collections import deque
import statistics
from datetime import datetime
from pathlib import Path
import re
import time

import cv2
import dxcam
import numpy as np

from winrt.windows.globalization import Language
from winrt.windows.graphics.imaging import (
    BitmapPixelFormat,
    SoftwareBitmap,
)
from winrt.windows.media.ocr import OcrEngine
from winrt.windows.storage.streams import DataWriter


# ============================================================
# CONFIG
# ============================================================

CAPTURE_FPS = 60
OCR_INTERVAL_SECONDS = 0.25

# F8 = stop
STOP_VK = 0x77

# F7 = save a manual debug sample immediately
DEBUG_VK = 0x76

# The screenshots supplied were approximately 1920x1080.
# Use normalized ROIs so tiny window-size differences do not matter.

# Bottom HUD money display.
# Example: "$505", "$1,110"
MONEY_ROI_NORM = (
    1260 / 1920,
    920 / 1080,
    1350 / 1920,
    960 / 1080,
)

# Current wave only, excluding "/ 45".
# Example: "3", later "30".
WAVE_ROI_NORM = (
    540 / 1920,
    105 / 1080,
    590 / 1920,
    155 / 1080,
)

# Money changes rapidly, so requiring an exact repeated value is not
# appropriate. Instead use a short rolling median.
MONEY_HISTORY_SIZE = 5
MONEY_MIN_SAMPLES = 3

# A single OCR outlier can be wildly high (for example "$503" being
# read as "5503"). The median is resistant to that; the average is not.
#
# In the final strategy bot, this history will be CLEARED immediately
# after every successful placement/upgrade so an old pre-purchase cash
# value cannot leak into the next affordability decision.

# Wave numbers only increase, so use repeated reads plus a monotonic guard.
WAVE_STABLE_READS = 2

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = (
    BASE_DIR
    / "test_output"
    / "state_detector"
    / datetime.now().strftime("%Y%m%d_%H%M%S")
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
# ROI
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


# ============================================================
# OCR PREPROCESSING
# ============================================================

def preprocess_white_game_text(
    crop: np.ndarray,
    *,
    scale: int = 6,
) -> np.ndarray:
    """
    Convert Roblox's white text with black outline into large
    black characters on a clean white background.

    Windows OCR generally performs better when the tiny HUD
    text is enlarged first.
    """

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    # Keep bright, low/moderate-saturation text.
    white_mask = cv2.inRange(
        hsv,
        np.array(
            [0, 0, 155],
            dtype=np.uint8,
        ),
        np.array(
            [179, 115, 255],
            dtype=np.uint8,
        ),
    )

    # Remove isolated specks.
    (
        count,
        labels,
        stats,
        _centroids,
    ) = cv2.connectedComponentsWithStats(
        white_mask,
        connectivity=8,
    )

    clean = np.zeros_like(
        white_mask
    )

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

        if area >= 3:
            clean[
                labels == index
            ] = 255

    ys, xs = np.where(
        clean > 0
    )

    if len(xs) == 0:
        # OCR a blank image rather than crashing.
        return np.full(
            (80, 200),
            255,
            dtype=np.uint8,
        )

    x1 = max(
        0,
        int(xs.min()) - 3,
    )
    x2 = min(
        clean.shape[1],
        int(xs.max()) + 4,
    )

    y1 = max(
        0,
        int(ys.min()) - 3,
    )
    y2 = min(
        clean.shape[0],
        int(ys.max()) + 4,
    )

    clean = clean[
        y1:y2,
        x1:x2,
    ]

    # Windows OCR gets conventional black-on-white text.
    black_on_white = (
        255
        - clean
    )

    enlarged = cv2.resize(
        black_on_white,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    # Re-binarize after cubic resizing.
    _, enlarged = cv2.threshold(
        enlarged,
        160,
        255,
        cv2.THRESH_BINARY,
    )

    enlarged = cv2.copyMakeBorder(
        enlarged,
        30,
        30,
        30,
        30,
        cv2.BORDER_CONSTANT,
        value=255,
    )

    return enlarged


def preprocess_money_text(
    crop: np.ndarray,
    *,
    scale: int = 7,
) -> np.ndarray:
    """
    Money-specific preprocessing.

    Critical fix:
    Roblox renders the currency as "$503". Windows OCR can sometimes
    interpret the dollar sign itself as a leading "5", producing "5503".

    We identify the leftmost full-height text glyph (the "$") and remove
    it BEFORE OCR. The digits remain unchanged.

    After that, use the same white-text enlargement pipeline.
    """

    hsv = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2HSV,
    )

    white_mask = cv2.inRange(
        hsv,
        np.array(
            [0, 0, 155],
            dtype=np.uint8,
        ),
        np.array(
            [179, 115, 255],
            dtype=np.uint8,
        ),
    )

    (
        count,
        labels,
        stats,
        _centroids,
    ) = cv2.connectedComponentsWithStats(
        white_mask,
        connectivity=8,
    )

    # Full-height text glyphs. This excludes commas and tiny HUD specks.
    glyphs = []

    for index in range(
        1,
        count,
    ):
        (
            x,
            y,
            width,
            height,
            area,
        ) = [
            int(value)
            for value in stats[index]
        ]

        if (
            area >= 20
            and height >= 9
            and y >= 5
        ):
            glyphs.append(
                (
                    x,
                    y,
                    width,
                    height,
                    area,
                    index,
                )
            )

    glyphs.sort(
        key=lambda item: item[0]
    )

    # In the supplied TDS HUD:
    #   $ glyph height ~= 15
    #   digit height   ~= 11
    #
    # Only remove the first glyph when it looks taller than the next
    # digit. This avoids accidentally deleting a real leading digit.
    if len(glyphs) >= 2:
        (
            first_x,
            first_y,
            first_width,
            first_height,
            _first_area,
            first_index,
        ) = glyphs[0]

        second_height = glyphs[1][3]

        if (
            first_height >= 13
            and first_height
            >= second_height + 2
        ):
            white_mask[
                labels == first_index
            ] = 0

    # Clean small isolated pieces after removing "$".
    (
        count,
        labels,
        stats,
        _centroids,
    ) = cv2.connectedComponentsWithStats(
        white_mask,
        connectivity=8,
    )

    clean = np.zeros_like(
        white_mask
    )

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

        if area >= 3:
            clean[
                labels == index
            ] = 255

    ys, xs = np.where(
        clean > 0
    )

    if len(xs) == 0:
        return np.full(
            (80, 200),
            255,
            dtype=np.uint8,
        )

    x1 = max(
        0,
        int(xs.min()) - 3,
    )
    x2 = min(
        clean.shape[1],
        int(xs.max()) + 4,
    )

    y1 = max(
        0,
        int(ys.min()) - 3,
    )
    y2 = min(
        clean.shape[0],
        int(ys.max()) + 4,
    )

    clean = clean[
        y1:y2,
        x1:x2,
    ]

    black_on_white = (
        255
        - clean
    )

    enlarged = cv2.resize(
        black_on_white,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    _, enlarged = cv2.threshold(
        enlarged,
        160,
        255,
        cv2.THRESH_BINARY,
    )

    return cv2.copyMakeBorder(
        enlarged,
        30,
        30,
        30,
        30,
        cv2.BORDER_CONSTANT,
        value=255,
    )


# ============================================================
# WINDOWS OCR
# ============================================================

def make_software_bitmap(
    grayscale: np.ndarray,
) -> SoftwareBitmap:
    """
    Convert the OpenCV image to a WinRT SoftwareBitmap.

    Current PyWinRT exposes the four-argument overload:
        buffer, pixel_format, width, height
    """

    rgba = cv2.cvtColor(
        grayscale,
        cv2.COLOR_GRAY2RGBA,
    )

    writer = DataWriter()

    try:
        writer.write_bytes(
            rgba.tobytes()
        )

        buffer = (
            writer.detach_buffer()
        )

    finally:
        writer.close()

    return (
        SoftwareBitmap
        .create_copy_from_buffer(
            buffer,
            BitmapPixelFormat.RGBA8,
            rgba.shape[1],
            rgba.shape[0],
        )
    )


async def recognize_async(
    engine: OcrEngine,
    image: np.ndarray,
) -> str:
    bitmap = make_software_bitmap(
        image
    )

    result = await engine.recognize_async(
        bitmap
    )

    return (
        result.text
        or ""
    ).strip()


def recognize(
    engine: OcrEngine,
    image: np.ndarray,
) -> str:
    return asyncio.run(
        recognize_async(
            engine,
            image,
        )
    )


# ============================================================
# PARSERS
# ============================================================

def normalize_numeric_ocr(
    text: str,
) -> str:
    """
    Limited substitutions that are reasonable inside a HUD
    field expected to contain only numbers/currency.
    """

    result = (
        text
        .upper()
        .replace("O", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("|", "1")
    )

    return result


def parse_money(
    text: str,
) -> int | None:
    text = normalize_numeric_ocr(
        text
    )

    # Ignore spaces introduced between glyphs.
    text = re.sub(
        r"\s+",
        "",
        text,
    )

    # Supports both the bottom full-number form ($1,110)
    # and K/M abbreviations in case the UI ever changes.
    match = re.search(
        r"\$?([0-9][0-9,]*(?:\.[0-9]+)?)([KM]?)",
        text,
    )

    if not match:
        return None

    number_text = (
        match.group(1)
        .replace(",", "")
    )

    try:
        value = float(
            number_text
        )
    except ValueError:
        return None

    suffix = match.group(2)

    if suffix == "K":
        value *= 1000
    elif suffix == "M":
        value *= 1_000_000

    value = int(
        round(value)
    )

    if (
        value < 0
        or value > 100_000_000
    ):
        return None

    return value


def parse_wave(
    text: str,
) -> int | None:
    text = normalize_numeric_ocr(
        text
    )

    match = re.search(
        r"\d{1,2}",
        text,
    )

    if not match:
        return None

    value = int(
        match.group(0)
    )

    if 0 <= value <= 99:
        return value

    return None


# ============================================================
# DEBOUNCER
# ============================================================

class StableValue:
    def __init__(
        self,
        required_reads: int,
    ):
        self.required_reads = (
            required_reads
        )

        self.candidate = None
        self.streak = 0

        self.accepted = None

    def update(
        self,
        value,
    ):
        if value is None:
            self.candidate = None
            self.streak = 0
            return None

        if value == self.candidate:
            self.streak += 1
        else:
            self.candidate = value
            self.streak = 1

        if (
            self.streak
            >= self.required_reads
            and value
            != self.accepted
        ):
            self.accepted = value
            return value

        return None


class MoneyMedianFilter:
    """
    Short rolling median for noisy OCR.

    Example:
        [503, 504, 5503, 505, 506]
    -> median = 505

    A legitimate large reward is not discarded. Once most new samples
    contain the higher value, the median moves upward automatically.
    """

    def __init__(
        self,
        history_size: int = MONEY_HISTORY_SIZE,
        minimum_samples: int = MONEY_MIN_SAMPLES,
    ):
        self.values = deque(
            maxlen=history_size
        )

        self.minimum_samples = (
            minimum_samples
        )

        self.last_reported = None

    def clear(self) -> None:
        self.values.clear()
        self.last_reported = None

    def update(
        self,
        value: int | None,
    ):
        if value is None:
            return None

        self.values.append(
            int(value)
        )

        if (
            len(self.values)
            < self.minimum_samples
        ):
            return None

        filtered = int(
            statistics.median(
                self.values
            )
        )

        if filtered != self.last_reported:
            self.last_reported = filtered

            return (
                filtered,
                list(
                    self.values
                ),
            )

        return None


class MonotonicWaveFilter:
    """
    Wave can stay the same or increase, but never decrease during a match.
    Require repeated matching reads before accepting a new value.
    """

    def __init__(
        self,
        required_reads: int,
    ):
        self.required_reads = (
            required_reads
        )

        self.candidate = None
        self.streak = 0

        self.accepted = None

    def update(
        self,
        value: int | None,
    ):
        if value is None:
            self.candidate = None
            self.streak = 0
            return None

        # Ignore impossible backward movement once we have a wave.
        if (
            self.accepted is not None
            and value < self.accepted
        ):
            return None

        if value == self.candidate:
            self.streak += 1
        else:
            self.candidate = value
            self.streak = 1

        if (
            self.streak
            >= self.required_reads
            and value != self.accepted
        ):
            self.accepted = value
            return value

        return None


# ============================================================
# DEBUG
# ============================================================

def save_debug_sample(
    frame: np.ndarray,
    money_crop: np.ndarray,
    money_processed: np.ndarray,
    money_raw: str,
    money_value,
    wave_crop: np.ndarray,
    wave_processed: np.ndarray,
    wave_raw: str,
    wave_value,
    prefix: str,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stamp = datetime.now().strftime(
        "%H%M%S_%f"
    )

    base = (
        OUTPUT_DIR
        / f"{prefix}_{stamp}"
    )

    cv2.imwrite(
        str(base) + "_screen.png",
        frame,
    )

    cv2.imwrite(
        str(base) + "_money_raw.png",
        money_crop,
    )

    cv2.imwrite(
        str(base) + "_money_processed.png",
        money_processed,
    )

    cv2.imwrite(
        str(base) + "_wave_raw.png",
        wave_crop,
    )

    cv2.imwrite(
        str(base) + "_wave_processed.png",
        wave_processed,
    )

    text_path = Path(
        str(base)
        + "_ocr.txt"
    )

    text_path.write_text(
        "\n".join(
            [
                f"money_raw={money_raw!r}",
                f"money_value={money_value!r}",
                f"wave_raw={wave_raw!r}",
                f"wave_value={wave_value!r}",
            ]
        ),
        encoding="utf-8",
    )

    print(
        f"DEBUG SAMPLE SAVED: {base}"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "TDS Money + Wave Detector Test - robust money filter"
    )

    print()
    print(
        "F7 = save debug sample"
    )

    print(
        "F8 = stop"
    )

    print()
    print(
        "Expected output:"
    )

    print(
        "  MONEY: 505"
    )

    print(
        "  WAVE: 3"
    )

    print()
    print(
        "For the future strategy:"
    )

    print(
        "  placement uses a rolling-median cash check"
    )

    print(
        "  selling begins when WAVE >= 30"
    )

    print()

    language = Language(
        "en-US"
    )

    if not OcrEngine.is_language_supported(
        language
    ):
        raise RuntimeError(
            "Windows English OCR is not installed. "
            "Install the Windows English OCR language "
            "capability and run this test again."
        )

    engine = (
        OcrEngine
        .try_create_from_language(
            language
        )
    )

    if engine is None:
        raise RuntimeError(
            "Could not create the Windows OCR engine."
        )

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

    money_filter = MoneyMedianFilter()

    wave_filter = MonotonicWaveFilter(
        WAVE_STABLE_READS
    )

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

                break

            frame = (
                camera.get_latest_frame()
            )

            if frame is None:
                time.sleep(
                    0.02
                )
                continue

            money_crop = crop_normalized(
                frame,
                MONEY_ROI_NORM,
            )

            wave_crop = crop_normalized(
                frame,
                WAVE_ROI_NORM,
            )

            money_processed = (
                preprocess_money_text(
                    money_crop,
                    scale=7,
                )
            )

            wave_processed = (
                preprocess_white_game_text(
                    wave_crop,
                    scale=6,
                )
            )

            money_raw = recognize(
                engine,
                money_processed,
            )

            wave_raw = recognize(
                engine,
                wave_processed,
            )

            money_value = parse_money(
                money_raw
            )

            wave_value = parse_wave(
                wave_raw
            )

            filtered_money = (
                money_filter.update(
                    money_value
                )
            )

            accepted_wave = (
                wave_filter.update(
                    wave_value
                )
            )

            if filtered_money is not None:
                (
                    money_median,
                    money_samples,
                ) = filtered_money

                print(
                    f"MONEY: "
                    f"{money_median:,} "
                    f"  samples={money_samples}"
                )

            if accepted_wave is not None:
                print(
                    f"WAVE: "
                    f"{accepted_wave}"
                )

                if accepted_wave >= 30:
                    print(
                        "  SELL CONDITION: TRUE "
                        "(wave >= 30)"
                    )

            debug_down = key_down(
                DEBUG_VK
            )

            if (
                debug_down
                and not previous_debug_down
            ):
                print()
                print(
                    f"RAW MONEY OCR: "
                    f"{money_raw!r}"
                )

                print(
                    f"PARSED MONEY: "
                    f"{money_value!r}"
                )

                print(
                    f"RAW WAVE OCR: "
                    f"{wave_raw!r}"
                )

                print(
                    f"PARSED WAVE: "
                    f"{wave_value!r}"
                )

                save_debug_sample(
                    frame,
                    money_crop,
                    money_processed,
                    money_raw,
                    money_value,
                    wave_crop,
                    wave_processed,
                    wave_raw,
                    wave_value,
                    "manual",
                )

            previous_debug_down = (
                debug_down
            )

            time.sleep(
                OCR_INTERVAL_SECONDS
            )

    finally:
        try:
            camera.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()
