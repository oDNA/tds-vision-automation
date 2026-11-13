from pathlib import Path
import time

import cv2
import dxcam


OUTPUT_FILE = Path(__file__).with_name("roblox_capture.png")


def main():
    print("Available displays:")
    print(dxcam.output_info())

    print()
    print("Using DXcam Windows Graphics Capture (WinRT) backend.")
    print("Capturing primary display in 5 seconds...")
    print("Switch to Roblox and make sure the READY button is visible.")

    time.sleep(5)

    try:
        with dxcam.create(
            backend="winrt",
            output_color="BGR",
        ) as camera:

            frame = camera.grab(new_frame_only=False)

        if frame is None:
            print("ERROR: DXcam returned no frame.")
            return

        height, width = frame.shape[:2]

        print()
        print(f"Captured resolution: {width}x{height}")
        print(f"Frame shape: {frame.shape}")

        success = cv2.imwrite(str(OUTPUT_FILE), frame)

        if not success:
            print("ERROR: Could not save screenshot.")
            return

        print()
        print("SUCCESS")
        print(f"Saved screenshot to: {OUTPUT_FILE}")

    except Exception as exc:
        print()
        print("CAPTURE FAILED")
        print(f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()