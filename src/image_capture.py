"""Capture chessboard images for camera calibration.

Opens the camera, lets the user press SPACE to grab frames; saves them as
calib_NNN.jpg into ../data/. Press ESC to finish.
"""

import cv2
from pathlib import Path

from lab_config import CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data' 


def _lock_exposure_focus(cap):
    """Freeze exposure and focus so intrinsics stay stable across the session.

    Auto-exposure is disabled by setting a fixed gain + exposure value; focus
    is locked by disabling continuous AF (manual focus keeps the lens where it
    was when you framed the board). Both are best-effort: some USB cameras
    ignore these properties, in which case we just warn and continue.
    """
    warnings = []

    # Exposure: disable auto, set a fixed exposure/gain.
    if not cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0):
        warnings.append("auto-exposure could not be disabled")
    exp = cap.get(cv2.CAP_PROP_EXPOSURE)
    if exp is None or float(exp) <= 0:
        cap.set(cv2.CAP_PROP_EXPOSURE, -4)  # mid-range fixed shutter
    cap.set(cv2.CAP_PROP_GAIN, 30)

    # Focus: switch off continuous AF so the lens doesn't hunt between frames.
    if not cap.set(cv2.CAP_PROP_AUTOFOCUS, 0):
        warnings.append("autofocus could not be disabled")

    return warnings


def main():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    lock_warnings = _lock_exposure_focus(cap)
    for w in lock_warnings:
        print(f"[warn] {w}; calibration accuracy may suffer.")

    img_count = 0
    print("Exposure & focus locked. Press SPACE to capture, ESC to finish")
    print("Move chessboard to different positions/orientations")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, f"Images captured: {img_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('Capture', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            filename = str(_DATA_DIR / f"calib_{img_count:03d}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
            img_count += 1
        elif key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {img_count} images")


if __name__ == "__main__":
    main()
