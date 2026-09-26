"""Chessboard camera calibration.

Detects chessboard corners across a set of captured images and estimates the
intrinsic matrix and distortion coefficients. Reads images from ../data/ and
writes camera_matrix.npy, dist_coeffs.npy, calibration_results.npz and
calibration_parameters.txt back into ../data/.
"""

import numpy as np
import cv2 as cv
import glob
import os
from pathlib import Path

from lab_config import (CHESSBOARD_SIZE, SQUARE_SIZE_MM, CALIB_IMAGE_PATTERN,
                        CAM_CALIB_RMS_GOOD_TARGET_PX, CAM_CALIB_RMS_RECAPTURE_PX)

_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR = _SCRIPT_DIR.parent / 'data'

def calibrate_camera(images_path='*.jpg', chessboard_size=(10, 7), square_size_mm=15.0):
    """Estimate intrinsics + distortion from chessboard images."""

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points (3D coordinates of chessboard corners in real world)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 
                           0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * square_size_mm  # Scale to real-world units (mm)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    good_images = []
    bad_images = []
    
    # Get list of images (resolved against the data directory)
    print("Searching ", str(_DATA_DIR / images_path))
    images = glob.glob(str(_DATA_DIR / images_path))
    print(f"Found {len(images)} images in {_DATA_DIR}")
    
    if len(images) == 0:
        print("ERROR: No images found! Check your path and file pattern.")
        return None
    
    # Process each image
    for idx, fname in enumerate(images):
        img = cv.imread(fname)
        if img is None:
            print(f"✗ Could not read: {fname}")
            bad_images.append(fname)
            continue
            
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboard_size, None)
        
        if ret:
            objpoints.append(objp)
            
            # Refine corners to sub-pixel accuracy
            corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            good_images.append(fname)
            
            # Draw and display corners for verification
            cv.drawChessboardCorners(img, chessboard_size, corners2, ret)
            cv.imshow('Calibration Progress', img)
            cv.waitKey(100)  # Show each for 0.1 seconds
            
            print(f"✓ [{len(good_images)}] Processed: {os.path.basename(fname)}")
        else:
            bad_images.append(fname)
            print(f"✗ No chessboard found: {os.path.basename(fname)}")
    
    cv.destroyAllWindows()
    
    # Summary
    print("\n" + "="*60)
    print("IMAGE PROCESSING SUMMARY")
    print("="*60)
    print(f"Total images: {len(images)}")
    print(f"Good images: {len(good_images)}")
    print(f"Bad images: {len(bad_images)}")
    
    if bad_images:
        print("\nBad images (remove or recapture these):")
        for img in bad_images:
            print(f"  - {os.path.basename(img)}")
    
    # Check if we have enough good images
    if len(good_images) < 5:
        print("\nERROR: Need at least 5 good images for calibration!")
        print(f"Only found {len(good_images)}. Please capture more images.")
        return None
    
    # Run calibration
    print("\n" + "="*60)
    print("RUNNING CALIBRATION...")
    print("="*60)
    
    # Get image size from first good image
    img = cv.imread(good_images[0])
    h, w = img.shape[:2]
    
    # Fit k1, k2, p1, p2 only; k3 pinned at zero (adds nothing, see check_k3_zero.py).
    dist_seed = np.zeros((5, 1), dtype=np.float64)
    rms_reproj_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, (w, h), None, dist_seed,
        flags=cv.CALIB_FIX_K3
    )

    # Display results
    print("\n" + "="*60)
    print("CALIBRATION RESULTS")
    print("="*60)
    print(f"Image size: {w} x {h} pixels")
    print(f"Reprojection error (RMS): {rms_reproj_error:.4f} pixels")
    print("Calibration quality: ", end="")

    # Worksheet: epsilon_rms "should be below 0.5 px; above 1.0 px, recapture."
    if rms_reproj_error <= CAM_CALIB_RMS_GOOD_TARGET_PX:
        print("EXCELLENT ✓")
    elif rms_reproj_error < CAM_CALIB_RMS_RECAPTURE_PX:
        print("ACCEPTABLE (target is below "
              f"{CAM_CALIB_RMS_GOOD_TARGET_PX:g} px)")
    else:
        print(f"POOR - consider recapturing with more varied views "
              f"(above {CAM_CALIB_RMS_RECAPTURE_PX:g} px)")
    
    print("\nCamera Matrix (intrinsic parameters):")
    print("┌" + "─"*50 + "┐")
    for row in camera_matrix:
        print(f"│ {row[0]:12.4f} {row[1]:12.4f} {row[2]:12.4f} │")
    print("└" + "─"*50 + "┘")
    
    print("\nDistortion Coefficients (k1, k2, p1, p2, k3):")
    print(dist_coeffs.flatten())
    
    print("\n" + "="*60)
    print("SAVING RESULTS")
    print("="*60)
    
    # Save as .npy files (easy to load in OpenCV)
    np.save(str(_DATA_DIR / 'camera_matrix.npy'), camera_matrix)
    np.save(str(_DATA_DIR / 'dist_coeffs.npy'), dist_coeffs)
    print("✓ Saved: camera_matrix.npy")
    print("✓ Saved: dist_coeffs.npy")

    # Save as .npz (contains all info)
    np.savez(str(_DATA_DIR / 'calibration_results.npz'),
             camera_matrix=camera_matrix,
             dist_coeffs=dist_coeffs,
             reprojection_error_rms=rms_reproj_error,
             image_size=(w, h),
             chessboard_size=chessboard_size,
             square_size_mm=square_size_mm)
    print("✓ Saved: calibration_results.npz")

    # Also save as text file for reference
    with open(str(_DATA_DIR / 'calibration_parameters.txt'), 'w') as f:
        f.write("CAMERA CALIBRATION PARAMETERS\n")
        f.write("="*40 + "\n\n")
        f.write(f"Image size: {w} x {h}\n")
        f.write(f"Reprojection error (RMS): {rms_reproj_error:.4f} pixels\n\n")
        f.write("Camera Matrix:\n")
        f.write(str(camera_matrix) + "\n\n")
        f.write("Distortion Coefficients:\n")
        f.write(str(dist_coeffs.flatten()) + "\n")
    print("✓ Saved: calibration_parameters.txt")

    # Write annotated report images (detected = green, reprojected = red)
    report_dir = _DATA_DIR / 'calibration_report'
    report_dir.mkdir(exist_ok=True)
    for i, fname in enumerate(good_images):
        img = cv.imread(fname)
        if img is None:
            continue
        # Green circles: originally detected corners
        cv.drawChessboardCorners(img, chessboard_size, imgpoints[i], True)
        # Red crosses: reprojected corners under fitted model
        reproj, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                      camera_matrix, dist_coeffs)
        reproj = reproj.reshape(-1, 2).astype(int)
        for pt in reproj:
            cv.circle(img, tuple(pt), 4, (0, 0, 255), -1)
        # Legend
        cv.putText(img, "Green: Detected   Red: Reprojected", (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        out_path = report_dir / f'report_{i:03d}.jpg'
        cv.imwrite(str(out_path), img)
    print(f"✓ Saved {len(good_images)} annotated images")

    return camera_matrix, dist_coeffs, rms_reproj_error


if __name__ == "__main__":
    # Chessboard geometry + image pattern come from lab_config.py — edit them
    # there rather than here so the whole pipeline stays consistent.
    IMAGE_PATTERN = CALIB_IMAGE_PATTERN

    print(f"Chessboard: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} inner corners, "
          f"{SQUARE_SIZE_MM} mm squares, pattern '{IMAGE_PATTERN}'")

    result = calibrate_camera(IMAGE_PATTERN, CHESSBOARD_SIZE, SQUARE_SIZE_MM)
    
    if result is not None:
        camera_matrix, dist_coeffs, error = result
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE!")
        print("="*60)
    else:
        print("\nCalibration failed. Please check your images and try again.")
