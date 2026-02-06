#!/usr/bin/env python3
"""
Improved Camera Calibration Tool
Calibrate camera với checkerboard pattern
"""

import cv2
import numpy as np
import argparse
import glob
import yaml
import os
from pathlib import Path


def calibrate_camera(images_dir: str, rows: int, cols: int, square_size: float, output_file: str):
    """
    Calibrate camera từ checkerboard images
    
    Args:
        images_dir: Directory chứa calibration images
        rows: Số hàng của checkerboard (inner corners)
        cols: Số cột của checkerboard (inner corners)
        square_size: Kích thước ô vuông (meters)
        output_file: File output để lưu calibration
    """
    # Prepare object points
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Find images
    images = glob.glob(os.path.join(images_dir, '*.jpg')) + \
             glob.glob(os.path.join(images_dir, '*.png'))
    
    if not images:
        print(f"No images found in {images_dir}")
        return False
    
    print(f"Found {len(images)} images")
    
    # Process each image
    for i, fname in enumerate(images):
        img = cv2.imread(fname)
        if img is None:
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        
        if ret:
            objpoints.append(objp)
            
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            
            # Draw and display corners
            cv2.drawChessboardCorners(img, (cols, rows), corners2, ret)
            print(f"Image {i+1}/{len(images)}: Found corners")
        else:
            print(f"Image {i+1}/{len(images)}: Failed to find corners")
    
    if len(objpoints) < 10:
        print(f"Not enough calibration images (found {len(objpoints)}, need at least 10)")
        return False
    
    # Calibrate camera
    print("Calibrating camera...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )
    
    # Compute reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error /= len(objpoints)
    
    print(f"Calibration complete!")
    print(f"Reprojection error: {mean_error:.3f} pixels")
    print(f"Camera matrix:\n{camera_matrix}")
    print(f"Distortion coefficients: {dist_coeffs.flatten()}")
    
    # Save calibration
    calibration_data = {
        'camera_matrix': camera_matrix.tolist(),
        'distortion_coefficients': dist_coeffs.tolist(),
        'image_size': list(gray.shape[::-1]),
        'reprojection_error': float(mean_error),
        'num_images': len(objpoints)
    }
    
    with open(output_file, 'w') as f:
        yaml.dump(calibration_data, f, default_flow_style=False)
    
    print(f"Calibration saved to {output_file}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Camera Calibration Tool')
    parser.add_argument('--images', required=True, help='Directory containing calibration images')
    parser.add_argument('--rows', type=int, default=7, help='Number of inner corners (rows)')
    parser.add_argument('--cols', type=int, default=9, help='Number of inner corners (cols)')
    parser.add_argument('--square', type=float, default=0.025, help='Square size in meters')
    parser.add_argument('--out', default='camera_calibration.yaml', help='Output file')
    
    args = parser.parse_args()
    
    calibrate_camera(args.images, args.rows, args.cols, args.square, args.out)


if __name__ == '__main__':
    main()

