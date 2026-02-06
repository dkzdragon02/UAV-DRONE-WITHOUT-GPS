#!/usr/bin/env python3
"""
Improved IMU Calibration Tool
Calibrate IMU offsets và noise parameters
"""

import numpy as np
import argparse
import json
import time
from typing import List, Tuple


def calibrate_imu_gyro(gyro_samples: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrate gyroscope
    
    Args:
        gyro_samples: List of gyro readings while stationary
    
    Returns:
        (offset, noise_std)
    """
    if not gyro_samples:
        return np.array([0.0, 0.0, 0.0]), np.array([0.01, 0.01, 0.01])
    
    samples = np.array(gyro_samples)
    
    # Offset is mean (should be ~0 when stationary)
    offset = np.mean(samples, axis=0)
    
    # Noise is standard deviation
    noise_std = np.std(samples, axis=0)
    
    return offset, noise_std


def calibrate_imu_accel(accel_samples: List[np.ndarray], gravity_magnitude: float = 9.81) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Calibrate accelerometer
    
    Args:
        accel_samples: List of accel readings in different orientations
        gravity_magnitude: Expected gravity magnitude
    
    Returns:
        (offset, scale, noise_std)
    """
    if not accel_samples:
        return np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0]), np.array([0.01, 0.01, 0.01])
    
    samples = np.array(accel_samples)
    
    # Compute magnitude of each sample
    magnitudes = np.linalg.norm(samples, axis=1)
    
    # Scale factor
    mean_magnitude = np.mean(magnitudes)
    scale = gravity_magnitude / mean_magnitude if mean_magnitude > 0 else 1.0
    
    # Offset (bias)
    # For accelerometer, offset is computed from samples when stationary
    # Assuming samples are in different orientations but stationary
    offset = np.mean(samples, axis=0) * scale
    
    # Noise
    scaled_samples = samples * scale
    noise_std = np.std(scaled_samples, axis=0)
    
    return offset, np.array([scale, scale, scale]), noise_std


def collect_samples(device, num_samples: int, sample_rate: float = 100.0) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Collect IMU samples
    
    Args:
        device: IMU device object
        num_samples: Number of samples to collect
        sample_rate: Sampling rate (Hz)
    
    Returns:
        (accel_samples, gyro_samples)
    """
    accel_samples = []
    gyro_samples = []
    
    print(f"Collecting {num_samples} samples at {sample_rate} Hz...")
    print("Keep IMU stationary and in different orientations")
    
    interval = 1.0 / sample_rate
    
    for i in range(num_samples):
        try:
            # Read IMU (adjust based on your IMU interface)
            # accel, gyro = device.read()
            # accel_samples.append(accel)
            # gyro_samples.append(gyro)
            
            # Placeholder
            accel_samples.append(np.array([0.0, 0.0, 9.81]))
            gyro_samples.append(np.array([0.0, 0.0, 0.0]))
            
            time.sleep(interval)
            
            if (i + 1) % 100 == 0:
                print(f"Collected {i+1}/{num_samples} samples")
        
        except Exception as e:
            print(f"Error reading IMU: {e}")
            break
    
    return accel_samples, gyro_samples


def main():
    parser = argparse.ArgumentParser(description='IMU Calibration Tool')
    parser.add_argument('--samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--rate', type=float, default=100.0, help='Sampling rate (Hz)')
    parser.add_argument('--out', default='imu_calibration.json', help='Output file')
    
    args = parser.parse_args()
    
    print("IMU Calibration Tool")
    print("=" * 40)
    print("Instructions:")
    print("1. Keep IMU stationary for gyro calibration")
    print("2. Rotate IMU to different orientations for accel calibration")
    print("3. Press Enter when ready to start...")
    input()
    
    # Collect samples (placeholder - implement actual IMU reading)
    accel_samples, gyro_samples = collect_samples(None, args.samples, args.rate)
    
    if not accel_samples or not gyro_samples:
        print("Failed to collect samples")
        return
    
    # Calibrate
    print("\nCalibrating...")
    gyro_offset, gyro_noise = calibrate_imu_gyro(gyro_samples)
    accel_offset, accel_scale, accel_noise = calibrate_imu_accel(accel_samples)
    
    # Save calibration
    calibration_data = {
        'gyroscope': {
            'offset': gyro_offset.tolist(),
            'noise_std': gyro_noise.tolist()
        },
        'accelerometer': {
            'offset': accel_offset.tolist(),
            'scale': accel_scale.tolist(),
            'noise_std': accel_noise.tolist()
        },
        'num_samples': len(accel_samples)
    }
    
    with open(args.out, 'w') as f:
        json.dump(calibration_data, f, indent=2)
    
    print(f"\nCalibration complete!")
    print(f"Gyro offset: {gyro_offset}")
    print(f"Accel offset: {accel_offset}")
    print(f"Accel scale: {accel_scale}")
    print(f"\nSaved to {args.out}")


if __name__ == '__main__':
    main()

