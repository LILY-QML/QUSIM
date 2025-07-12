#!/usr/bin/env python3
"""
Ultra-Simple Photon Plotter
Makes two separate plots from the two measurement files
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
import os


def read_measurement_data(filepath):
    """Read the counts from a measurement file"""
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Find the data line (last line without #)
    data_line = None
    for line in reversed(lines):
        if not line.startswith('#') and line.strip():
            data_line = line.strip()
            break
    
    if not data_line:
        raise ValueError(f"No data found in {filepath}")
    
    # Parse counts (skip first column which is measurement_id)
    parts = data_line.split()
    counts = np.array([float(x) for x in parts[1:]])
    
    return counts


def main():
    """Make two separate plots"""
    
    # Look for data files in script directory first, then core/results
    script_files = sorted(glob.glob("*.dat"))
    core_files = sorted(glob.glob("../core/results/*.dat"))
    
    files = script_files if script_files else core_files
    
    print(f"Found files: {files}")
    
    if len(files) != 2:
        print(f"Expected 2 files, found {len(files)}")
        return
    
    # Read both datasets
    counts1 = read_measurement_data(files[0])
    counts2 = read_measurement_data(files[1])
    
    print(f"Dataset 1: {len(counts1)} points, first 5: {counts1[:5]}")
    print(f"Dataset 2: {len(counts2)} points, first 5: {counts2[:5]}")
    
    # Create time arrays (3ns bins, starting at 23000 and 26000)
    times1 = 23000 + np.arange(len(counts1)) * 3
    times2 = 26000 + np.arange(len(counts2)) * 3
    
    # Plot 1 - Measurement 1 (Electron) in RED
    plt.figure(figsize=(12, 6))
    plt.step(times1, counts1, where='mid', color='red', linewidth=1, label='Electron Measurement')
    plt.title('Measurement 1 - Electron Readout (RED)')
    plt.xlabel('Time (ns)')
    plt.ylabel('Photon Counts')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('measurement_1_electron.png', dpi=150, bbox_inches='tight')
    plt.close()  # Important: close before next plot
    print("✅ Saved: measurement_1_electron.png")
    
    # Plot 2 - Measurement 2 (Nuclear) in BLUE  
    plt.figure(figsize=(12, 6))
    plt.step(times2, counts2, where='mid', color='blue', linewidth=1, label='Nuclear Measurement')
    plt.title('Measurement 2 - Nuclear Readout (BLUE)')
    plt.xlabel('Time (ns)')
    plt.ylabel('Photon Counts')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('measurement_2_nuclear.png', dpi=150, bbox_inches='tight')
    plt.close()  # Important: close before next plot
    print("✅ Saved: measurement_2_nuclear.png")
    
    # Verification comparison
    print(f"\nData verification:")
    print(f"Measurement 1 (red):  mean={counts1.mean():.2f}, first 3 values: {counts1[:3]}")
    print(f"Measurement 2 (blue): mean={counts2.mean():.2f}, first 3 values: {counts2[:3]}")
    print(f"Are they identical? {np.array_equal(counts1, counts2)}")


if __name__ == "__main__":
    main()