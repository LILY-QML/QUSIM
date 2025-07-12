#!/usr/bin/env python3
"""
Simple Photon Trace Plotter
Reads raw photon data files and creates simple trace plots
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def read_photon_data(filepath: str) -> Dict[str, Any]:
    """Read photon count data from .dat file"""
    
    # Read metadata from header
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#'):
                if 'Bin width:' in line:
                    metadata['bin_width_ns'] = float(line.split('Bin width:')[1].split('ns')[0].strip())
                elif 'Total shots:' in line:
                    metadata['shots'] = int(line.split('Total shots:')[1].strip())
                elif 'Start time:' in line:
                    metadata['start_time_ns'] = float(line.split('Start time:')[1].split('ns')[0].strip())
                elif 'Measurement ID:' in line:
                    metadata['measurement_id'] = int(line.split('Measurement ID:')[1].strip())
                elif 'Experiment:' in line:
                    metadata['experiment'] = line.split('Experiment:')[1].strip()
            else:
                # Found data line
                data_line = line.strip()
                break
    
    # Parse data line (skip first column which is measurement_id)
    counts = np.array([float(x) for x in data_line.split()[1:]])
    
    # Create time array
    times_ns = metadata['start_time_ns'] + np.arange(len(counts)) * metadata['bin_width_ns']
    
    return {
        'times_ns': times_ns,
        'counts': counts,
        'metadata': metadata
    }


def plot_single_trace(data: Dict[str, Any], title: str = None, save_path: str = None):
    """Plot single photon trace"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Simple step plot
    ax.step(data['times_ns'], data['counts'], where='mid', linewidth=0.8, color='blue')
    
    # Labels and title
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Photon Counts')
    
    if title:
        ax.set_title(title)
    else:
        meta = data['metadata']
        ax.set_title(f"Photon Trace - Measurement {meta['measurement_id']+1}")
    
    ax.grid(True, alpha=0.3)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    else:
        plt.show()


def plot_comparison(data1: Dict[str, Any], data2: Dict[str, Any], save_path: str = None):
    """Plot two measurements in comparison"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    
    # Plot 1
    ax1.step(data1['times_ns'], data1['counts'], where='mid', linewidth=0.8, color='blue')
    meta1 = data1['metadata']
    ax1.set_title(f"Measurement {meta1['measurement_id']+1} - Electron Readout")
    ax1.set_ylabel('Photon Counts')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2
    ax2.step(data2['times_ns'], data2['counts'], where='mid', linewidth=0.8, color='red')
    meta2 = data2['metadata']
    ax2.set_title(f"Measurement {meta2['measurement_id']+1} - Nuclear Readout")
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Photon Counts')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison: {save_path}")
    else:
        plt.show()


def main():
    """Main function to process all .dat files in results/"""
    
    # Find all measurement data files
    data_files = glob.glob("results/*measurement*.dat")
    
    if not data_files:
        print("No measurement data files found in results/")
        return
    
    # Sort files to ensure consistent ordering
    data_files.sort()
    
    print(f"Found {len(data_files)} data files:")
    for f in data_files:
        print(f"  {f}")
    
    # Read all data
    all_data = []
    for filepath in data_files:
        data = read_photon_data(filepath)
        all_data.append(data)
        print(f"Loaded: {os.path.basename(filepath)} - {len(data['counts'])} points")
    
    # Create individual plots
    for i, data in enumerate(all_data):
        meta = data['metadata']
        filename = f"photon_trace_measurement_{meta['measurement_id']+1}.png"
        plot_single_trace(data, save_path=filename)
    
    # Create comparison plot if we have exactly 2 measurements
    if len(all_data) == 2:
        plot_comparison(all_data[0], all_data[1], save_path="photon_comparison.png")
        
        # Print some stats for comparison
        print("\nData comparison:")
        for i, data in enumerate(all_data):
            counts = data['counts']
            meta = data['metadata']
            print(f"Measurement {meta['measurement_id']+1}:")
            print(f"  Mean: {counts.mean():.2f}")
            print(f"  Std:  {counts.std():.2f}")
            print(f"  Min:  {counts.min():.0f}")
            print(f"  Max:  {counts.max():.0f}")
            print(f"  First 5: {counts[:5]}")
    
    print(f"\nAll plots saved to current directory")


if __name__ == "__main__":
    main()