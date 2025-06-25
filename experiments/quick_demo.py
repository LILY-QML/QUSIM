#!/usr/bin/env python3
"""
Quick Demo: π-Pulse Readout Experiment

Simplified version for quick testing and demonstration.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add QUSIM modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvcore'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvcore', 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvcore', 'helper'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvcore', 'modules'))

from pi_pulse_readout import PiPulseReadoutExperiment


def quick_demo():
    """Run a quick demonstration of the π-pulse readout experiment."""
    
    print("⚡ Quick π-Pulse Readout Demo")
    print("=" * 40)
    
    # Create experiment (fast mode)
    experiment = PiPulseReadoutExperiment(
        B_field=np.array([0.0, 0.0, 0.005]),  # 5 mT
        enable_noise=False  # Disable noise for cleaner demo
    )
    
    # Run with fewer measurements for speed
    results = experiment.run_complete_experiment(
        num_measurements=100,  # Faster
        rabi_frequency=2 * np.pi * 20e6  # 20 MHz (faster pulse)
    )
    
    # Quick summary
    fidelity = results['readout_fidelity']['fidelity']
    contrast = results['readout_fidelity']['contrast']
    
    print(f"\n🎯 QUICK RESULTS:")
    print(f"   Readout Fidelity: {fidelity:.3f} ({fidelity*100:.1f}%)")
    print(f"   Contrast: {contrast:.3f}")
    print(f"   π-Pulse worked: {'✅' if fidelity > 0.8 else '❌'}")
    
    # Simple plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Photon counts
    bright_counts = results['bright_measurement']['counts']
    dark_counts = results['dark_measurement']['counts']
    
    ax1.hist(bright_counts, alpha=0.7, label='|0⟩ (bright)', color='orange', bins=15)
    ax1.hist(dark_counts, alpha=0.7, label='|±1⟩ (dark)', color='blue', bins=15)
    ax1.axvline(results['readout_fidelity']['threshold'], color='red', linestyle='--', label='Threshold')
    ax1.set_xlabel('Photon Counts')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Photon Detection Results')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Population evolution
    times = results['quantum_evolution']['times']
    rho_history = results['quantum_evolution']['density_matrices']
    populations = np.array([[rho[i,i].real for i in range(3)] for rho in rho_history])
    
    ax2.plot(times*1e9, populations[:, 0], 'b-', label='|−1⟩', linewidth=2)
    ax2.plot(times*1e9, populations[:, 1], 'r-', label='|0⟩', linewidth=2)
    ax2.plot(times*1e9, populations[:, 2], 'g-', label='|+1⟩', linewidth=2)
    ax2.set_xlabel('Time (ns)')
    ax2.set_ylabel('Population')
    ax2.set_title('π-Pulse Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.suptitle('Quick π-Pulse Demo Results', y=1.02)
    plt.show()
    
    return results


if __name__ == "__main__":
    quick_demo()