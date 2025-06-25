#!/usr/bin/env python3
"""
Minimal Demo: π-Pulse with Photon Emission

Ultra-simplified demonstration showing the core physics:
1. π-pulse flips |0⟩ → |±1⟩ 
2. Measure photon emission
3. Show fluorescence contrast
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def simulate_nv_experiment():
    """Minimal simulation of NV π-pulse experiment."""
    
    print("🔬 Minimal NV π-Pulse Experiment")
    print("=" * 40)
    
    # Simulate populations after π-pulse
    # Perfect π-pulse: |0⟩ → |+1⟩
    
    # Before π-pulse: |0⟩ (bright state)
    initial_population_0 = 1.0
    initial_population_plus1 = 0.0
    
    # After π-pulse: |+1⟩ (dark state) 
    final_population_0 = 0.0
    final_population_plus1 = 1.0
    
    print(f"Initial state: |0⟩ = {initial_population_0:.1f}, |+1⟩ = {initial_population_plus1:.1f}")
    print(f"After π-pulse: |0⟩ = {final_population_0:.1f}, |+1⟩ = {final_population_plus1:.1f}")
    
    # Fluorescence rates (photons/μs)
    bright_rate = 1000  # |0⟩ is bright
    dark_rate = 50      # |+1⟩ is dark
    readout_time = 1.0  # μs
    
    # Calculate expected photon counts
    bright_counts = bright_rate * readout_time
    dark_counts = dark_rate * readout_time
    
    print(f"\nPhoton emission rates:")
    print(f"  |0⟩ (bright): {bright_rate} photons/μs")
    print(f"  |+1⟩ (dark):  {dark_rate} photons/μs")
    
    # Simulate measurements with Poisson noise
    num_shots = 500
    
    # Before π-pulse (bright)
    measured_bright = np.random.poisson(bright_counts, num_shots)
    
    # After π-pulse (dark)
    measured_dark = np.random.poisson(dark_counts, num_shots)
    
    # Calculate contrast
    mean_bright = np.mean(measured_bright)
    mean_dark = np.mean(measured_dark)
    contrast = (mean_bright - mean_dark) / (mean_bright + mean_dark)
    
    print(f"\nMeasured photon counts ({num_shots} shots):")
    print(f"  Before π-pulse (bright): {mean_bright:.1f} ± {np.std(measured_bright):.1f}")
    print(f"  After π-pulse (dark):    {mean_dark:.1f} ± {np.std(measured_dark):.1f}")
    print(f"  Contrast: {contrast:.3f}")
    
    # Simple fidelity calculation
    threshold = (mean_bright + mean_dark) / 2
    correct_bright = np.sum(measured_bright > threshold)
    correct_dark = np.sum(measured_dark < threshold)
    fidelity = (correct_bright + correct_dark) / (2 * num_shots)
    
    print(f"  Readout fidelity: {fidelity:.3f} ({fidelity*100:.1f}%)")
    
    # Plot results
    plt.figure(figsize=(12, 4))
    
    # Histogram of photon counts
    plt.subplot(1, 3, 1)
    bins = np.linspace(0, max(np.max(measured_bright), np.max(measured_dark)) * 1.1, 30)
    plt.hist(measured_bright, bins=bins, alpha=0.7, label='Before π-pulse (|0⟩)', color='orange')
    plt.hist(measured_dark, bins=bins, alpha=0.7, label='After π-pulse (|+1⟩)', color='blue')
    plt.axvline(threshold, color='red', linestyle='--', label=f'Threshold = {threshold:.0f}')
    plt.xlabel('Photon Counts')
    plt.ylabel('Frequency')
    plt.title('Photon Detection')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Population diagram
    plt.subplot(1, 3, 2)
    states = ['Before π-pulse', 'After π-pulse']
    pop_0 = [initial_population_0, final_population_0]
    pop_plus1 = [initial_population_plus1, final_population_plus1]
    
    x = np.arange(len(states))
    width = 0.35
    
    plt.bar(x - width/2, pop_0, width, label='|0⟩', color='orange', alpha=0.7)
    plt.bar(x + width/2, pop_plus1, width, label='|+1⟩', color='blue', alpha=0.7)
    plt.xlabel('Experiment Phase')
    plt.ylabel('Population')
    plt.title('State Populations')
    plt.xticks(x, states)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Summary
    plt.subplot(1, 3, 3)
    metrics = ['Contrast', 'Fidelity']
    values = [contrast, fidelity]
    colors = ['green', 'purple']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.ylabel('Value')
    plt.title('Experiment Metrics')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.suptitle('NV Center π-Pulse Experiment Results', y=1.02, fontsize=14)
    plt.show()
    
    print(f"\n✅ Experiment completed!")
    print(f"π-pulse successfully flipped |0⟩ → |+1⟩")
    print(f"Fluorescence contrast: {contrast:.3f}")
    print(f"Readout works: {'✅' if fidelity > 0.8 else '❌'}")
    
    return {
        'contrast': contrast,
        'fidelity': fidelity,
        'bright_counts': measured_bright,
        'dark_counts': measured_dark,
        'bright_mean': mean_bright,
        'dark_mean': mean_dark
    }


if __name__ == "__main__":
    results = simulate_nv_experiment()