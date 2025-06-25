#!/usr/bin/env python3
"""
Fast Time-Resolved Photon Trace

Simplified but realistic photon counting after MW π-pulse.
Shows exactly what you asked for:
- MW pulse at resonance frequency
- Time-resolved photon counting 0-600 ns  
- Count rate every nanosecond
"""

import numpy as np
import matplotlib.pyplot as plt
import time

def fast_photon_trace_experiment():
    """
    Fast simulation of time-resolved photon counting.
    
    Simulates:
    1. MW π-pulse |0⟩ → |+1⟩
    2. Laser readout starting immediately  
    3. Photon counting every 1 ns for 600 ns
    4. Realistic photon emission rates
    """
    
    print("📡 Fast Time-Resolved Photon Counting")
    print("=" * 45)
    
    # Experimental parameters
    B_field = 10e-3  # 10 mT magnetic field
    gamma_e = 28e9   # Gyromagnetic ratio (Hz/T)
    mw_frequency = gamma_e * B_field  # Resonance frequency
    
    print(f"⚛️  NV Center Parameters:")
    print(f"   Magnetic field: {B_field*1000:.0f} mT")
    print(f"   MW frequency: {mw_frequency/1e9:.3f} GHz")
    
    # π-pulse simulation (simplified)
    rabi_freq = 15e6  # 15 MHz
    pi_pulse_duration = np.pi / (2 * np.pi * rabi_freq)
    
    print(f"\n🌊 π-Pulse:")
    print(f"   Frequency: {mw_frequency/1e9:.3f} GHz (resonant)")
    print(f"   Duration: {pi_pulse_duration*1e9:.1f} ns")
    print(f"   Rabi frequency: {rabi_freq/1e6:.0f} MHz")
    
    # State after π-pulse
    initial_pop_0 = 1.0      # Start in |0⟩
    final_pop_0 = 0.2        # Some residual after π-pulse (more realistic)
    final_pop_plus1 = 0.8    # Most transferred to |+1⟩
    
    print(f"\n⚡ Population Transfer:")
    print(f"   Before: |0⟩ = {initial_pop_0:.2f}, |+1⟩ = 0.00")
    print(f"   After:  |0⟩ = {final_pop_0:.2f}, |+1⟩ = {final_pop_plus1:.2f}")
    print(f"   Transfer efficiency: {(1-final_pop_0)*100:.0f}%")
    
    # Photon emission rates
    bright_rate = 5e6      # |0⟩ emission rate (photons/s) - higher for better detection
    dark_rate = 0.2e6      # |+1⟩ emission rate (photons/s)
    background = 5e3       # Background rate
    
    # Detection efficiency
    collection_eff = 0.1    # 10% collection - better optics
    detector_eff = 0.9      # 90% quantum efficiency - good detector
    total_eff = collection_eff * detector_eff
    
    print(f"\n📡 Detection Setup:")
    print(f"   Bright rate (|0⟩): {bright_rate/1e6:.1f} Mcps")
    print(f"   Dark rate (|+1⟩):  {dark_rate/1e6:.2f} Mcps")
    print(f"   Collection eff: {collection_eff*100:.0f}%")
    print(f"   Detector eff: {detector_eff*100:.0f}%")
    print(f"   Total efficiency: {total_eff*100:.1f}%")
    
    # Time-resolved measurement
    readout_duration = 600e-9  # 600 ns
    time_bin = 1e-9           # 1 ns resolution
    n_bins = int(readout_duration / time_bin)
    
    print(f"\n📸 Time-Resolved Readout:")
    print(f"   Duration: {readout_duration*1e9:.0f} ns")
    print(f"   Time bins: {n_bins} bins of {time_bin*1e9:.0f} ns each")
    
    # Calculate effective count rate after π-pulse
    effective_rate = (
        final_pop_0 * bright_rate +
        final_pop_plus1 * dark_rate +
        background
    ) * total_eff
    
    print(f"   Expected count rate: {effective_rate/1e6:.3f} Mcps")
    print(f"   Counts per ns: {effective_rate * time_bin:.4f}")
    
    # Generate time-resolved photon trace
    print(f"\n🔢 Generating photon trace...")
    
    time_ns = np.arange(0, readout_duration*1e9, time_bin*1e9)  # Time in ns
    photon_counts = np.zeros(len(time_ns))
    
    # Add realistic time-dependent effects
    for i, t in enumerate(time_ns):
        
        # Base count rate
        base_rate = effective_rate
        
        # Add realistic fluctuations:
        
        # 1. Laser intensity noise (2% RIN)
        laser_noise = 1.0 + 0.02 * np.random.randn()
        
        # 2. Detector response variations (5%)
        detector_noise = 1.0 + 0.05 * np.random.randn()
        
        # 3. Occasional charge state jumps (blinking)
        if np.random.random() < 0.0005:  # 0.05% chance per ns
            charge_jump = 0.1  # Temporary dark state
        else:
            charge_jump = 1.0
            
        # 4. Background fluctuations
        background_noise = 1.0 + 0.1 * np.random.randn()
        
        # Combined rate for this time bin
        instantaneous_rate = base_rate * laser_noise * detector_noise * charge_jump * background_noise
        instantaneous_rate = max(0, instantaneous_rate)  # No negative rates
        
        # Expected counts in this 1 ns bin
        expected_counts = instantaneous_rate * time_bin
        
        # Poisson statistics (shot noise)
        if expected_counts > 0:
            photon_counts[i] = np.random.poisson(expected_counts)
        else:
            photon_counts[i] = 0
    
    # Calculate statistics
    total_photons = np.sum(photon_counts)
    average_rate = total_photons / readout_duration
    peak_counts = np.max(photon_counts)
    peak_rate = peak_counts / time_bin
    
    print(f"\n📊 Results:")
    print(f"   Total photons: {total_photons:.0f}")
    print(f"   Average rate: {average_rate/1e6:.3f} Mcps")
    print(f"   Peak counts/ns: {peak_counts:.0f}")
    print(f"   Peak rate: {peak_rate/1e6:.3f} Mcps")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Full time trace
    ax = axes[0, 0]
    ax.plot(time_ns, photon_counts, 'b-', linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Photon Counts per ns')
    ax.set_title(f'Time-Resolved Photon Emission (0-{readout_duration*1e9:.0f} ns)')
    ax.grid(True, alpha=0.3)
    
    # Add average line
    avg_counts_per_ns = average_rate * time_bin
    ax.axhline(avg_counts_per_ns, color='red', linestyle='--', 
              label=f'Average: {avg_counts_per_ns:.3f} counts/ns')
    ax.legend()
    
    # 2. First 100 ns (zoomed)
    ax = axes[0, 1]
    zoom_mask = time_ns <= 100
    ax.plot(time_ns[zoom_mask], photon_counts[zoom_mask], 'ro-', 
           markersize=3, linewidth=1, alpha=0.8)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Photon Counts per ns')
    ax.set_title('First 100 ns (High Resolution)')
    ax.grid(True, alpha=0.3)
    
    # 3. Cumulative counts
    ax = axes[1, 0]
    cumulative = np.cumsum(photon_counts)
    ax.plot(time_ns, cumulative, 'g-', linewidth=2)
    ax.set_xlabel('Time (ns)')
    ax.set_ylabel('Cumulative Photon Counts')
    ax.set_title('Cumulative Photon Detection')
    ax.grid(True, alpha=0.3)
    
    # Add linear fit
    fit_slope = average_rate * 1e-9  # counts per ns
    fit_line = fit_slope * time_ns
    ax.plot(time_ns, fit_line, 'r--', linewidth=2, 
           label=f'Linear fit: {average_rate/1e6:.3f} Mcps')
    ax.legend()
    
    # 4. Count statistics
    ax = axes[1, 1]
    ax.hist(photon_counts, bins=20, alpha=0.7, color='purple', edgecolor='black')
    ax.set_xlabel('Counts per ns')
    ax.set_ylabel('Frequency')
    ax.set_title('Count Distribution')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_counts = np.mean(photon_counts)
    std_counts = np.std(photon_counts)
    ax.axvline(mean_counts, color='red', linestyle='--', 
              label=f'Mean: {mean_counts:.3f}')
    ax.axvline(mean_counts + std_counts, color='orange', linestyle=':', 
              label=f'±σ: {std_counts:.3f}')
    ax.axvline(mean_counts - std_counts, color='orange', linestyle=':')
    ax.legend()
    
    plt.tight_layout()
    plt.suptitle('MW π-Pulse → Time-Resolved Photon Counting', fontsize=16, y=0.98)
    
    # Add experimental info
    info_text = f"""
Experiment Parameters:
• MW frequency: {mw_frequency/1e9:.3f} GHz
• π-pulse duration: {pi_pulse_duration*1e9:.1f} ns  
• Population transfer: {(1-final_pop_0)*100:.0f}%
• Count rate: {average_rate/1e6:.3f} Mcps
• Total photons: {total_photons:.0f}
    """
    
    fig.text(0.02, 0.02, info_text, fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.show()
    
    print(f"\n✅ Time-resolved photon counting completed!")
    print(f"   Successfully measured photon emission after π-pulse")
    print(f"   Time resolution: 1 ns over 600 ns duration")
    
    return {
        'time_ns': time_ns,
        'photon_counts': photon_counts,
        'total_photons': total_photons,
        'average_rate': average_rate,
        'peak_rate': peak_rate,
        'mw_frequency': mw_frequency,
        'pi_pulse_duration': pi_pulse_duration,
        'population_transfer': 1 - final_pop_0
    }


if __name__ == "__main__":
    results = fast_photon_trace_experiment()