#!/usr/bin/env python3
"""
Test and fix photon rate calculations for realistic NV emission
"""

import numpy as np

def calculate_realistic_nv_rates():
    """Calculate realistic NV photon emission rates"""
    
    print("🔬 NV Center Photon Rate Analysis")
    print("=" * 50)
    
    # Load NV parameters from system.json
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'nvcore', 'helper'))
    from noise_sources import SYSTEM
    
    intrinsic_bright = SYSTEM.get_noise_param('optical', 'readout', 'bright_state_rate')
    intrinsic_dark = SYSTEM.get_noise_param('optical', 'readout', 'dark_state_rate')
    background = SYSTEM.get_empirical_param('optical_system', 'detector_dark_counts')
    
    # Detection system parameters from empirical measurements
    collection_eff = SYSTEM.get_empirical_param('optical_system', 'collection_efficiency')
    detector_qe = SYSTEM.get_noise_param('optical', 'readout', 'detector_efficiency')
    total_eff = collection_eff * detector_qe
    
    # Effective detected rates
    effective_bright = intrinsic_bright * total_eff
    effective_dark = intrinsic_dark * total_eff
    effective_bg = background * total_eff
    
    print(f"Intrinsic rates:")
    print(f"  Bright state: {intrinsic_bright/1e6:.0f} Mcps")
    print(f"  Dark state: {intrinsic_dark/1e6:.0f} Mcps")
    print(f"  Background: {background/1e3:.0f} kcps")
    
    print(f"\nDetection efficiency:")
    print(f"  Collection: {collection_eff*100:.0f}%")
    print(f"  Detector QE: {detector_qe*100:.0f}%")
    print(f"  Total: {total_eff*100:.1f}%")
    
    print(f"\nEffective detected rates:")
    print(f"  Bright state: {effective_bright/1e6:.2f} Mcps")
    print(f"  Dark state: {effective_dark/1e3:.0f} kcps")
    print(f"  Background: {effective_bg/1e3:.0f} kcps")
    
    return effective_bright, effective_dark, effective_bg

def test_time_binning(effective_bright, effective_dark, effective_bg):
    """Test different time binning strategies"""
    
    print(f"\n⏱️ Time Binning Analysis")
    print("=" * 30)
    
    # After π-pulse state populations
    pop_0 = 0.15       # Some residual bright
    pop_dark = 0.85    # Mostly dark
    
    total_rate = pop_0 * effective_bright + pop_dark * effective_dark + effective_bg
    print(f"Total effective rate after π-pulse: {total_rate/1e6:.2f} Mcps")
    
    # Test different time bin sizes
    bin_sizes = [1e-9, 5e-9, 10e-9, 50e-9, 100e-9]  # 1 ns to 100 ns
    
    for bin_size in bin_sizes:
        expected_counts = total_rate * bin_size
        print(f"  {bin_size*1e9:.0f} ns bin: {expected_counts:.3f} counts/bin")
        
        # For good statistics, want ~1-10 counts per bin
        if 0.5 <= expected_counts <= 10:
            print(f"    ✅ Good for Poisson statistics")
        elif expected_counts < 0.5:
            print(f"    ⚠️ Too few counts - consider larger bins")
        else:
            print(f"    ⚠️ Too many counts - consider smaller bins")

def simulate_realistic_trace(effective_bright, effective_dark, effective_bg):
    """Simulate a realistic photon counting trace"""
    
    print(f"\n📊 Realistic Trace Simulation")
    print("=" * 30)
    
    # Parameters (optimized for GUI)
    readout_time = 5e-6    # 5 μs total readout (longer = more photons)
    time_bin = 100e-9      # 100 ns bins (better statistics)
    n_bins = int(readout_time / time_bin)
    
    # State populations after π-pulse
    pop_0 = 0.15
    pop_dark = 0.85
    
    total_rate = pop_0 * effective_bright + pop_dark * effective_dark + effective_bg
    
    print(f"Simulation parameters:")
    print(f"  Total readout: {readout_time*1e9:.0f} ns")
    print(f"  Time bin size: {time_bin*1e9:.0f} ns")
    print(f"  Number of bins: {n_bins}")
    print(f"  Expected counts/bin: {total_rate * time_bin:.2f}")
    
    # Generate photon trace
    photon_counts = []
    total_photons = 0
    
    for i in range(n_bins):
        # Add realistic noise
        laser_rin = 1.0 + 0.005 * np.random.randn()  # 0.5% RIN
        detector_noise = 1.0 + 0.01 * np.random.randn()  # 1% detector noise
        
        # Rare charge jumps
        charge_factor = 0.1 if np.random.random() < 1e-4 else 1.0
        
        # Calculate instantaneous rate
        instantaneous_rate = total_rate * laser_rin * detector_noise * charge_factor
        instantaneous_rate = max(0, instantaneous_rate)
        
        # Expected counts for this bin
        expected = instantaneous_rate * time_bin
        
        # Generate Poisson counts
        if expected > 0:
            counts = np.random.poisson(expected)
        else:
            counts = 0
            
        photon_counts.append(counts)
        total_photons += counts
    
    photon_counts = np.array(photon_counts)
    average_rate = total_photons / readout_time
    
    print(f"\nSimulation results:")
    print(f"  Total photons detected: {total_photons}")
    print(f"  Average detection rate: {average_rate/1e6:.2f} Mcps")
    print(f"  Max counts in any bin: {np.max(photon_counts)}")
    print(f"  Bins with photons: {np.sum(photon_counts > 0)}/{n_bins}")
    print(f"  Mean counts/bin: {np.mean(photon_counts):.2f}")
    print(f"  Std counts/bin: {np.std(photon_counts):.2f}")
    
    # Check if this looks realistic
    if total_photons > 50 and np.sum(photon_counts > 0) > n_bins/10:
        print(f"  ✅ Realistic photon trace!")
        return True
    else:
        print(f"  ❌ Trace might appear as flatline")
        return False

def main():
    """Main analysis"""
    
    # Calculate realistic rates
    effective_bright, effective_dark, effective_bg = calculate_realistic_nv_rates()
    
    # Test time binning
    test_time_binning(effective_bright, effective_dark, effective_bg)
    
    # Simulate realistic trace
    success = simulate_realistic_trace(effective_bright, effective_dark, effective_bg)
    
    print(f"\n🎯 Recommendations for GUI:")
    print(f"  - Use 10-50 ns time bins instead of 1 ns")
    print(f"  - Expect ~100-400 photons total for 600 ns readout")
    print(f"  - Should see clear signal above noise")
    print(f"  - Contrast should be visible in accumulated counts")
    
    if success:
        print(f"\n✅ GUI should show realistic photon data!")
    else:
        print(f"\n❌ May still appear as flatline - check parameters")
    
    return success

if __name__ == "__main__":
    main()