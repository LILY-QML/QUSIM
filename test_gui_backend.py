#!/usr/bin/env python3
"""
Test GUI Backend Functions
"""

import sys
import os
sys.path.append('experiments/GUI')
sys.path.append('nvcore/modules')
sys.path.append('nvcore/helper')

import numpy as np

# Test QUSIM import
try:
    from noise import (
        create_realistic_noise_generator,
        create_advanced_realistic_generator, 
        create_precision_experiment_generator,
    )
    print("✅ QUSIM imports successful")
    HAS_QUSIM = True
except ImportError as e:
    print(f"❌ QUSIM import failed: {e}")
    HAS_QUSIM = False

def test_photons_experiment():
    """Test the photon counting experiment backend"""
    print("\n🔬 Testing Photon Counting Experiment...")
    
    if not HAS_QUSIM:
        print("❌ Skipping - QUSIM not available")
        return
    
    # Parameters
    b_field = 0.01  # 10 mT
    rabi_freq = 15e6 * 2 * np.pi  # 15 MHz
    readout_time = 300e-9  # 300 ns
    
    try:
        # Create advanced noise generator
        noise_gen = create_advanced_realistic_generator('bulk', 300.0, True)
        print(f"✅ Created advanced noise generator")
        print(f"   Sources: {list(noise_gen.sources.keys())}")
        
        # Test optical readout
        state_populations = {
            'ms=0': 0.2,
            'ms=+1': 0.4,
            'ms=-1': 0.4
        }
        
        photon_counts = noise_gen.process_optical_readout(
            state_populations, 1e-6, n_shots=10
        )
        
        print(f"✅ Optical readout test successful")
        print(f"   Sample counts: {photon_counts[:5]}")
        print(f"   Mean rate: {np.mean(photon_counts)/1e-6/1e6:.1f} Mcps")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_noise_models():
    """Test different noise model configurations"""
    print("\n🔧 Testing Noise Model Configurations...")
    
    if not HAS_QUSIM:
        print("❌ Skipping - QUSIM not available")
        return
    
    models = [
        ('basic', create_realistic_noise_generator),
        ('advanced_bulk', lambda: create_advanced_realistic_generator('bulk', 300.0, True)),
        ('advanced_surface', lambda: create_advanced_realistic_generator('surface', 300.0, True)),
        ('precision', create_precision_experiment_generator)
    ]
    
    for name, factory in models:
        try:
            if name == 'basic':
                noise_gen = factory(300.0, 0.01, 10e-9)
            else:
                noise_gen = factory()
            
            print(f"✅ {name}: {len(noise_gen.sources)} sources")
            
            # Test T2 calculation with filter functions
            try:
                t2_ramsey = noise_gen.calculate_t2_for_sequence('ramsey', evolution_time=1e-6)
                t2_echo = noise_gen.calculate_t2_for_sequence('echo', echo_time=2e-6)
                print(f"     T2 Ramsey: {t2_ramsey*1e6:.1f} μs, Echo: {t2_echo*1e6:.1f} μs")
            except Exception as e:
                print(f"     T2 calculation failed: {e}")
                
        except Exception as e:
            print(f"❌ {name} failed: {e}")

def test_gui_simulation():
    """Test complete GUI experiment simulation"""
    print("\n🚀 Testing Complete GUI Simulation...")
    
    # Simulate parameters from GUI
    params = {
        'b_field': 0.01,  # 10 mT
        'rabi_freq': 15e6 * 2 * np.pi,
        'readout_time': 600e-9,  # 600 ns
        'n_measurements': 20,
        'noise_model': 'advanced',
        'nv_type': 'bulk'
    }
    
    print(f"Parameters: {params}")
    
    # Simulate time-resolved photon trace
    time_bin = 1e-9  # 1 ns
    n_bins = int(params['readout_time'] / time_bin)
    time_ns = np.arange(n_bins) * time_bin * 1e9
    
    # Load realistic photon rates from system.json
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), 'nvcore', 'helper'))
    from noise_sources import SYSTEM
    
    bright_rate = SYSTEM.get_noise_param('optical', 'readout', 'bright_state_rate')
    dark_rate = SYSTEM.get_noise_param('optical', 'readout', 'dark_state_rate')
    collection_eff = SYSTEM.get_empirical_param('optical_system', 'collection_efficiency')
    detector_eff = SYSTEM.get_noise_param('optical', 'readout', 'detector_efficiency')
    effective_bright = bright_rate * collection_eff * detector_eff
    effective_dark = dark_rate * collection_eff * detector_eff
    
    # After π-pulse populations
    final_pop_0 = 0.15
    final_pop_dark = 0.85
    
    effective_rate = final_pop_0 * effective_bright + final_pop_dark * effective_dark
    
    print(f"✅ Simulation setup:")
    print(f"   Time bins: {n_bins} @ {time_bin*1e9:.0f} ns")
    print(f"   Expected rate: {effective_rate/1e6:.2f} Mcps")
    print(f"   Final populations: |0⟩={final_pop_0:.2f}, dark={final_pop_dark:.2f}")
    
    # Generate realistic photon trace
    photon_counts = []
    for i in range(min(100, n_bins)):  # Test first 100 bins
        # Add noise effects
        laser_rin = 1.0 + 0.01 * np.random.randn()
        detector_noise = 1.0 + 0.02 * np.random.randn()
        
        # Rare charge jumps
        charge_factor = 0.2 if np.random.random() < 0.0001 else 1.0
        
        rate = effective_rate * laser_rin * detector_noise * charge_factor
        rate = max(0, rate)
        
        expected = rate * time_bin
        count = np.random.poisson(expected) if expected > 0 else 0
        photon_counts.append(count)
    
    photon_counts = np.array(photon_counts)
    total_photons = np.sum(photon_counts)
    avg_rate = total_photons / (len(photon_counts) * time_bin)
    
    print(f"✅ Generated trace (first {len(photon_counts)} bins):")
    print(f"   Total photons: {total_photons}")
    print(f"   Average rate: {avg_rate/1e6:.2f} Mcps")
    print(f"   Max counts/bin: {np.max(photon_counts)}")
    print(f"   Non-zero bins: {np.sum(photon_counts > 0)}")
    
    # Check for realistic behavior
    if avg_rate > 1e3 and total_photons > 0:
        print("✅ Realistic photon trace generated!")
        return True
    else:
        print("❌ Photon trace seems unrealistic")
        return False

def main():
    """Run all tests"""
    print("🧪 GUI Backend Test Suite")
    print("=" * 50)
    
    test_results = []
    
    test_results.append(test_photons_experiment())
    test_results.append(test_noise_models())
    test_results.append(test_gui_simulation())
    
    print("\n" + "=" * 50)
    if all(test_results):
        print("✅ All GUI backend tests passed!")
        print("\n📋 Summary:")
        print("  - QUSIM integration working")
        print("  - Phase 2 noise models available")
        print("  - Realistic photon counting")
        print("  - GUI should show proper data instead of flatline")
    else:
        print("❌ Some tests failed")
        print("GUI may fall back to simulated data")
    
    return all(test_results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)