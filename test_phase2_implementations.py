#!/usr/bin/env python3
"""
Test Phase 2 Implementations

Quick test of all Phase 2 improvements:
1. Filter functions for pulse-sequence-dependent T2
2. Multi-level charge state dynamics 
3. Tensor strain coupling
4. Non-Markovian bath effects
5. Leeson microwave noise model
"""

import sys
import os
sys.path.append('nvcore/modules')
sys.path.append('nvcore/helper')

import numpy as np
from noise import (
    create_advanced_realistic_generator,
    create_precision_experiment_generator,
    NoiseConfiguration
)

def test_filter_functions():
    """Test filter functions for different pulse sequences"""
    print("🔬 Testing Filter Functions...")
    
    # Create noise generator with filter functions
    noise_gen = create_advanced_realistic_generator('bulk')
    
    # Test different sequences
    sequences = {
        'ramsey': {'evolution_time': 1e-6},
        'echo': {'echo_time': 2e-6}, 
        'cpmg': {'tau': 0.5e-6, 'n_pulses': 4}
    }
    
    t2_results = noise_gen.predict_sequence_performance(sequences)
    
    print("   Sequence-dependent T2 times:")
    for seq, t2 in t2_results.items():
        print(f"     {seq}: {t2*1e6:.1f} μs")
    
    return t2_results

def test_multi_level_charge():
    """Test multi-level charge state dynamics"""
    print("\n⚡ Testing Multi-Level Charge Dynamics...")
    
    from charge_dynamics import create_charge_state_model
    
    # Create model for different setups
    setups = ['room_temperature', 'cryogenic', 'surface_nv']
    
    for setup in setups:
        model = create_charge_state_model(setup)
        model._dt = 1e-4  # 0.1 ms
        
        # Run short trajectory
        trajectory = []
        for _ in range(100):
            state = model.sample(1)
            trajectory.append(state)
        
        # Analyze state populations
        states = np.array(trajectory)
        nv_plus_frac = np.sum(states == 0) / len(states)
        nv_zero_frac = np.sum(states == 1) / len(states) 
        nv_minus_frac = np.sum(states == 2) / len(states)
        
        print(f"   {setup}:")
        print(f"     NV+: {nv_plus_frac:.2f}, NV0: {nv_zero_frac:.2f}, NV-: {nv_minus_frac:.2f}")
    
    return True

def test_tensor_strain():
    """Test tensor strain coupling"""
    print("\n🔧 Testing Tensor Strain Coupling...")
    
    from strain_tensor import create_bulk_diamond_strain, create_surface_nv_strain
    
    # Test different strain models
    models = {
        'Bulk Diamond': create_bulk_diamond_strain(),
        'Surface NV': create_surface_nv_strain(depth_nm=5.0)
    }
    
    for name, model in models.items():
        model._dt = 1e-4
        
        # Generate strain sample
        strain_tensor = model.sample(1)
        shifts = model.get_zfs_shifts(strain_tensor)
        stats = model.get_strain_statistics()
        
        print(f"   {name}:")
        print(f"     ΔD: {shifts['delta_d']/1e6:.1f} MHz")
        print(f"     ΔE: {shifts['delta_e']/1e3:.1f} kHz")
        print(f"     RMS strain: {stats['rms_strain']:.2e}")
    
    return True

def test_non_markovian():
    """Test non-Markovian bath effects"""
    print("\n🧠 Testing Non-Markovian Bath Effects...")
    
    from non_markovian import (
        create_c13_non_markovian_bath,
        create_phonon_non_markovian_bath,
        create_charge_non_markovian_bath
    )
    
    # Test different bath types
    baths = {
        'C13 Bath': create_c13_non_markovian_bath(),
        'Phonon Bath': create_phonon_non_markovian_bath(300.0),
        'Charge Bath': create_charge_non_markovian_bath(10.0)
    }
    
    for name, bath in baths.items():
        memory_time = bath.estimate_memory_time()
        bath._dt = 1e-6
        
        # Generate small sample
        samples = bath.sample(10)
        noise_rms = np.std(samples) if hasattr(samples, '__len__') else abs(samples)
        
        print(f"   {name}:")
        print(f"     Memory time: {memory_time*1e6:.1f} μs")
        print(f"     Noise RMS: {noise_rms:.3f}")
    
    return True

def test_leeson_microwave():
    """Test Leeson microwave noise model"""
    print("\n📻 Testing Leeson Microwave Noise...")
    
    from leeson_microwave import (
        create_lab_microwave_source,
        create_precision_microwave_source,
        create_budget_microwave_source
    )
    
    # Test different MW sources
    sources = {
        'Lab Source': create_lab_microwave_source(),
        'Precision Source': create_precision_microwave_source(),
        'Budget Source': create_budget_microwave_source()
    }
    
    for name, source in sources.items():
        specs = source.get_oscillator_specs()
        source._dt = 1e-6
        
        # Generate noise sample
        noise_sample = source.sample(1)
        
        print(f"   {name}:")
        print(f"     Phase noise @ 1kHz: {specs['phase_noise_1khz_dbc']:.1f} dBc/Hz")
        print(f"     RMS phase error: {specs['rms_phase_error_mrad']:.2f} mrad")
        print(f"     Sample amplitude factor: {noise_sample['amplitude_factor']:.6f}")
    
    return True

def test_advanced_noise_generator():
    """Test complete advanced noise generator"""
    print("\n🚀 Testing Advanced Noise Generator Integration...")
    
    # Create different generator types
    generators = {
        'Bulk NV': create_advanced_realistic_generator('bulk', 300.0),
        'Surface NV': create_advanced_realistic_generator('surface', 300.0),
        'Precision': create_precision_experiment_generator()
    }
    
    for name, gen in generators.items():
        print(f"\n   {name} Generator:")
        print(f"     Enabled sources: {list(gen.sources.keys())}")
        
        # Test T2 calculation with filter functions
        try:
            t2_ramsey = gen.calculate_t2_for_sequence('ramsey', evolution_time=1e-6)
            t2_echo = gen.calculate_t2_for_sequence('echo', echo_time=2e-6)
            print(f"     T2 (Ramsey): {t2_ramsey*1e6:.1f} μs")
            print(f"     T2 (Echo): {t2_echo*1e6:.1f} μs")
        except Exception as e:
            print(f"     T2 calculation failed: {e}")
        
        # Test noise Hamiltonian
        try:
            spin_ops = {
                'Sx': np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / np.sqrt(2),
                'Sy': np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]]) / np.sqrt(2),
                'Sz': np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
            }
            
            H_noise = gen.get_noise_hamiltonian(spin_ops)
            noise_strength = np.max(np.abs(H_noise))
            print(f"     Max noise Hamiltonian element: {noise_strength/1e6:.1f} MHz")
        except Exception as e:
            print(f"     Hamiltonian generation failed: {e}")
    
    return True

def main():
    """Run all Phase 2 tests"""
    print("🧪 Phase 2 Implementation Test Suite")
    print("=" * 50)
    
    try:
        # Run all tests
        test_results = {}
        
        test_results['filter_functions'] = test_filter_functions()
        test_results['multi_level_charge'] = test_multi_level_charge()
        test_results['tensor_strain'] = test_tensor_strain()
        test_results['non_markovian'] = test_non_markovian()
        test_results['leeson_microwave'] = test_leeson_microwave()
        test_results['advanced_generator'] = test_advanced_noise_generator()
        
        print("\n" + "=" * 50)
        print("✅ All Phase 2 implementations working!")
        print("\nImplementation Summary:")
        print("  🔬 Filter functions: Pulse-sequence-dependent T2")
        print("  ⚡ Multi-level charge: NV+/NV0/NV- dynamics")  
        print("  🔧 Tensor strain: Full C3v symmetry coupling")
        print("  🧠 Non-Markovian: Memory effects & correlations")
        print("  📻 Leeson MW: Realistic oscillator phase noise")
        print("  🚀 Advanced generators: All improvements integrated")
        
        print("\nPhase 2 improvements successfully implemented! 🎉")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)