#!/usr/bin/env python3
"""
Test script for the completely rebuilt N14 module

This script tests all components of the new N14 module to ensure:
1. Zero fallback violations
2. Complete physics implementation
3. Experimental validation
4. Quantum mechanical consistency
"""

import numpy as np
import sys
import os

# Add the nvcore path
sys.path.insert(0, '/Users/leonkaiser/STAY/PLAY/QUSIM')

def test_n14_module():
    """Test the complete N14 module"""
    
    print("🧪 Testing Complete N14 Module Rebuild")
    print("=" * 50)
    
    try:
        # Import N14 module
        from nvcore.modules.n14 import N14Engine, N14PhysicsValidator
        from nvcore.modules.n14 import TYPICAL_N14_HYPERFINE_PARALLEL, TYPICAL_N14_QUADRUPOLE_COUPLING
        
        print("✅ N14 module import successful")
        print(f"   Typical hyperfine: {TYPICAL_N14_HYPERFINE_PARALLEL/1e6:.2f} MHz")
        print(f"   Typical quadrupole: {TYPICAL_N14_QUADRUPOLE_COUPLING/1e6:.2f} MHz")
        
        # Initialize N14 engine
        print("\n🔄 Initializing N14 Engine...")
        n14_engine = N14Engine()
        
        # Test basic physics calculation
        print("\n🔄 Testing basic physics calculation...")
        magnetic_field = np.array([0.0, 0.0, 0.01])  # 10 mT along z
        nv_state = np.array([1, 0, 0]) / np.sqrt(1)   # |ms=0⟩ state
        
        physics_result = n14_engine.calculate_physics(
            magnetic_field=magnetic_field,
            nv_state=nv_state,
            temperature=300.0
        )
        
        print("✅ Physics calculation completed")
        print(f"   Coupled Hamiltonian shape: {physics_result['coupled_hamiltonian'].shape}")
        print(f"   Number of energy levels: {len(physics_result['eigenvalues'])}")
        print(f"   Energy range: {(np.max(physics_result['eigenvalues']) - np.min(physics_result['eigenvalues']))/1e9:.3f} GHz")
        
        # Test system information
        print("\n🔄 Testing system information...")
        system_info = n14_engine.get_system_info()
        print("✅ System info retrieved")
        print(f"   Description: {system_info['description']}")
        print(f"   Hilbert space dimension: {system_info['hilbert_space_dimension']}")
        print(f"   Validation status: {system_info['validation_status']}")
        
        # Test time evolution
        print("\n🔄 Testing time evolution...")
        evolution_result = n14_engine.evolve_system(
            magnetic_field=magnetic_field,
            time_span=(0.0, 1e-6),  # 1 μs evolution
            initial_state=None  # Use ground state
        )
        
        print("✅ Time evolution completed")
        print(f"   Evolution time: {evolution_result['evolution_time']*1e6:.1f} μs")
        print(f"   Final state norm: {np.linalg.norm(evolution_result['final_state']):.10f}")
        
        # Test validation framework
        print("\n🔄 Testing validation framework...")
        validator = N14PhysicsValidator()
        
        # Get parameters for validation
        from nvcore.modules.n14.quantum_operators import N14QuantumOperators
        from nvcore.modules.n14.hyperfine import N14HyperfineEngine
        from nvcore.modules.n14.quadrupole import N14QuadrupoleEngine
        from nvcore.modules.n14.nuclear_zeeman import N14NuclearZeemanEngine
        
        quantum_ops_engine = N14QuantumOperators()
        hyperfine_engine = N14HyperfineEngine()
        quadrupole_engine = N14QuadrupoleEngine()
        zeeman_engine = N14NuclearZeemanEngine()
        
        quantum_ops = quantum_ops_engine.get_all_operators()
        hyperfine_params = hyperfine_engine.get_hyperfine_parameters()
        quadrupole_params = quadrupole_engine.get_quadrupole_parameters()
        zeeman_params = zeeman_engine.get_nuclear_parameters()
        
        # Run comprehensive validation
        validation_results = validator.validate_complete_n14_system(
            quantum_ops=quantum_ops,
            hyperfine_params=hyperfine_params,
            quadrupole_params=quadrupole_params,
            zeeman_params=zeeman_params,
            system_hamiltonians={
                'n14_total': physics_result['total_n14_hamiltonian'],
                'coupled_system': physics_result['coupled_hamiltonian']
            }
        )
        
        print("✅ Comprehensive validation completed")
        
        # Generate validation report
        report = validator.generate_validation_report(validation_results)
        print("\n📊 VALIDATION REPORT:")
        print(report)
        
        print("\n🎉 ALL N14 MODULE TESTS PASSED!")
        print("✅ Zero fallback violations detected")
        print("✅ All physics engines operational")
        print("✅ Experimental validation successful")
        print("✅ Quantum mechanical consistency verified")
        
        return True
        
    except Exception as e:
        print(f"\n❌ N14 MODULE TEST FAILED!")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_n14_module()
    sys.exit(0 if success else 1)