"""
N14 Nuclear Spin Module - Ultra-Realistic Quantum Simulation

ZERO TOLERANCE POLICY:
- NO MOCKS whatsoever
- NO FALLBACKS of any kind  
- NO APPROXIMATIONS in quantum mechanics
- NO np.zeros() returns
- NO exception catching without proper handling
- NO inf/nan values as fallbacks

This module implements COMPLETE quantum mechanical treatment of ¹⁴N nuclear spin
strongly coupled to NV centers with I=1 nuclear angular momentum.

Author: Leon Kaiser
Institution: Goethe University Frankfurt, MSQC
Contact: l.kaiser@em.uni-frankfurt.de
"""

from .core import N14Engine
from .quantum_operators import N14QuantumOperators
from .hyperfine import N14HyperfineEngine
from .quadrupole import N14QuadrupoleEngine
from .nuclear_zeeman import N14NuclearZeemanEngine
from .rf_control import N14RFControlEngine
from .validation import N14PhysicsValidator

__version__ = "2.0.0"
__author__ = "Leon Kaiser"

# Physical constants for N14
N14_NUCLEAR_SPIN = 1.0
N14_GYROMAGNETIC_RATIO = 0.3077e6  # Hz/T
N14_QUADRUPOLE_MOMENT = 2.044e-30  # C⋅m²
N14_NATURAL_ABUNDANCE = 0.9963  # 99.63%

# Coupling constants (experimentally validated)
TYPICAL_N14_HYPERFINE_PARALLEL = -2.16e6  # Hz
TYPICAL_N14_HYPERFINE_PERPENDICULAR = -2.7e6  # Hz  
TYPICAL_N14_QUADRUPOLE_COUPLING = -4.95e6  # Hz

__all__ = [
    'N14Engine',
    'N14QuantumOperators', 
    'N14HyperfineEngine',
    'N14QuadrupoleEngine',
    'N14NuclearZeemanEngine',
    'N14RFControlEngine',
    'N14PhysicsValidator',
    'N14_NUCLEAR_SPIN',
    'N14_GYROMAGNETIC_RATIO',
    'N14_QUADRUPOLE_MOMENT',
    'TYPICAL_N14_HYPERFINE_PARALLEL',
    'TYPICAL_N14_HYPERFINE_PERPENDICULAR',
    'TYPICAL_N14_QUADRUPOLE_COUPLING'
]

def validate_n14_module():
    """Validate complete N14 module - ZERO tolerance for failures"""
    from .validation import N14PhysicsValidator
    
    validator = N14PhysicsValidator()
    results = validator.validate_complete_module()
    
    failed_tests = [test for test, passed in results.items() if not passed]
    
    if failed_tests:
        raise RuntimeError(
            f"N14 MODULE VALIDATION FAILED!\n"
            f"Failed tests: {failed_tests}\n"
            f"EVERY SINGLE TEST MUST PASS - NO EXCEPTIONS!"
        )
    
    print("✅ N14 module validation: ALL TESTS PASSED")
    return True