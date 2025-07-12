"""
C13 Nuclear Spin Bath Module

Ultra-realistic quantum mechanical implementation of ¹³C nuclear spin baths
for NV center quantum simulations.

Features:
- Full quantum mechanical C13 spins (I=½)
- Anisotropic hyperfine coupling to NV center
- Nuclear-nuclear dipolar interactions  
- Dynamic environment effects
- RF and MW control capabilities
- NO MOCKS, NO FALLBACKS - pure quantum physics
"""

from .core import C13BathEngine
from .hyperfine import HyperfineEngine
from .nuclear_zeeman import NuclearZeemanEngine
from .knight_shift import KnightShiftEngine
from .rf_control import RFControlEngine
from .mw_dnp import MicrowaveDNPEngine
from .quantum_operators import C13QuantumOperators

__all__ = [
    'C13BathEngine',
    'HyperfineEngine', 
    'NuclearZeemanEngine',
    'KnightShiftEngine',
    'RFControlEngine',
    'MicrowaveDNPEngine',
    'C13QuantumOperators'
]