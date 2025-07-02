"""
Knight Shift Engine

Implementation of Knight shift / Overhauser field effects.
NV spin polarization affects C13 effective magnetic field.
"""

import numpy as np
from typing import Dict, Optional
import sys
import os

# Add path for system constants
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'helper'))
from noise_sources import SYSTEM


class KnightShiftEngine:
    """
    Knight shift implementation
    
    The NV electronic spin creates an additional magnetic field at the C13 nuclei:
    B_eff = B_applied + χ⟨S⟩
    
    This modifies the nuclear Zeeman interaction.
    """
    
    def __init__(self):
        """Initialize Knight shift engine"""
        # Knight shift tensor components (empirical)
        self.chi_parallel = 1e-4     # Along NV axis
        self.chi_perpendicular = 5e-5  # Perpendicular to NV axis
        
        self.gamma_n = SYSTEM.get_constant('nv_center', 'gamma_n_13c')
        
    def get_knight_shift_hamiltonian(self, c13_operators: Dict[int, Dict[str, np.ndarray]],
                                   nv_state: np.ndarray) -> np.ndarray:
        """
        Get Knight shift Hamiltonian
        
        Args:
            c13_operators: C13 spin operators
            nv_state: Current NV quantum state
            
        Returns:
            Knight shift Hamiltonian [Hz]
        """
        n_c13 = len(c13_operators)
        if n_c13 == 0:
            return np.array([[0.0]])
            
        # Extract NV spin expectation values
        if nv_state.ndim == 1:
            # State vector - need to compute ⟨S⟩
            Sz_expectation = np.real(np.conj(nv_state) @ np.diag([-1, 0, 1]) @ nv_state)
        else:
            # Density matrix
            Sz_expectation = np.real(np.trace(np.diag([-1, 0, 1]) @ nv_state))
            
        # Knight shift field
        B_knight = self.chi_parallel * Sz_expectation  # Simplified: only z-component
        
        # Construct Hamiltonian
        dim = 2**n_c13
        H_knight = np.zeros((dim, dim), dtype=complex)
        
        for i in range(n_c13):
            # Additional Zeeman interaction
            knight_freq = -self.gamma_n * B_knight
            H_knight += knight_freq * c13_operators[i]['Iz']
            
        return H_knight * 2 * np.pi