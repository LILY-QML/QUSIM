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
        Get Knight shift Hamiltonian with full 3D components
        
        Args:
            c13_operators: C13 spin operators
            nv_state: Current NV quantum state
            
        Returns:
            Knight shift Hamiltonian [Hz]
        """
        n_c13 = len(c13_operators)
        if n_c13 == 0:
            return np.array([[0.0]])
            
        # Volle 3D Knight Shift
        S_expectation = self._calculate_full_spin_expectation(nv_state)
        
        H_knight = np.zeros((2**n_c13, 2**n_c13), dtype=complex)
        
        for i in range(n_c13):
            # Anisotrope Knight Shift
            B_knight = np.array([
                self.chi_perpendicular * S_expectation[0],
                self.chi_perpendicular * S_expectation[1],
                self.chi_parallel * S_expectation[2]
            ])
            
            # Volle Vektor-Kopplung
            for k, B_k in enumerate(B_knight):
                op_name = ['Ix', 'Iy', 'Iz'][k]
                H_knight -= self.gamma_n * B_k * c13_operators[i][op_name]
        
        return H_knight * 2 * np.pi
    
    def _calculate_full_spin_expectation(self, nv_state: np.ndarray) -> np.ndarray:
        """Calculate full 3D spin expectation value"""
        
        # NV spin operators (S=1)
        Sx = (1/np.sqrt(2)) * np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex)
        Sy = (1/np.sqrt(2)) * np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex)
        Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]], dtype=complex)
        
        if nv_state.ndim == 1:
            # State vector
            Sx_expectation = np.real(np.conj(nv_state) @ Sx @ nv_state)
            Sy_expectation = np.real(np.conj(nv_state) @ Sy @ nv_state)
            Sz_expectation = np.real(np.conj(nv_state) @ Sz @ nv_state)
        else:
            # Density matrix
            Sx_expectation = np.real(np.trace(Sx @ nv_state))
            Sy_expectation = np.real(np.trace(Sy @ nv_state))
            Sz_expectation = np.real(np.trace(Sz @ nv_state))
        
        return np.array([Sx_expectation, Sy_expectation, Sz_expectation])