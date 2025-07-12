"""
Nuclear Zeeman Engine

Implementation of ¹³C nuclear Zeeman effect in magnetic fields.
"""

import numpy as np
from typing import Dict, Optional
import sys
import os

# Add path for system constants
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'helper'))
from noise_sources import SYSTEM


class NuclearZeemanEngine:
    """
    Nuclear Zeeman effect for ¹³C nuclei
    
    H_zeeman = -γₙ · B · I = -γₙ(Bₓ·Iₓ + Bᵧ·Iᵧ + Bᵤ·Iᵤ)
    """
    
    def __init__(self, magnetic_field: np.ndarray = None):
        """
        Initialize nuclear Zeeman engine
        
        Args:
            magnetic_field: Applied magnetic field [T], shape (3,)
        """
        self.magnetic_field = magnetic_field if magnetic_field is not None else np.zeros(3)
        self.gamma_n = SYSTEM.get_constant('nv_center', 'gamma_n_13c')
        
    def set_magnetic_field(self, B_field: np.ndarray):
        """Set magnetic field"""
        self.magnetic_field = np.asarray(B_field)
        
    def get_zeeman_hamiltonian(self, c13_operators: Dict[int, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Get nuclear Zeeman Hamiltonian
        
        Args:
            c13_operators: C13 spin operators
            
        Returns:
            Zeeman Hamiltonian matrix [Hz]
        """
        n_c13 = len(c13_operators)
        if n_c13 == 0:
            return np.array([[0.0]])
            
        dim = 2**n_c13
        H_zeeman = np.zeros((dim, dim), dtype=complex)
        
        # Add Zeeman term for each C13
        for i in range(n_c13):
            # -γₙ · B · I
            zeeman_freq = -self.gamma_n * self.magnetic_field
            
            H_zeeman += zeeman_freq[0] * c13_operators[i]['Ix']
            H_zeeman += zeeman_freq[1] * c13_operators[i]['Iy'] 
            H_zeeman += zeeman_freq[2] * c13_operators[i]['Iz']
            
        return H_zeeman * 2 * np.pi  # Convert to angular frequency