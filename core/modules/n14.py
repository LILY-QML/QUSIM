#!/usr/bin/env python3
"""
N14 Nuclear Spin Physics Module
Hyperfine coupling, quadrupole interaction, nuclear Zeeman effect
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any


class N14Physics:
    """N14 nuclear physics for NV centers"""
    
    def __init__(self, physical_constants: Dict[str, float], n14_params: Dict[str, float]):
        """
        Args:
            physical_constants: hbar, gamma_n_14n
            n14_params: A_para_Hz, A_perp_Hz, quadrupole_P_Hz, etc.
        """
        # Physical constants
        self.hbar = physical_constants['hbar']
        self.gamma_n14 = physical_constants.get('gamma_n_14n', 3.077e6)  # Hz/T
        
        # N14 parameters
        self.A_para_Hz = n14_params['A_para_Hz']
        self.A_perp_Hz = n14_params['A_perp_Hz']
        self.quadrupole_P_Hz = n14_params.get('quadrupole_P_Hz', 0.0)
        self.nuclear_zeeman_factor = n14_params.get('nuclear_zeeman_factor', 1.0)
        
        # Build N14 operators
        self._build_operators()
    
    def _build_operators(self):
        """N14 spin-1 operators"""
        # Ladder operators for I=1: |+1⟩, |0⟩, |-1⟩
        I_plus = np.array([
            [0, np.sqrt(2), 0],
            [0, 0, np.sqrt(2)],
            [0, 0, 0]
        ], dtype=complex)
        
        I_minus = np.array([
            [0, 0, 0],
            [np.sqrt(2), 0, 0],
            [0, np.sqrt(2), 0]
        ], dtype=complex)
        
        self.operators = {
            'Ix': (I_plus + I_minus) / 2,
            'Iy': (I_plus - I_minus) / (2j),
            'Iz': np.diag([1., 0., -1.]).astype(complex),
            'I+': I_plus,
            'I-': I_minus,
            'I': np.eye(3, dtype=complex),
            'Iz2': np.diag([1., 0., 1.]).astype(complex),  # Iz²
            'I2': 2 * np.eye(3, dtype=complex)  # I(I+1) = 2 for I=1
        }
    
    def hyperfine_hamiltonian(self, nv_operators: Dict[str, np.ndarray], 
                            c13_dimension: int = 1) -> np.ndarray:
        """NV-N14 hyperfine coupling"""
        nv_dim = 3
        n14_dim = 3
        joint_dim = nv_dim * n14_dim * c13_dimension
        
        H_hf = np.zeros((joint_dim, joint_dim), dtype=complex)
        
        # Joint operators: NV ⊗ N14 ⊗ C13
        Sz_joint = np.kron(np.kron(nv_operators['Sz'], np.eye(n14_dim)), np.eye(c13_dimension))
        Sx_joint = np.kron(np.kron(nv_operators['Sx'], np.eye(n14_dim)), np.eye(c13_dimension))
        
        Iz_joint = np.kron(np.kron(np.eye(nv_dim), self.operators['Iz']), np.eye(c13_dimension))
        Ix_joint = np.kron(np.kron(np.eye(nv_dim), self.operators['Ix']), np.eye(c13_dimension))
        
        # Hyperfine coupling: A∥SzIz + A⊥SxIx
        H_hf += 2 * np.pi * self.A_para_Hz * (Sz_joint @ Iz_joint)
        H_hf += 2 * np.pi * self.A_perp_Hz * (Sx_joint @ Ix_joint)
        
        return jnp.array(H_hf)
    
    def quadrupole_hamiltonian(self, c13_dimension: int = 1) -> np.ndarray:
        """N14 quadrupole interaction"""
        if abs(self.quadrupole_P_Hz) < 1e-6:
            nv_dim = 3
            n14_dim = 3
            joint_dim = nv_dim * n14_dim * c13_dimension
            return jnp.zeros((joint_dim, joint_dim), dtype=complex)
        
        nv_dim = 3
        n14_dim = 3
        joint_dim = nv_dim * n14_dim * c13_dimension
        
        # Quadrupole: P[Iz² - I(I+1)/3] = P[Iz² - 2/3]
        Iz2_joint = np.kron(np.kron(np.eye(nv_dim), self.operators['Iz2']), np.eye(c13_dimension))
        I_avg = (2/3) * np.eye(joint_dim, dtype=complex)
        
        H_quad = 2 * np.pi * self.quadrupole_P_Hz * (Iz2_joint - I_avg)
        
        return jnp.array(H_quad)
    
    def nuclear_zeeman_hamiltonian(self, B_field: np.ndarray, c13_dimension: int = 1) -> np.ndarray:
        """N14 nuclear Zeeman interaction"""
        nv_dim = 3
        n14_dim = 3
        joint_dim = nv_dim * n14_dim * c13_dimension
        
        # Nuclear Zeeman: -γₙ B⃗ · I⃗
        Ix_joint = np.kron(np.kron(np.eye(nv_dim), self.operators['Ix']), np.eye(c13_dimension))
        Iy_joint = np.kron(np.kron(np.eye(nv_dim), self.operators['Iy']), np.eye(c13_dimension))
        Iz_joint = np.kron(np.kron(np.eye(nv_dim), self.operators['Iz']), np.eye(c13_dimension))
        
        H_zeeman = -2 * np.pi * self.gamma_n14 * self.nuclear_zeeman_factor * (
            B_field[0] * Ix_joint + 
            B_field[1] * Iy_joint + 
            B_field[2] * Iz_joint
        )
        
        return jnp.array(H_zeeman)
    
    def complete_n14_hamiltonian(self, nv_operators: Dict[str, np.ndarray],
                               B_field: np.ndarray, c13_dimension: int = 1) -> np.ndarray:
        """Complete N14 Hamiltonian"""
        H_hf = self.hyperfine_hamiltonian(nv_operators, c13_dimension)
        H_quad = self.quadrupole_hamiltonian(c13_dimension)
        H_zeeman = self.nuclear_zeeman_hamiltonian(B_field, c13_dimension)
        
        return H_hf + H_quad + H_zeeman
    
    def energy_levels(self, B_field_T: float = 0.0) -> np.ndarray:
        """N14 energy levels in magnetic field"""
        # Bare N14 Hamiltonian
        H_n14 = np.zeros((3, 3), dtype=complex)
        
        # Quadrupole splitting
        if abs(self.quadrupole_P_Hz) > 1e-6:
            H_quad = self.quadrupole_P_Hz * (self.operators['Iz2'] - (2/3) * self.operators['I'])
            H_n14 += 2 * np.pi * H_quad
        
        # Nuclear Zeeman
        if abs(B_field_T) > 1e-9:
            H_zeeman = -self.gamma_n14 * B_field_T * self.operators['Iz']
            H_n14 += 2 * np.pi * H_zeeman
        
        eigenvals, _ = np.linalg.eigh(H_n14)
        return eigenvals / (2 * np.pi)  # Return in Hz
    
    def transition_frequencies(self, B_field_T: float = 0.0) -> Dict[str, float]:
        """N14 transition frequencies"""
        levels = self.energy_levels(B_field_T)
        
        # For I=1: |+1⟩, |0⟩, |-1⟩
        return {
            'plus1_to_0': levels[1] - levels[0],    # |+1⟩ → |0⟩
            '0_to_minus1': levels[2] - levels[1],   # |0⟩ → |-1⟩
            'plus1_to_minus1': levels[2] - levels[0]  # |+1⟩ → |-1⟩
        }
    
    def validate_operators(self) -> bool:
        """Validate N14 operator commutation relations"""
        Ix = self.operators['Ix']
        Iy = self.operators['Iy']
        Iz = self.operators['Iz']
        
        # [Ix, Iy] = i Iz for I=1
        commutator = Ix @ Iy - Iy @ Ix
        expected = 1j * Iz
        
        if not np.allclose(commutator, expected, atol=1e-12):
            return False
        
        # I² = I(I+1) = 2 for I=1
        I_squared = Ix @ Ix + Iy @ Iy + Iz @ Iz
        expected_magnitude = 2 * np.eye(3, dtype=complex)
        
        if not np.allclose(I_squared, expected_magnitude, atol=1e-12):
            return False
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """N14 system information"""
        return {
            'spin': 1,
            'hilbert_dimension': 3,
            'A_para_Hz': self.A_para_Hz,
            'A_perp_Hz': self.A_perp_Hz,
            'quadrupole_P_Hz': self.quadrupole_P_Hz,
            'gamma_n14_Hz_per_T': self.gamma_n14,
            'nuclear_zeeman_factor': self.nuclear_zeeman_factor,
            'operators_validated': self.validate_operators(),
            'energy_levels_0T_Hz': self.energy_levels(0.0).tolist(),
            'transition_frequencies_0T_Hz': self.transition_frequencies(0.0)
        }