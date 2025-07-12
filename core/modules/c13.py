#!/usr/bin/env python3
"""
C13 Nuclear Spin Physics Module
Hyperfine coupling, dipole-dipole interactions, nuclear bath effects
"""

import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any


class C13Physics:
    """C13 nuclear spin physics for NV centers"""
    
    def __init__(self, c13_positions: np.ndarray, physical_constants: Dict[str, float]):
        """
        Args:
            c13_positions: C13 positions in meters, shape (N, 3)
            physical_constants: mu_0, gamma_e, gamma_n_13c, hbar
        """
        self.c13_positions = np.asarray(c13_positions)
        if self.c13_positions.ndim == 1:
            self.c13_positions = self.c13_positions.reshape(1, 3)
        
        self.n_c13 = len(self.c13_positions)
        self.nv_position = np.array([0.0, 0.0, 0.0])
        
        # Physical constants
        self.mu_0 = physical_constants['mu_0']
        self.gamma_e = physical_constants['gamma_e']
        self.gamma_n = physical_constants['gamma_n_13c']
        self.hbar = physical_constants['hbar']
        
        # Build operators and hyperfine tensors
        self._build_operators()
        self._compute_hyperfine_tensors()
    
    def _build_operators(self):
        """C13 spin operators in multi-nucleus space"""
        # Single C13 operators (spin-1/2)
        sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
        sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
        identity = np.eye(2, dtype=complex)
        
        single_ops = {
            'Ix': sigma_x / 2,
            'Iy': sigma_y / 2,
            'Iz': sigma_z / 2,
            'I+': sigma_plus,
            'I-': sigma_minus,
            'I': identity
        }
        
        # Multi-nucleus operators via tensor products
        self.operators = {}
        
        for i in range(self.n_c13):
            self.operators[i] = {}
            
            for op_name, single_op in single_ops.items():
                tensor_op = self._tensor_product_operator(single_op, i)
                self.operators[i][op_name] = tensor_op
    
    def _tensor_product_operator(self, operator: np.ndarray, target_index: int) -> np.ndarray:
        """Build I ⊗ I ⊗ ... ⊗ op_i ⊗ ... ⊗ I"""
        operators = []
        
        for i in range(self.n_c13):
            if i == target_index:
                operators.append(operator)
            else:
                operators.append(np.eye(2, dtype=complex))
        
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
        
        return result
    
    def _compute_hyperfine_tensors(self):
        """Hyperfine coupling tensors from dipolar interaction"""
        self.hyperfine_tensors = {}
        
        for i, pos in enumerate(self.c13_positions):
            A_par, A_perp = self._dipolar_coupling(pos)
            self.hyperfine_tensors[i] = (A_par, A_perp)
    
    def _dipolar_coupling(self, position: np.ndarray) -> Tuple[float, float]:
        """Dipolar coupling strength from first principles"""
        r_vec = position - self.nv_position
        r = np.linalg.norm(r_vec)
        
        if r < 1e-12:
            return 0.0, 0.0
        
        r_hat = r_vec / r
        nv_axis = np.array([0, 0, 1])
        cos_theta = np.dot(r_hat, nv_axis)
        
        # Dipolar coupling: μ₀γₑγₙℏ/(4πr³)
        prefactor = (self.mu_0 * self.gamma_e * self.gamma_n * self.hbar) / (4 * np.pi * r**3)
        prefactor_hz = prefactor / (2 * np.pi)
        
        # Angular dependence
        A_par = prefactor_hz * (3 * cos_theta**2 - 1)
        A_perp = prefactor_hz * 3 * np.sin(np.arccos(cos_theta))**2 / 2
        
        return A_par, A_perp
    
    def nv_c13_hamiltonian(self, nv_operators: Dict[str, np.ndarray]) -> np.ndarray:
        """NV-C13 hyperfine coupling Hamiltonian"""
        nv_dim = 3
        c13_dim = 2**self.n_c13 if self.n_c13 > 0 else 1
        joint_dim = nv_dim * c13_dim
        
        H_hf = np.zeros((joint_dim, joint_dim), dtype=complex)
        
        for i in range(self.n_c13):
            A_par, A_perp = self.hyperfine_tensors[i]
            
            # Joint space operators
            Sz_joint = np.kron(nv_operators['Sz'], np.eye(c13_dim))
            S_plus_joint = np.kron(nv_operators['S+'], np.eye(c13_dim))
            S_minus_joint = np.kron(nv_operators['S-'], np.eye(c13_dim))
            
            Iz_joint = np.kron(np.eye(nv_dim), self.operators[i]['Iz'])
            I_plus_joint = np.kron(np.eye(nv_dim), self.operators[i]['I+'])
            I_minus_joint = np.kron(np.eye(nv_dim), self.operators[i]['I-'])
            
            # Hyperfine Hamiltonian: A∥SzIz + A⊥(S⁺I⁻ + S⁻I⁺)
            H_hf += 2 * np.pi * A_par * (Sz_joint @ Iz_joint)
            H_hf += np.pi * A_perp * (S_plus_joint @ I_minus_joint + S_minus_joint @ I_plus_joint)
        
        return jnp.array(H_hf)
    
    def c13_dipolar_hamiltonian(self) -> np.ndarray:
        """C13-C13 dipole-dipole coupling Hamiltonian"""
        if self.n_c13 < 2:
            return jnp.zeros((2**self.n_c13, 2**self.n_c13), dtype=complex)
        
        c13_dim = 2**self.n_c13
        H_dipolar = np.zeros((c13_dim, c13_dim), dtype=complex)
        
        for i in range(self.n_c13):
            for j in range(i+1, self.n_c13):
                r_ij = self.c13_positions[i] - self.c13_positions[j]
                r = np.linalg.norm(r_ij)
                
                if r < 1e-12:
                    continue
                
                r_hat = r_ij / r
                
                # Dipole-dipole coupling: μ₀γ²ℏ/(4πr³)
                coupling = (self.mu_0 * self.gamma_n**2 * self.hbar) / (4*np.pi * r**3)
                coupling_hz = coupling / (2 * np.pi)
                
                # Operators
                Ix_i, Iy_i, Iz_i = (self.operators[i]['Ix'], self.operators[i]['Iy'], self.operators[i]['Iz'])
                Ix_j, Iy_j, Iz_j = (self.operators[j]['Ix'], self.operators[j]['Iy'], self.operators[j]['Iz'])
                
                # Dipolar Hamiltonian: [IᵢIⱼ - 3(Iᵢ·r̂)(Iⱼ·r̂)]
                I_dot_I = Ix_i @ Ix_j + Iy_i @ Iy_j + Iz_i @ Iz_j
                I_r_I_r = (Ix_i*r_hat[0] + Iy_i*r_hat[1] + Iz_i*r_hat[2]) @ \
                          (Ix_j*r_hat[0] + Iy_j*r_hat[1] + Iz_j*r_hat[2])
                
                H_dipolar += coupling_hz * (I_dot_I - 3*I_r_I_r)
        
        return jnp.array(H_dipolar)
    
    def nuclear_zeeman_hamiltonian(self, B_field: np.ndarray) -> np.ndarray:
        """C13 nuclear Zeeman interaction"""
        c13_dim = 2**self.n_c13
        H_zeeman = np.zeros((c13_dim, c13_dim), dtype=complex)
        
        for i in range(self.n_c13):
            # γₙ B⃗ · I⃗
            Bx, By, Bz = B_field
            H_i = -2 * np.pi * self.gamma_n * (
                Bx * self.operators[i]['Ix'] +
                By * self.operators[i]['Iy'] + 
                Bz * self.operators[i]['Iz']
            )
            H_zeeman += H_i
        
        return jnp.array(H_zeeman)
    
    def complete_c13_hamiltonian(self, nv_operators: Dict[str, np.ndarray], 
                               B_field: np.ndarray) -> np.ndarray:
        """Complete C13 Hamiltonian"""
        H_nv_c13 = self.nv_c13_hamiltonian(nv_operators)
        
        if self.n_c13 > 1:
            H_c13_dipolar = self.c13_dipolar_hamiltonian()
            # Embed C13-only terms in joint space
            nv_dim = 3
            H_c13_joint = jnp.kron(jnp.eye(nv_dim), H_c13_dipolar)
            H_total = H_nv_c13 + H_c13_joint
        else:
            H_total = H_nv_c13
        
        # Add nuclear Zeeman (embed in joint space)
        if jnp.linalg.norm(B_field) > 0:
            H_zeeman_c13 = self.nuclear_zeeman_hamiltonian(B_field)
            nv_dim = 3
            H_zeeman_joint = jnp.kron(jnp.eye(nv_dim), H_zeeman_c13)
            H_total += H_zeeman_joint
        
        return H_total
    
    def validate_operators(self) -> bool:
        """Validate C13 operator commutation relations"""
        for i in range(self.n_c13):
            Ix = self.operators[i]['Ix']
            Iy = self.operators[i]['Iy'] 
            Iz = self.operators[i]['Iz']
            
            # [Ix, Iy] = i Iz for I=1/2
            commutator = Ix @ Iy - Iy @ Ix
            expected = 1j * Iz
            
            if not np.allclose(commutator, expected, atol=1e-12):
                return False
            
            # I² = 3/4 for spin-1/2
            I_squared = Ix @ Ix + Iy @ Iy + Iz @ Iz
            expected_magnitude = 0.75 * np.eye(2**self.n_c13, dtype=complex)
            
            if not np.allclose(I_squared, expected_magnitude, atol=1e-12):
                return False
        
        return True
    
    def get_info(self) -> Dict[str, Any]:
        """C13 system information"""
        if self.hyperfine_tensors:
            A_pars = [A_par for A_par, A_perp in self.hyperfine_tensors.values()]
            A_perps = [A_perp for A_par, A_perp in self.hyperfine_tensors.values()]
            
            hyperfine_stats = {
                'A_par_mean_Hz': np.mean(A_pars),
                'A_par_std_Hz': np.std(A_pars),
                'A_par_range_Hz': [np.min(A_pars), np.max(A_pars)],
                'A_perp_mean_Hz': np.mean(A_perps),
                'A_perp_std_Hz': np.std(A_perps),
                'A_perp_range_Hz': [np.min(A_perps), np.max(A_perps)]
            }
        else:
            hyperfine_stats = {}
        
        return {
            'n_c13': self.n_c13,
            'positions_nm': self.c13_positions * 1e9,
            'hilbert_dimension': 2**self.n_c13,
            'hyperfine_statistics': hyperfine_stats,
            'operators_validated': self.validate_operators()
        }