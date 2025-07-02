"""
Quantum Operators for C13 Nuclear Spins

Provides exact quantum mechanical operators for ¹³C nuclear spins (I=½)
and composite NV-C13 systems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.sparse import csr_matrix, kron, eye
import itertools


class C13QuantumOperators:
    """
    Quantum mechanical operators for C13 nuclear spins and NV-C13 systems
    
    Handles:
    - Single C13 spin-½ operators
    - Multi-C13 tensor products
    - NV-C13 composite systems
    - Sparse matrix optimization
    """
    
    def __init__(self, n_c13: int = 1, use_sparse: bool = True):
        """
        Initialize C13 quantum operators
        
        Args:
            n_c13: Number of C13 nuclei
            use_sparse: Use sparse matrices for efficiency
        """
        self.n_c13 = n_c13
        self.use_sparse = use_sparse
        
        # Single spin-½ operators (Pauli matrices)
        self._sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
        self._sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self._sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
        self._sigma_plus = np.array([[0, 1], [0, 0]], dtype=complex)
        self._sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
        self._identity_2 = np.eye(2, dtype=complex)
        
        # NV spin-1 operators
        self._setup_nv_operators()
        
        # Cache for multi-spin operators
        self._operator_cache = {}
        
        # Generate multi-C13 operators
        self._generate_c13_operators()
        
    def _setup_nv_operators(self):
        """Setup NV spin-1 operators"""
        # S+ and S- operators for spin-1
        S_plus = np.array([
            [0, np.sqrt(2), 0],
            [0, 0, np.sqrt(2)],
            [0, 0, 0]
        ], dtype=complex)
        
        S_minus = np.array([
            [0, 0, 0],
            [np.sqrt(2), 0, 0],
            [0, np.sqrt(2), 0]
        ], dtype=complex)
        
        # Cartesian operators
        self.nv_operators = {
            'Sx': (S_plus + S_minus) / 2,
            'Sy': (S_plus - S_minus) / (2j),
            'Sz': np.diag([-1., 0., 1.]).astype(complex),
            'S+': S_plus,
            'S-': S_minus,
            'I_nv': np.eye(3, dtype=complex)
        }
        
    def _generate_c13_operators(self):
        """Generate operators for multi-C13 system"""
        if self.n_c13 == 0:
            self.c13_operators = {}
            return
            
        # Single C13 operators in terms of Pauli matrices
        single_ops = {
            'Ix': self._sigma_x / 2,
            'Iy': self._sigma_y / 2,
            'Iz': self._sigma_z / 2,
            'I+': self._sigma_plus,
            'I-': self._sigma_minus,
            'I_c13': self._identity_2
        }
        
        # Multi-C13 operators using tensor products
        self.c13_operators = {}
        
        for i in range(self.n_c13):
            self.c13_operators[i] = {}
            
            for op_name, single_op in single_ops.items():
                # Build tensor product: I ⊗ I ⊗ ... ⊗ op_i ⊗ ... ⊗ I
                tensor_op = self._build_tensor_operator(single_op, i)
                self.c13_operators[i][op_name] = tensor_op
                
    def _build_tensor_operator(self, operator: np.ndarray, target_index: int) -> np.ndarray:
        """
        Build tensor product operator acting on specific C13
        
        Args:
            operator: Single-spin operator
            target_index: Index of target C13 spin
            
        Returns:
            Operator in full multi-C13 Hilbert space
        """
        cache_key = (tuple(operator.flatten()), target_index, self.n_c13)
        
        if cache_key in self._operator_cache:
            return self._operator_cache[cache_key]
            
        # Build tensor product
        if self.use_sparse and self.n_c13 > 3:
            # Use sparse matrices for large systems
            result = self._build_sparse_tensor_operator(operator, target_index)
        else:
            # Use dense matrices for small systems
            result = self._build_dense_tensor_operator(operator, target_index)
            
        self._operator_cache[cache_key] = result
        return result
        
    def _build_dense_tensor_operator(self, operator: np.ndarray, target_index: int) -> np.ndarray:
        """Build dense tensor product operator"""
        operators = []
        
        for i in range(self.n_c13):
            if i == target_index:
                operators.append(operator)
            else:
                operators.append(self._identity_2)
                
        # Compute tensor product
        result = operators[0]
        for op in operators[1:]:
            result = np.kron(result, op)
            
        return result
        
    def _build_sparse_tensor_operator(self, operator: np.ndarray, target_index: int) -> csr_matrix:
        """Build sparse tensor product operator"""
        operators = []
        
        for i in range(self.n_c13):
            if i == target_index:
                operators.append(csr_matrix(operator))
            else:
                operators.append(eye(2, format='csr'))
                
        # Compute sparse tensor product
        result = operators[0]
        for op in operators[1:]:
            result = kron(result, op, format='csr')
            
        return result
        
    def get_collective_operators(self) -> Dict[str, np.ndarray]:
        """
        Get collective C13 operators
        
        Returns:
            Dictionary with collective operators (Ix_total, Iy_total, Iz_total)
        """
        if self.n_c13 == 0:
            return {}
            
        collective_ops = {
            'Ix_total': self._sum_individual_operators('Ix'),
            'Iy_total': self._sum_individual_operators('Iy'),
            'Iz_total': self._sum_individual_operators('Iz'),
            'I+_total': self._sum_individual_operators('I+'),
            'I-_total': self._sum_individual_operators('I-')
        }
        
        return collective_ops
        
    def _sum_individual_operators(self, op_name: str) -> np.ndarray:
        """Sum operator over all C13 spins"""
        if self.n_c13 == 0:
            return np.array([[0]])
            
        total_op = self.c13_operators[0][op_name]
        
        for i in range(1, self.n_c13):
            total_op = total_op + self.c13_operators[i][op_name]
            
        return total_op
        
    def get_nv_c13_joint_operators(self) -> Dict[str, np.ndarray]:
        """
        Get operators for joint NV-C13 Hilbert space
        
        Returns:
            Dictionary with joint operators in (3 × 2^N) space
        """
        if self.n_c13 == 0:
            return self.nv_operators.copy()
            
        joint_ops = {}
        c13_dim = 2**self.n_c13
        
        # NV operators in joint space: NV ⊗ I_c13
        c13_identity = self._get_c13_identity()
        
        for op_name, nv_op in self.nv_operators.items():
            if self.use_sparse and self.n_c13 > 8:
                joint_ops[op_name] = kron(csr_matrix(nv_op), c13_identity, format='csr')
            else:
                joint_ops[op_name] = np.kron(nv_op, c13_identity)
                
        # C13 operators in joint space: I_nv ⊗ C13
        nv_identity = self.nv_operators['I_nv']
        
        for i in range(self.n_c13):
            joint_ops[f'C13_{i}'] = {}
            for op_name, c13_op in self.c13_operators[i].items():
                if self.use_sparse and self.n_c13 > 8:
                    joint_ops[f'C13_{i}'][op_name] = kron(csr_matrix(nv_identity), c13_op, format='csr')
                else:
                    joint_ops[f'C13_{i}'][op_name] = np.kron(nv_identity, c13_op)
                    
        return joint_ops
        
    def _get_c13_identity(self) -> np.ndarray:
        """Get identity operator in C13 Hilbert space"""
        if self.n_c13 == 0:
            return np.array([[1]])
            
        if self.use_sparse and self.n_c13 > 8:
            return eye(2**self.n_c13, format='csr')
        else:
            return np.eye(2**self.n_c13, dtype=complex)
            
    def get_hilbert_space_dimensions(self) -> Tuple[int, int, int]:
        """
        Get Hilbert space dimensions
        
        Returns:
            (nv_dim, c13_dim, joint_dim) tuple
        """
        nv_dim = 3
        c13_dim = 2**self.n_c13 if self.n_c13 > 0 else 1
        joint_dim = nv_dim * c13_dim
        
        return nv_dim, c13_dim, joint_dim
        
    def get_computational_basis_states(self) -> Dict[str, np.ndarray]:
        """
        Get computational basis states
        
        Returns:
            Dictionary with basis state vectors
        """
        basis_states = {}
        
        # NV basis states
        nv_basis = {
            'ms_minus1': np.array([1, 0, 0], dtype=complex),
            'ms_0': np.array([0, 1, 0], dtype=complex),
            'ms_plus1': np.array([0, 0, 1], dtype=complex)
        }
        
        # C13 basis states
        if self.n_c13 > 0:
            c13_basis = {}
            for i in range(2**self.n_c13):
                # Convert to binary representation
                binary = format(i, f'0{self.n_c13}b')
                state_name = f'c13_{"".join(binary)}'
                
                state_vector = np.zeros(2**self.n_c13, dtype=complex)
                state_vector[i] = 1.0
                c13_basis[state_name] = state_vector
                
            basis_states['c13'] = c13_basis
            
        # Joint basis states
        if self.n_c13 > 0:
            joint_basis = {}
            for nv_state_name, nv_state in nv_basis.items():
                for c13_state_name, c13_state in c13_basis.items():
                    joint_state_name = f'{nv_state_name}_{c13_state_name}'
                    joint_state = np.kron(nv_state, c13_state)
                    joint_basis[joint_state_name] = joint_state
                    
            basis_states['joint'] = joint_basis
        else:
            basis_states['joint'] = nv_basis
            
        basis_states['nv'] = nv_basis
        
        return basis_states
        
    def thermal_state(self, temperature: float, hamiltonian: np.ndarray) -> np.ndarray:
        """
        Generate thermal equilibrium state
        
        Args:
            temperature: Temperature [K]
            hamiltonian: System Hamiltonian
            
        Returns:
            Thermal density matrix
        """
        from helper.noise_sources import SYSTEM
        
        kb = SYSTEM.get_constant('fundamental', 'kb')
        hbar = SYSTEM.get_constant('fundamental', 'hbar')
        
        # Boltzmann factor
        beta = 1 / (kb * temperature)
        
        # Compute thermal state: ρ = exp(-βH) / Tr(exp(-βH))
        H_scaled = beta * hamiltonian
        
        # Use eigendecomposition for numerical stability
        eigenvals, eigenvecs = np.linalg.eigh(H_scaled)
        
        # Compute exp(-βH) in eigenstate basis
        exp_beta_H = eigenvecs @ np.diag(np.exp(-eigenvals)) @ eigenvecs.conj().T
        
        # Normalize
        partition_function = np.trace(exp_beta_H)
        thermal_rho = exp_beta_H / partition_function
        
        return thermal_rho
        
    def measure_expectation_value(self, operator: np.ndarray, 
                                state: np.ndarray) -> float:
        """
        Measure expectation value of operator
        
        Args:
            operator: Observable operator
            state: Quantum state (vector or density matrix)
            
        Returns:
            Expectation value
        """
        if state.ndim == 1:
            # State vector
            expectation = np.real(np.conj(state) @ operator @ state)
        else:
            # Density matrix
            expectation = np.real(np.trace(operator @ state))
            
        return expectation
        
    def validate_operators(self) -> Dict[str, bool]:
        """
        Validate quantum mechanical properties of operators
        
        Returns:
            Dictionary with validation results
        """
        validation = {}
        
        # Check Hermiticity of observables
        observables = ['Ix', 'Iy', 'Iz']
        
        for i in range(self.n_c13):
            for obs in observables:
                op = self.c13_operators[i][obs]
                # Convert sparse to dense for validation
                if hasattr(op, 'toarray'):
                    op = op.toarray()
                is_hermitian = np.allclose(op, op.conj().T, atol=1e-12)
                validation[f'C13_{i}_{obs}_hermitian'] = is_hermitian
                
        # Check commutation relations [Ix, Iy] = iIz
        for i in range(self.n_c13):
            Ix = self.c13_operators[i]['Ix']
            Iy = self.c13_operators[i]['Iy']
            Iz = self.c13_operators[i]['Iz']
            
            # Convert sparse to dense for calculations
            if hasattr(Ix, 'toarray'):
                Ix = Ix.toarray()
                Iy = Iy.toarray()
                Iz = Iz.toarray()
            
            commutator = Ix @ Iy - Iy @ Ix
            expected = 1j * Iz
            
            commutation_correct = np.allclose(commutator, expected, atol=1e-12)
            validation[f'C13_{i}_commutation_relations'] = commutation_correct
            
        # Check spin magnitude: I² = ¾ for spin-½
        for i in range(self.n_c13):
            Ix = self.c13_operators[i]['Ix']
            Iy = self.c13_operators[i]['Iy']
            Iz = self.c13_operators[i]['Iz']
            
            # Convert sparse to dense
            if hasattr(Ix, 'toarray'):
                Ix = Ix.toarray()
                Iy = Iy.toarray()
                Iz = Iz.toarray()
            
            I_squared = Ix @ Ix + Iy @ Iy + Iz @ Iz
            expected_magnitude = 0.75 * self._get_c13_identity()
            
            # Convert expected magnitude to dense if needed
            if hasattr(expected_magnitude, 'toarray'):
                expected_magnitude = expected_magnitude.toarray()
            
            magnitude_correct = np.allclose(I_squared, expected_magnitude, atol=1e-12)
            validation[f'C13_{i}_spin_magnitude'] = magnitude_correct
            
        return validation
        
    def optimize_memory_usage(self):
        """Optimize memory usage for large systems"""
        if self.n_c13 > 10:
            # Clear operator cache to save memory
            self._operator_cache.clear()
            
            # Use sparse matrices
            self.use_sparse = True
            
            # Regenerate operators with sparse representation
            self._generate_c13_operators()
            
    def get_memory_usage_estimate(self) -> Dict[str, float]:
        """
        Estimate memory usage for different system sizes
        
        Returns:
            Dictionary with memory estimates [MB]
        """
        estimates = {}
        
        # Dense matrix storage
        nv_dim, c13_dim, joint_dim = self.get_hilbert_space_dimensions()
        
        # Complex numbers are 16 bytes
        bytes_per_complex = 16
        
        # Single C13 operators
        single_c13_memory = len(self.c13_operators) * 6 * c13_dim**2 * bytes_per_complex
        estimates['c13_operators_MB'] = single_c13_memory / (1024**2)
        
        # Joint operators
        joint_memory = 10 * joint_dim**2 * bytes_per_complex  # Estimate 10 joint operators
        estimates['joint_operators_MB'] = joint_memory / (1024**2)
        
        # State vectors
        state_memory = joint_dim * bytes_per_complex
        estimates['state_vector_MB'] = state_memory / (1024**2)
        
        # Density matrices
        density_memory = joint_dim**2 * bytes_per_complex
        estimates['density_matrix_MB'] = density_memory / (1024**2)
        
        # Total estimate
        total_memory = single_c13_memory + joint_memory + density_memory
        estimates['total_estimate_MB'] = total_memory / (1024**2)
        
        return estimates