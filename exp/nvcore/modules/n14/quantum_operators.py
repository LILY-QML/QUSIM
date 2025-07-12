"""
N14 Quantum Operators - Exact I=1 Nuclear Spin Mathematics

EXACT analytical matrix elements for I=1 nuclear angular momentum.
NO approximations, NO numerical errors - pure analytical quantum mechanics.
"""

import numpy as np
from typing import Dict, Tuple, List
from .base import N14PhysicsEngine, FallbackViolationError

class N14QuantumOperators(N14PhysicsEngine):
    """
    Exact quantum mechanical operators for N14 nuclear spin (I=1)
    
    Provides all operators needed for complete N14 quantum mechanical treatment:
    - Cartesian angular momentum operators (Ix, Iy, Iz)
    - Ladder operators (I+, I-)
    - Quadratic operators for quadrupole interaction
    - Spherical tensor operators for crystal field effects
    
    All matrix elements are EXACT analytical values - NO numerical approximations
    """
    
    def __init__(self):
        super().__init__()
        self._nuclear_spin = 1.0
        self._operators = self._construct_exact_operators()
        self._validate_quantum_mechanics()
    
    def calculate_physics(self) -> Dict[str, np.ndarray]:
        """Return complete set of I=1 operators"""
        return self._operators.copy()
    
    def _construct_exact_operators(self) -> Dict[str, np.ndarray]:
        """Construct exact I=1 operators using analytical matrix elements"""
        
        # I=1 system: |m⟩ = |+1⟩, |0⟩, |-1⟩
        # Basis ordering: [|+1⟩, |0⟩, |-1⟩]
        
        # Cartesian operators - EXACT analytical values
        Ix = np.array([
            [0,     1/np.sqrt(2),  0    ],
            [1/np.sqrt(2), 0,      1/np.sqrt(2)],
            [0,     1/np.sqrt(2),  0    ]
        ], dtype=complex) 
        
        Iy = np.array([
            [0,      -1j/np.sqrt(2),  0     ],
            [1j/np.sqrt(2),  0,       -1j/np.sqrt(2)],
            [0,      1j/np.sqrt(2),   0     ]
        ], dtype=complex)
        
        Iz = np.array([
            [1,  0,  0],
            [0,  0,  0], 
            [0,  0, -1]
        ], dtype=complex)
        
        # Ladder operators
        I_plus = Ix + 1j * Iy
        I_minus = Ix - 1j * Iy
        
        # Total angular momentum squared
        I_squared = Ix @ Ix + Iy @ Iy + Iz @ Iz
        
        # Quadratic operators for quadrupole interaction
        Ix2 = Ix @ Ix
        Iy2 = Iy @ Iy  
        Iz2 = Iz @ Iz
        
        # Quadrupole tensor operators T²ₘ (spherical tensors)
        # T²₀ = (3Iz² - I²)/sqrt(6)
        T20 = (3 * Iz2 - I_squared) / np.sqrt(6)
        
        # T²₊₁ = -(Iz⊗I+ + I+⊗Iz)/sqrt(2)
        T2plus1 = -(Iz @ I_plus + I_plus @ Iz) / np.sqrt(2)
        
        # T²₋₁ = (Iz⊗I- + I-⊗Iz)/sqrt(2)  
        T2minus1 = (Iz @ I_minus + I_minus @ Iz) / np.sqrt(2)
        
        # T²₊₂ = I+⊗I+
        T2plus2 = I_plus @ I_plus
        
        # T²₋₂ = I-⊗I-
        T2minus2 = I_minus @ I_minus
        
        # Anti-commutators for hyperfine coupling
        # {Ix, Iy} = IxIy + IyIx
        anticomm_xy = Ix @ Iy + Iy @ Ix
        anticomm_xz = Ix @ Iz + Iz @ Ix
        anticomm_yz = Iy @ Iz + Iz @ Iy
        
        operators = {
            # Basic angular momentum
            'Ix': Ix,
            'Iy': Iy, 
            'Iz': Iz,
            'I+': I_plus,
            'I-': I_minus,
            'I²': I_squared,
            
            # Quadratic operators
            'Ix²': Ix2,
            'Iy²': Iy2,
            'Iz²': Iz2,
            
            # Spherical tensor operators
            'T²₀': T20,
            'T²₊₁': T2plus1,
            'T²₋₁': T2minus1, 
            'T²₊₂': T2plus2,
            'T²₋₂': T2minus2,
            
            # Anti-commutators
            '{Ix,Iy}': anticomm_xy,
            '{Ix,Iz}': anticomm_xz,
            '{Iy,Iz}': anticomm_yz
        }
        
        return operators
    
    def _validate_quantum_mechanics(self):
        """Validate ALL quantum mechanical commutation relations"""
        
        print("🔍 Validating I=1 quantum mechanics...")
        
        Ix, Iy, Iz = self._operators['Ix'], self._operators['Iy'], self._operators['Iz']
        I_squared = self._operators['I²']
        
        # 1. Commutation relations: [Ix, Iy] = iℏIz (ℏ=1 in natural units)
        comm_xy = Ix @ Iy - Iy @ Ix
        expected_comm_xy = 1j * Iz
        
        if not np.allclose(comm_xy, expected_comm_xy, atol=1e-15):
            raise FallbackViolationError(
                f"QUANTUM MECHANICS VIOLATION: [Ix, Iy] ≠ iIz\n"
                f"Calculated commutator:\n{comm_xy}\n"
                f"Expected:\n{expected_comm_xy}\n"
                f"Max error: {np.max(np.abs(comm_xy - expected_comm_xy)):.2e}"
            )
        
        # 2. [Iy, Iz] = iIx
        comm_yz = Iy @ Iz - Iz @ Iy
        expected_comm_yz = 1j * Ix
        
        if not np.allclose(comm_yz, expected_comm_yz, atol=1e-15):
            raise FallbackViolationError(
                f"QUANTUM MECHANICS VIOLATION: [Iy, Iz] ≠ iIx\n"
                f"Max error: {np.max(np.abs(comm_yz - expected_comm_yz)):.2e}"
            )
        
        # 3. [Iz, Ix] = iIy
        comm_zx = Iz @ Ix - Ix @ Iz
        expected_comm_zx = 1j * Iy
        
        if not np.allclose(comm_zx, expected_comm_zx, atol=1e-15):
            raise FallbackViolationError(
                f"QUANTUM MECHANICS VIOLATION: [Iz, Ix] ≠ iIy\n"
                f"Max error: {np.max(np.abs(comm_zx - expected_comm_zx)):.2e}"
            )
        
        # 4. I² eigenvalue check: I²|m⟩ = I(I+1)|m⟩ = 2|m⟩ for I=1
        expected_i_squared = 2.0 * np.eye(3)
        
        if not np.allclose(I_squared, expected_i_squared, atol=1e-15):
            raise FallbackViolationError(
                f"QUANTUM MECHANICS VIOLATION: I² ≠ 2⋅I for I=1\n"
                f"Calculated I²:\n{I_squared}\n"
                f"Expected: 2⋅I\n{expected_i_squared}"
            )
        
        # 5. Iz eigenvalue check: Iz|m⟩ = m|m⟩
        Iz_eigenvals = np.linalg.eigvals(Iz)
        expected_eigenvals = np.array([1, 0, -1])
        
        if not np.allclose(np.sort(Iz_eigenvals), np.sort(expected_eigenvals), atol=1e-15):
            raise FallbackViolationError(
                f"QUANTUM MECHANICS VIOLATION: Iz eigenvalues incorrect\n"
                f"Calculated: {np.sort(Iz_eigenvals)}\n" 
                f"Expected: {np.sort(expected_eigenvals)}"
            )
        
        # 6. Hermiticity check for angular momentum operators (not ladder operators)
        hermitian_operators = ['Ix', 'Iy', 'Iz', 'Ix²', 'Iy²', 'Iz²', 'I²']
        for name in hermitian_operators:
            op = self._operators[name]
            if not np.allclose(op, op.conj().T, atol=1e-15):
                raise FallbackViolationError(
                    f"QUANTUM MECHANICS VIOLATION: Operator {name} not Hermitian!\n"
                    f"Max deviation: {np.max(np.abs(op - op.conj().T)):.2e}"
                )
        
        # 7. Check ladder operator relationships: I+† = I-
        I_plus = self._operators['I+']
        I_minus = self._operators['I-']
        if not np.allclose(I_plus.conj().T, I_minus, atol=1e-15):
            raise FallbackViolationError(
                f"QUANTUM MECHANICS VIOLATION: I+ and I- not Hermitian conjugates!\n"
                f"Max deviation: {np.max(np.abs(I_plus.conj().T - I_minus)):.2e}"
            )
        
        print("✅ ALL quantum mechanical validations passed!")
    
    def get_operator(self, name: str) -> np.ndarray:
        """Get specific operator by name"""
        if name not in self._operators:
            available = list(self._operators.keys())
            raise ValueError(
                f"Operator '{name}' not available.\n"
                f"Available operators: {available}"
            )
        
        return self._operators[name].copy()
    
    def get_all_operators(self) -> Dict[str, np.ndarray]:
        """Get all operators as dictionary"""
        return self._operators.copy()
    
    def calculate_matrix_element(self, operator_name: str, 
                               initial_state: int, final_state: int) -> complex:
        """
        Calculate matrix element ⟨final|operator|initial⟩
        
        Args:
            operator_name: Name of operator
            initial_state: Initial state m quantum number (+1, 0, or -1)
            final_state: Final state m quantum number (+1, 0, or -1)
            
        Returns:
            Complex matrix element
        """
        
        # Convert m quantum numbers to matrix indices
        m_to_index = {1: 0, 0: 1, -1: 2}
        
        if initial_state not in m_to_index:
            raise ValueError(f"Invalid initial state: {initial_state}. Must be +1, 0, or -1")
        if final_state not in m_to_index:
            raise ValueError(f"Invalid final state: {final_state}. Must be +1, 0, or -1")
        
        operator = self.get_operator(operator_name)
        
        i_idx = m_to_index[initial_state]
        f_idx = m_to_index[final_state]
        
        return operator[f_idx, i_idx]
    
    def validate_selection_rules(self, transition_operator: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        Validate selection rules for given transition operator
        
        Returns dictionary with 'allowed' and 'forbidden' transitions
        """
        
        operator = self.get_operator(transition_operator)
        
        allowed_transitions = []
        forbidden_transitions = []
        
        states = [1, 0, -1]  # m quantum numbers
        
        for initial in states:
            for final in states:
                matrix_element = self.calculate_matrix_element(transition_operator, initial, final)
                
                if abs(matrix_element) > 1e-12:  # Non-zero matrix element
                    allowed_transitions.append((initial, final))
                else:
                    forbidden_transitions.append((initial, final))
        
        return {
            'allowed': allowed_transitions,
            'forbidden': forbidden_transitions
        }
    
    def calculate_expectation_value(self, operator_name: str, state: np.ndarray) -> complex:
        """
        Calculate expectation value ⟨ψ|operator|ψ⟩
        
        Args:
            operator_name: Name of operator
            state: 3-component state vector in |m⟩ basis
            
        Returns:
            Complex expectation value
        """
        
        if state.shape != (3,):
            raise ValueError(f"State must be 3-component vector, got shape {state.shape}")
        
        # Normalize state
        norm = np.linalg.norm(state)
        if norm == 0:
            raise FallbackViolationError("Cannot calculate expectation value for zero state!")
        
        normalized_state = state / norm
        
        operator = self.get_operator(operator_name)
        
        expectation = np.conj(normalized_state) @ operator @ normalized_state
        
        return expectation