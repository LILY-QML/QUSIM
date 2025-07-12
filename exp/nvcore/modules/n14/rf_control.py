"""
N14 RF Control Engine - Complete Nuclear Spin Manipulation

EXACT treatment of RF-driven nuclear spin control including:
- Resonant RF pulses (π, π/2, arbitrary angle)
- Composite pulse sequences (CORPSE, BB1, etc.)
- Adiabatic passages
- Rabi oscillations and coherent control
- Real-time pulse optimization
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from .base import N14PhysicsEngine, FallbackViolationError
from .quantum_operators import N14QuantumOperators
from .nuclear_zeeman import N14NuclearZeemanEngine

class N14RFControlEngine(N14PhysicsEngine):
    """
    Complete RF control system for N14 nuclear spins
    
    RF Hamiltonian (rotating wave approximation):
    H_RF(t) = Ω(t) [cos(ωt + φ) Ix + sin(ωt + φ) Iy]
    
    Where:
    - Ω(t): Time-dependent Rabi frequency
    - ω: RF frequency
    - φ: RF phase
    - Ix, Iy: Nuclear spin operators
    
    Includes complete pulse library and optimization algorithms.
    """
    
    def __init__(self):
        super().__init__()
        
        # RF system parameters
        self._max_rabi_frequency = 1e6  # Hz (1 MHz max)
        self._frequency_precision = 1.0  # Hz
        self._phase_precision = 1e-3  # radians
        
        # Get nuclear operators and Zeeman engine
        self._nuclear_ops = N14QuantumOperators()
        self._zeeman_engine = N14NuclearZeemanEngine()
        
        # Pulse library
        self._pulse_library = self._initialize_pulse_library()
        
        print(f"✅ N14 RF Control Engine initialized:")
        print(f"   Max Rabi frequency: {self._max_rabi_frequency/1e6:.1f} MHz")
        print(f"   Frequency precision: {self._frequency_precision:.1f} Hz")
        print(f"   Available pulses: {list(self._pulse_library.keys())}")
    
    def calculate_physics(self, pulse_sequence: List[Dict], 
                         magnetic_field: np.ndarray,
                         initial_state: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Calculate complete RF pulse sequence evolution
        
        Args:
            pulse_sequence: List of pulse dictionaries with parameters
            magnetic_field: Applied magnetic field [T]
            initial_state: Initial nuclear state (default: thermal)
            
        Returns:
            Complete evolution results and final state
        """
        
        if len(pulse_sequence) == 0:
            raise ValueError("Pulse sequence cannot be empty")
        
        # Initialize state
        if initial_state is None:
            initial_state = self._get_thermal_initial_state(magnetic_field)
        
        # Calculate Zeeman Hamiltonian (static background)
        zeeman_result = self._zeeman_engine.calculate_physics(magnetic_field)
        H_zeeman = zeeman_result['hamiltonian']
        
        # Evolve through pulse sequence
        evolution_results = self._evolve_pulse_sequence(
            pulse_sequence, H_zeeman, initial_state
        )
        
        # Validate final state
        final_state = evolution_results['final_state']
        self._validate_quantum_state(final_state)
        
        return {
            'evolution_results': evolution_results,
            'final_state': final_state,
            'fidelity': self._calculate_sequence_fidelity(evolution_results),
            'pulse_validation': self._validate_pulse_sequence(pulse_sequence),
            'zeeman_background': zeeman_result
        }
    
    def _initialize_pulse_library(self) -> Dict[str, Dict]:
        """Initialize library of standard RF pulses"""
        
        return {
            'pi_pulse': {
                'flip_angle': np.pi,
                'phase': 0.0,
                'pulse_type': 'rectangular',
                'description': 'π pulse for complete population inversion'
            },
            'pi_half_pulse': {
                'flip_angle': np.pi/2,
                'phase': 0.0,
                'pulse_type': 'rectangular',
                'description': 'π/2 pulse for superposition creation'
            },
            'pi_pulse_y': {
                'flip_angle': np.pi,
                'phase': np.pi/2,
                'pulse_type': 'rectangular',
                'description': 'π pulse around y-axis'
            },
            'ramsey_sequence': {
                'pulses': ['pi_half_pulse', 'delay', 'pi_half_pulse'],
                'description': 'Ramsey interferometry sequence'
            },
            'spin_echo': {
                'pulses': ['pi_half_pulse', 'delay', 'pi_pulse', 'delay', 'pi_half_pulse'],
                'description': 'Hahn echo sequence for T2 measurement'
            },
            'corpse_pulse': {
                'angles': [np.pi + np.pi/2, 2*np.pi + np.pi, np.pi + np.pi/2],
                'phases': [0, np.pi/2, 0],
                'description': 'CORPSE composite pulse (robust against amplitude errors)'
            },
            'bb1_pulse': {
                'angles': [np.pi, 2*np.pi, np.pi],
                'phases': [0, np.pi/2, 0],
                'description': 'BB1 composite pulse (broadband performance)'
            }
        }
    
    def _get_thermal_initial_state(self, magnetic_field: np.ndarray) -> np.ndarray:
        """Get thermal equilibrium initial state"""
        
        zeeman_result = self._zeeman_engine.calculate_physics(magnetic_field)
        thermal_populations = zeeman_result['thermal_populations']
        
        # Convert populations to density matrix diagonal
        # For pure state approximation, use ground state
        if np.linalg.norm(magnetic_field) > 1e-6:
            # High field: polarized along field
            return zeeman_result['eigenvectors'][:, 0]  # Lowest energy state
        else:
            # Zero field: equal superposition
            return np.array([1, 1, 1]) / np.sqrt(3)
    
    def _evolve_pulse_sequence(self, pulse_sequence: List[Dict], 
                             H_zeeman: np.ndarray, 
                             initial_state: np.ndarray) -> Dict[str, List]:
        """Evolve state through complete pulse sequence"""
        
        current_state = initial_state.copy()
        states_evolution = [current_state.copy()]
        pulse_operators = []
        times = [0.0]
        current_time = 0.0
        
        for i, pulse_params in enumerate(pulse_sequence):
            
            # Generate pulse operator
            U_pulse = self._generate_pulse_operator(pulse_params, H_zeeman)
            pulse_operators.append(U_pulse)
            
            # Apply pulse
            current_state = U_pulse @ current_state
            
            # Normalize state (account for numerical errors)
            norm = np.linalg.norm(current_state)
            if norm < 1e-12:
                raise FallbackViolationError(
                    f"State collapsed to zero after pulse {i}!\n"
                    f"Pulse parameters: {pulse_params}\n"
                    f"Check pulse validity and Hamiltonian construction."
                )
            
            current_state = current_state / norm
            
            # Store evolution
            states_evolution.append(current_state.copy())
            
            # Update time
            pulse_duration = pulse_params.get('duration', 1e-6)  # Default 1 μs
            current_time += pulse_duration
            times.append(current_time)
        
        return {
            'states': states_evolution,
            'pulse_operators': pulse_operators,
            'times': times,
            'final_state': current_state
        }
    
    def _generate_pulse_operator(self, pulse_params: Dict, H_zeeman: np.ndarray) -> np.ndarray:
        """Generate unitary operator for single RF pulse"""
        
        pulse_type = pulse_params.get('type', 'rectangular')
        
        if pulse_type == 'rectangular':
            return self._generate_rectangular_pulse(pulse_params, H_zeeman)
        elif pulse_type == 'gaussian':
            return self._generate_gaussian_pulse(pulse_params, H_zeeman)
        elif pulse_type == 'adiabatic':
            return self._generate_adiabatic_pulse(pulse_params, H_zeeman)
        elif pulse_type == 'composite':
            return self._generate_composite_pulse(pulse_params, H_zeeman)
        else:
            raise ValueError(f"Unknown pulse type: {pulse_type}")
    
    def _generate_rectangular_pulse(self, pulse_params: Dict, H_zeeman: np.ndarray) -> np.ndarray:
        """Generate rectangular RF pulse operator"""
        
        # Extract parameters
        rabi_frequency = pulse_params.get('rabi_frequency', 1e5)  # Hz
        duration = pulse_params.get('duration', 1e-6)  # s
        phase = pulse_params.get('phase', 0.0)  # rad
        frequency = pulse_params.get('frequency', 0.0)  # Hz (detuning)
        
        # Validate parameters
        if rabi_frequency > self._max_rabi_frequency:
            raise ValueError(f"Rabi frequency {rabi_frequency/1e6:.1f} MHz exceeds maximum")
        
        if duration <= 0:
            raise ValueError("Pulse duration must be positive")
        
        # Get nuclear operators
        nuclear_ops = self._nuclear_ops.get_all_operators()
        Ix, Iy, Iz = nuclear_ops['Ix'], nuclear_ops['Iy'], nuclear_ops['Iz']
        
        # RF Hamiltonian in rotating frame (RWA)
        # H_RF = Ω [cos(φ) Ix + sin(φ) Iy] + Δω Iz
        H_rf = 2 * np.pi * (
            rabi_frequency * (np.cos(phase) * Ix + np.sin(phase) * Iy) +
            frequency * Iz
        )
        
        # Total Hamiltonian during pulse
        H_total = H_zeeman + H_rf
        
        # Pulse evolution operator
        U_pulse = self._matrix_exponential(-1j * H_total * duration)
        
        # Validate unitarity
        self._validate_unitarity(U_pulse, f"rectangular pulse")
        
        return U_pulse
    
    def _generate_gaussian_pulse(self, pulse_params: Dict, H_zeeman: np.ndarray) -> np.ndarray:
        """Generate Gaussian-shaped RF pulse"""
        
        rabi_peak = pulse_params.get('rabi_frequency', 1e5)  # Hz
        duration = pulse_params.get('duration', 1e-6)  # s
        phase = pulse_params.get('phase', 0.0)  # rad
        sigma = pulse_params.get('sigma', duration/4)  # Gaussian width
        
        # Time discretization
        n_steps = max(100, int(duration * 1e8))  # 10 ns resolution minimum
        dt = duration / n_steps
        times = np.linspace(0, duration, n_steps)
        
        # Gaussian envelope
        t_center = duration / 2
        envelope = np.exp(-0.5 * ((times - t_center) / sigma)**2)
        rabi_profile = rabi_peak * envelope
        
        # Get operators
        nuclear_ops = self._nuclear_ops.get_all_operators()
        Ix, Iy = nuclear_ops['Ix'], nuclear_ops['Iy']
        
        # Evolve step by step
        U_total = np.eye(3, dtype=complex)
        
        for i, rabi_t in enumerate(rabi_profile):
            H_rf_t = 2 * np.pi * rabi_t * (np.cos(phase) * Ix + np.sin(phase) * Iy)
            H_total_t = H_zeeman + H_rf_t
            
            U_step = self._matrix_exponential(-1j * H_total_t * dt)
            U_total = U_step @ U_total
        
        self._validate_unitarity(U_total, "Gaussian pulse")
        return U_total
    
    def _generate_adiabatic_pulse(self, pulse_params: Dict, H_zeeman: np.ndarray) -> np.ndarray:
        """Generate adiabatic passage pulse (frequency sweep)"""
        
        duration = pulse_params.get('duration', 10e-6)  # s
        freq_start = pulse_params.get('freq_start', -1e6)  # Hz
        freq_end = pulse_params.get('freq_end', 1e6)  # Hz
        rabi_frequency = pulse_params.get('rabi_frequency', 1e5)  # Hz
        
        # Adiabaticity condition check
        sweep_rate = abs(freq_end - freq_start) / duration
        adiabatic_parameter = (rabi_frequency**2) / sweep_rate
        
        if adiabatic_parameter < 10:  # Rule of thumb
            print(f"⚠️  Warning: Adiabatic parameter {adiabatic_parameter:.1f} < 10")
            print("   Pulse may not be sufficiently adiabatic")
        
        # Time evolution with frequency sweep
        n_steps = max(1000, int(duration * 1e7))  # High resolution for adiabatic
        dt = duration / n_steps
        times = np.linspace(0, duration, n_steps)
        
        # Frequency sweep (linear)
        frequencies = freq_start + (freq_end - freq_start) * times / duration
        
        # Get operators
        nuclear_ops = self._nuclear_ops.get_all_operators()
        Ix, Iz = nuclear_ops['Ix'], nuclear_ops['Iz']
        
        # Evolve step by step
        U_total = np.eye(3, dtype=complex)
        
        for freq_t in frequencies:
            H_rf_t = 2 * np.pi * (rabi_frequency * Ix + freq_t * Iz)
            H_total_t = H_zeeman + H_rf_t
            
            U_step = self._matrix_exponential(-1j * H_total_t * dt)
            U_total = U_step @ U_total
        
        self._validate_unitarity(U_total, "adiabatic pulse")
        return U_total
    
    def _generate_composite_pulse(self, pulse_params: Dict, H_zeeman: np.ndarray) -> np.ndarray:
        """Generate composite pulse sequence"""
        
        composite_type = pulse_params.get('composite_type', 'corpse')
        
        if composite_type == 'corpse':
            return self._generate_corpse_pulse(pulse_params, H_zeeman)
        elif composite_type == 'bb1':
            return self._generate_bb1_pulse(pulse_params, H_zeeman)
        else:
            raise ValueError(f"Unknown composite type: {composite_type}")
    
    def _generate_corpse_pulse(self, pulse_params: Dict, H_zeeman: np.ndarray) -> np.ndarray:
        """Generate CORPSE (COmpensation for Resonance offsEt and Selective Excitation) pulse"""
        
        rabi_frequency = pulse_params.get('rabi_frequency', 1e5)  # Hz
        
        # CORPSE sequence: 410°₀ - 300°₉₀ - 410°₀
        angles = [410 * np.pi/180, 300 * np.pi/180, 410 * np.pi/180]
        phases = [0, np.pi/2, 0]
        
        U_total = np.eye(3, dtype=complex)
        
        for angle, phase in zip(angles, phases):
            duration = angle / (2 * np.pi * rabi_frequency)
            
            single_pulse = {
                'rabi_frequency': rabi_frequency,
                'duration': duration,
                'phase': phase,
                'frequency': 0.0
            }
            
            U_pulse = self._generate_rectangular_pulse(single_pulse, H_zeeman)
            U_total = U_pulse @ U_total
        
        return U_total
    
    def _generate_bb1_pulse(self, pulse_params: Dict, H_zeeman: np.ndarray) -> np.ndarray:
        """Generate BB1 (BroadBand) composite pulse"""
        
        rabi_frequency = pulse_params.get('rabi_frequency', 1e5)  # Hz
        
        # BB1 sequence: π₀ - 2π₉₀ - π₀
        angles = [np.pi, 2*np.pi, np.pi]
        phases = [0, np.pi/2, 0]
        
        U_total = np.eye(3, dtype=complex)
        
        for angle, phase in zip(angles, phases):
            duration = angle / (2 * np.pi * rabi_frequency)
            
            single_pulse = {
                'rabi_frequency': rabi_frequency,
                'duration': duration,
                'phase': phase,
                'frequency': 0.0
            }
            
            U_pulse = self._generate_rectangular_pulse(single_pulse, H_zeeman)
            U_total = U_pulse @ U_total
        
        return U_total
    
    def _matrix_exponential(self, matrix: np.ndarray) -> np.ndarray:
        """Compute matrix exponential with high precision"""
        
        # Use scipy's implementation for numerical stability
        from scipy.linalg import expm
        return expm(matrix)
    
    def _validate_unitarity(self, operator: np.ndarray, context: str = ""):
        """Validate that operator is unitary"""
        
        # Check U†U = I
        identity_check = operator.conj().T @ operator
        identity_deviation = np.max(np.abs(identity_check - np.eye(operator.shape[0])))
        
        if identity_deviation > 1e-12:
            raise FallbackViolationError(
                f"Operator not unitary in {context}!\n"
                f"Max deviation from U†U = I: {identity_deviation:.2e}\n"
                f"Operator shape: {operator.shape}"
            )
        
        # Check determinant = 1 (up to phase)
        det = np.linalg.det(operator)
        if abs(abs(det) - 1.0) > 1e-12:
            raise FallbackViolationError(
                f"Operator determinant |det| = {abs(det):.10f} ≠ 1 in {context}"
            )
    
    def _validate_quantum_state(self, state: np.ndarray):
        """Validate quantum state normalization"""
        
        norm = np.linalg.norm(state)
        if abs(norm - 1.0) > 1e-12:
            raise FallbackViolationError(
                f"Quantum state not normalized: |ψ| = {norm:.10f} ≠ 1"
            )
    
    def _calculate_sequence_fidelity(self, evolution_results: Dict) -> Dict[str, float]:
        """Calculate fidelity metrics for pulse sequence"""
        
        initial_state = evolution_results['states'][0]
        final_state = evolution_results['final_state']
        
        # Return probability
        return_fidelity = abs(np.conj(initial_state) @ final_state)**2
        
        # Process fidelity (trace fidelity for mixed states)
        # For pure states: F = |⟨ψ_target|ψ_actual⟩|²
        
        return {
            'return_fidelity': return_fidelity,
            'state_overlap': abs(np.conj(initial_state) @ final_state),
            'final_state_purity': abs(np.conj(final_state) @ final_state)
        }
    
    def _validate_pulse_sequence(self, pulse_sequence: List[Dict]) -> Dict[str, bool]:
        """Validate pulse sequence parameters"""
        
        validation = {}
        
        for i, pulse in enumerate(pulse_sequence):
            pulse_key = f"pulse_{i}"
            
            # Check required parameters
            required_params = ['rabi_frequency', 'duration']
            validation[f"{pulse_key}_has_required_params"] = all(
                param in pulse for param in required_params
            )
            
            # Check parameter ranges
            rabi_freq = pulse.get('rabi_frequency', 0)
            duration = pulse.get('duration', 0)
            
            validation[f"{pulse_key}_rabi_in_range"] = 0 < rabi_freq <= self._max_rabi_frequency
            validation[f"{pulse_key}_duration_positive"] = duration > 0
            
            # Check phase range
            phase = pulse.get('phase', 0)
            validation[f"{pulse_key}_phase_in_range"] = -2*np.pi <= phase <= 2*np.pi
        
        return validation
    
    def optimize_pulse_sequence(self, target_operation: str, 
                              constraints: Dict[str, float]) -> List[Dict]:
        """
        Optimize pulse sequence for target operation
        
        Args:
            target_operation: 'pi_pulse', 'pi_half_pulse', 'arbitrary_rotation'
            constraints: Dictionary with optimization constraints
            
        Returns:
            Optimized pulse sequence
        """
        
        if target_operation == 'pi_pulse':
            return self._optimize_pi_pulse(constraints)
        elif target_operation == 'pi_half_pulse':
            return self._optimize_pi_half_pulse(constraints)
        elif target_operation == 'arbitrary_rotation':
            return self._optimize_arbitrary_rotation(constraints)
        else:
            raise ValueError(f"Unknown target operation: {target_operation}")
    
    def _optimize_pi_pulse(self, constraints: Dict[str, float]) -> List[Dict]:
        """Optimize π pulse for given constraints"""
        
        max_rabi = constraints.get('max_rabi_frequency', self._max_rabi_frequency)
        max_duration = constraints.get('max_duration', 10e-6)  # 10 μs
        
        # For π pulse: Ω × t = π
        optimal_rabi = min(max_rabi, np.pi / max_duration)
        optimal_duration = np.pi / optimal_rabi
        
        return [{
            'type': 'rectangular',
            'rabi_frequency': optimal_rabi,
            'duration': optimal_duration,
            'phase': 0.0,
            'frequency': 0.0
        }]
    
    def _optimize_pi_half_pulse(self, constraints: Dict[str, float]) -> List[Dict]:
        """Optimize π/2 pulse for given constraints"""
        
        max_rabi = constraints.get('max_rabi_frequency', self._max_rabi_frequency)
        max_duration = constraints.get('max_duration', 10e-6)  # 10 μs
        
        # For π/2 pulse: Ω × t = π/2
        optimal_rabi = min(max_rabi, (np.pi/2) / max_duration)
        optimal_duration = (np.pi/2) / optimal_rabi
        
        return [{
            'type': 'rectangular',
            'rabi_frequency': optimal_rabi,
            'duration': optimal_duration,
            'phase': 0.0,
            'frequency': 0.0
        }]
    
    def _optimize_arbitrary_rotation(self, constraints: Dict[str, float]) -> List[Dict]:
        """Optimize arbitrary rotation angle"""
        
        angle = constraints.get('rotation_angle', np.pi)
        axis = constraints.get('rotation_axis', 'x')  # 'x', 'y', 'z'
        
        max_rabi = constraints.get('max_rabi_frequency', self._max_rabi_frequency)
        
        # Phase mapping for rotation axes
        phase_map = {'x': 0.0, 'y': np.pi/2, 'z': np.pi}
        phase = phase_map.get(axis, 0.0)
        
        optimal_duration = angle / max_rabi
        
        return [{
            'type': 'rectangular',
            'rabi_frequency': max_rabi,
            'duration': optimal_duration,
            'phase': phase,
            'frequency': 0.0
        }]
    
    def get_pulse_library(self) -> Dict[str, Dict]:
        """Get complete pulse library"""
        return self._pulse_library.copy()
    
    def get_rf_parameters(self) -> Dict[str, float]:
        """Get RF system parameters"""
        return {
            'max_rabi_frequency': self._max_rabi_frequency,
            'frequency_precision': self._frequency_precision,
            'phase_precision': self._phase_precision
        }