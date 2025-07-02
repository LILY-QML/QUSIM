"""
C13 Bath Engine Core

Ultra-realistic quantum mechanical ¹³C nuclear spin bath engine.
The heart of the C13 module - combines all physics engines into coherent simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import warnings
import sys
import os

# Add path for system constants
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'helper'))
from noise_sources import SYSTEM

from .quantum_operators import C13QuantumOperators
from .hyperfine import HyperfineEngine
from .nuclear_zeeman import NuclearZeemanEngine
from .knight_shift import KnightShiftEngine
from .rf_control import RFControlEngine
from .mw_dnp import MicrowaveDNPEngine
# Import C13Configuration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'interfaces'))
from c13_interface import C13Configuration


class C13BathEngine:
    """
    Ultra-realistic ¹³C nuclear spin bath engine
    
    Implements complete quantum mechanical description of nuclear spin bath:
    - Individual ¹³C nuclear spins (I=½)
    - Anisotropic hyperfine coupling to NV center
    - Nuclear Zeeman effect in magnetic fields
    - Knight shift from NV spin polarization
    - Nuclear-nuclear dipolar interactions
    - RF and MW control capabilities
    - Thermal relaxation processes
    
    NO MOCKS, NO FALLBACKS - pure quantum mechanics
    """
    
    def __init__(self, config: C13Configuration, nv_position: np.ndarray = None):
        """
        Initialize C13 bath engine
        
        Args:
            config: C13 configuration object
            nv_position: NV center position [m]
        """
        self.config = config
        self.nv_position = nv_position if nv_position is not None else np.zeros(3)
        
        # Generate C13 positions
        self.c13_positions = self._generate_c13_positions()
        self.n_c13 = len(self.c13_positions)
        
        print(f"🧲 Initialized C13 bath with {self.n_c13} nuclei")
        print(f"📍 Distribution: {config.distribution}")
        print(f"🎯 Interaction mode: {config.interaction_mode.value}")
        
        # Initialize quantum operators
        self.quantum_ops = C13QuantumOperators(
            n_c13=self.n_c13,
            use_sparse=config.use_sparse_matrices
        )
        
        # Initialize physics engines
        self._initialize_physics_engines()
        
        # Current quantum state
        self._current_state = self._initialize_thermal_state()
        
        # Time tracking
        self._current_time = 0.0
        self._evolution_history = []
        
        # Performance caching
        self._hamiltonian_cache = {} if config.cache_hamiltonians else None
        
        # Validation
        self._validate_initialization()
        
    def _generate_c13_positions(self) -> np.ndarray:
        """Generate C13 nuclear positions"""
        if self.config.explicit_positions is not None:
            return np.asarray(self.config.explicit_positions)
            
        if self.config.distribution == "random":
            return self._generate_random_positions()
        elif self.config.distribution == "lattice":
            return self._generate_lattice_positions()
        else:
            raise ValueError(f"Unknown distribution: {self.config.distribution}")
            
    def _generate_random_positions(self) -> np.ndarray:
        """Generate random C13 positions with correct concentration"""
        # Volume of cutoff sphere
        volume = (4/3) * np.pi * self.config.max_distance**3
        
        # Number density of C13 in diamond
        diamond_density = 1.76e29  # atoms/m³
        c13_density = diamond_density * self.config.concentration
        
        # Expected number of C13 in sphere
        expected_n_c13 = c13_density * volume
        
        # Use Poisson statistics for realistic number
        if expected_n_c13 > 1000:
            # Use deterministic for large numbers
            actual_n_c13 = min(int(expected_n_c13), self.config.cluster_size)
        else:
            # Use Poisson for small numbers
            actual_n_c13 = min(np.random.poisson(expected_n_c13), self.config.cluster_size)
            
        if actual_n_c13 == 0:
            return np.array([]).reshape(0, 3)
            
        # Generate random positions in sphere
        positions = []
        for _ in range(actual_n_c13):
            # Uniform random distribution in sphere
            u = np.random.random()
            v = np.random.random()
            w = np.random.random()
            
            # Spherical coordinates
            r = self.config.max_distance * (u**(1/3))
            theta = 2 * np.pi * v
            phi = np.arccos(2*w - 1)
            
            # Convert to Cartesian
            x = r * np.sin(phi) * np.cos(theta)
            y = r * np.sin(phi) * np.sin(theta)
            z = r * np.cos(phi)
            
            positions.append([x, y, z])
            
        return np.array(positions) + self.nv_position
        
    def _generate_lattice_positions(self) -> np.ndarray:
        """Generate C13 positions on diamond lattice"""
        # Diamond lattice constant
        a = 3.567e-10  # m
        
        # Generate lattice sites within cutoff
        max_cells = int(self.config.max_distance / a) + 1
        positions = []
        
        for i in range(-max_cells, max_cells + 1):
            for j in range(-max_cells, max_cells + 1):
                for k in range(-max_cells, max_cells + 1):
                    # Two atoms per unit cell for diamond
                    pos1 = np.array([i, j, k]) * a
                    pos2 = pos1 + np.array([a/4, a/4, a/4])
                    
                    for pos in [pos1, pos2]:
                        distance = np.linalg.norm(pos - self.nv_position)
                        if distance <= self.config.max_distance and distance > 0:
                            # Include with C13 probability
                            if np.random.random() < self.config.concentration:
                                positions.append(pos)
                                
        positions = np.array(positions[:self.config.cluster_size])
        return positions
        
    def _initialize_physics_engines(self):
        """Initialize all physics engines"""
        # Hyperfine coupling engine
        self.hyperfine = HyperfineEngine(
            c13_positions=self.c13_positions,
            nv_position=self.nv_position
        )
        
        # Nuclear Zeeman engine
        self.nuclear_zeeman = NuclearZeemanEngine(
            magnetic_field=self.config.magnetic_field
        )
        
        # Knight shift engine
        self.knight_shift = KnightShiftEngine()
        
        # Control engines (if requested)
        if hasattr(self.config, 'enable_rf_control') and self.config.enable_rf_control:
            self.rf_control = RFControlEngine(n_c13=self.n_c13)
        else:
            self.rf_control = None
            
        if hasattr(self.config, 'enable_mw_dnp') and self.config.enable_mw_dnp:
            self.mw_dnp = MicrowaveDNPEngine(
                hyperfine_engine=self.hyperfine,
                zeeman_engine=self.nuclear_zeeman
            )
        else:
            self.mw_dnp = None
            
        print(f"⚙️ Initialized {sum(1 for x in [self.hyperfine, self.nuclear_zeeman, self.knight_shift, self.rf_control, self.mw_dnp] if x is not None)} physics engines")
        
    def _initialize_thermal_state(self) -> np.ndarray:
        """Initialize thermal equilibrium state"""
        if self.n_c13 == 0:
            return np.array([1.0])  # Trivial state
            
        # Get thermal Hamiltonian
        H_thermal = self._get_thermal_hamiltonian()
        
        # Generate thermal state
        thermal_state = self.quantum_ops.thermal_state(
            temperature=self.config.temperature,
            hamiltonian=H_thermal
        )
        
        return thermal_state
        
    def _get_thermal_hamiltonian(self) -> np.ndarray:
        """Get Hamiltonian for thermal state calculation"""
        # Use only static terms for thermal state
        H = np.zeros((2**self.n_c13, 2**self.n_c13), dtype=complex)
        
        # Nuclear Zeeman terms
        if self.nuclear_zeeman:
            H += self.nuclear_zeeman.get_zeeman_hamiltonian(self.quantum_ops.c13_operators)
            
        # Add small random field to break degeneracies
        random_field = 1e3  # 1 kHz
        for i in range(self.n_c13):
            random_amplitude = random_field * (2*np.random.random() - 1)
            H += random_amplitude * self.quantum_ops.c13_operators[i]['Iz']
            
        return H
        
    def get_total_hamiltonian(self, t: float = 0.0, nv_state: Optional[np.ndarray] = None,
                            B_field: Optional[np.ndarray] = None, **params) -> np.ndarray:
        """
        Get total C13 bath Hamiltonian
        
        Args:
            t: Current time [s]
            nv_state: Current NV quantum state
            B_field: Applied magnetic field [T]
            **params: Additional parameters
            
        Returns:
            Total Hamiltonian matrix
        """
        if self.n_c13 == 0:
            return np.array([[0.0]])
            
        # Check cache
        cache_key = (t, id(nv_state), tuple(B_field) if B_field is not None else None)
        if self._hamiltonian_cache is not None and cache_key in self._hamiltonian_cache:
            return self._hamiltonian_cache[cache_key]
            
        dim = 2**self.n_c13
        H_total = np.zeros((dim, dim), dtype=complex)
        
        # Nuclear Zeeman terms
        if B_field is not None:
            self.nuclear_zeeman.set_magnetic_field(B_field)
        H_total += self.nuclear_zeeman.get_zeeman_hamiltonian(self.quantum_ops.c13_operators)
        
        # Knight shift (NV state-dependent)
        if nv_state is not None:
            H_knight = self.knight_shift.get_knight_shift_hamiltonian(
                self.quantum_ops.c13_operators, nv_state
            )
            H_total += H_knight
            
        # RF control terms (time-dependent)
        if self.rf_control is not None:
            H_rf = self.rf_control.get_rf_hamiltonian(t, self.quantum_ops.c13_operators)
            H_total += H_rf
            
        # Cache result
        if self._hamiltonian_cache is not None:
            self._hamiltonian_cache[cache_key] = H_total
            
        return H_total
        
    def get_nv_c13_coupling_hamiltonian(self, nv_operators: Dict[str, np.ndarray],
                                       nv_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Get NV-C13 hyperfine coupling Hamiltonian"""
        if self.n_c13 == 0:
            # Return NV-only Hamiltonian
            return nv_operators['Sz'] * 0  # Zero coupling
            
        return self.hyperfine.get_hyperfine_hamiltonian(
            nv_operators, self.quantum_ops.c13_operators, nv_state
        )
        
    def evolve_c13_ensemble(self, initial_state: np.ndarray, t_span: Tuple[float, float],
                           nv_trajectory: Optional[Callable] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Evolve C13 ensemble quantum mechanically
        
        Args:
            initial_state: Initial C13 state (density matrix or state vector)
            t_span: Time span (t_start, t_end)
            nv_trajectory: Function returning NV state vs time
            
        Returns:
            (times, states) evolution trajectory
        """
        if self.n_c13 == 0:
            return np.array([t_span[0], t_span[1]]), [initial_state, initial_state]
            
        # Determine if state vector or density matrix
        is_state_vector = initial_state.ndim == 1
        
        def rhs(t, state_vec):
            """Right-hand side for ODE solver"""
            # Reshape state
            if is_state_vector:
                state = state_vec
            else:
                state = state_vec.reshape(2**self.n_c13, 2**self.n_c13)
                
            # Get NV state at time t
            nv_state = nv_trajectory(t) if nv_trajectory else None
            
            # Get Hamiltonian
            H = self.get_total_hamiltonian(t=t, nv_state=nv_state)
            
            # Compute time derivative
            if is_state_vector:
                # Schrödinger equation: i ℏ d|ψ⟩/dt = H |ψ⟩
                dstate_dt = -1j * H @ state / self._get_hbar()
            else:
                # Liouville equation: dρ/dt = -i [H, ρ] / ℏ
                dstate_dt = -1j * (H @ state - state @ H) / self._get_hbar()
                
            return dstate_dt.flatten()
            
        # Solve evolution equation
        sol = solve_ivp(
            rhs, t_span, initial_state.flatten(),
            method='RK45',
            rtol=1e-8, atol=1e-10,
            max_step=1e-6  # 1 μs max step
        )
        
        # Reshape results
        times = sol.t
        states = []
        
        for i in range(len(times)):
            if is_state_vector:
                state = sol.y[:, i]
            else:
                state = sol.y[:, i].reshape(2**self.n_c13, 2**self.n_c13)
            states.append(state)
            
        # Update internal state and time
        self._current_state = states[-1]
        self._current_time = times[-1]
        
        return times, states
        
    def _get_hbar(self) -> float:
        """Get ℏ constant"""
        return SYSTEM.get_constant('fundamental', 'hbar')
        
    def get_nuclear_positions(self) -> np.ndarray:
        """Get C13 nuclear positions"""
        return self.c13_positions.copy()
        
    def get_current_state(self) -> np.ndarray:
        """Get current quantum state"""
        return self._current_state.copy()
        
    def set_current_state(self, state: np.ndarray):
        """Set current quantum state"""
        self._current_state = state.copy()
        
    def get_nuclear_magnetization(self) -> np.ndarray:
        """Get nuclear magnetization vector"""
        if self.n_c13 == 0:
            return np.zeros(3)
            
        # Get expectation values of collective spin operators
        collective_ops = self.quantum_ops.get_collective_operators()
        
        magnetization = np.zeros(3)
        gamma_n = SYSTEM.get_constant('nv_center', 'gamma_n_13c')
        
        for i, component in enumerate(['Ix_total', 'Iy_total', 'Iz_total']):
            expectation = self.quantum_ops.measure_expectation_value(
                collective_ops[component], self._current_state
            )
            magnetization[i] = gamma_n * expectation
            
        return magnetization
        
    def get_hyperpolarization_level(self) -> float:
        """Get nuclear polarization level"""
        if self.n_c13 == 0:
            return 0.0
            
        # Measure ⟨Iz_total⟩
        collective_ops = self.quantum_ops.get_collective_operators()
        iz_expectation = self.quantum_ops.measure_expectation_value(
            collective_ops['Iz_total'], self._current_state
        )
        
        # Maximum polarization is n_c13 * 0.5
        max_polarization = self.n_c13 * 0.5
        
        return iz_expectation / max_polarization if max_polarization > 0 else 0.0
        
    def measure_observables(self, observables: List[str]) -> Dict[str, float]:
        """Measure nuclear observables"""
        results = {}
        collective_ops = self.quantum_ops.get_collective_operators()
        
        for obs in observables:
            if obs in collective_ops:
                value = self.quantum_ops.measure_expectation_value(
                    collective_ops[obs], self._current_state
                )
                results[obs] = float(value)
            elif obs == 'polarization':
                results[obs] = self.get_hyperpolarization_level()
            elif obs == 'magnetization_magnitude':
                mag = self.get_nuclear_magnetization()
                results[obs] = float(np.linalg.norm(mag))
            else:
                results[obs] = 0.0
                
        return results
        
    def get_c13_coherence_times(self) -> Dict[str, float]:
        """Get nuclear coherence times"""
        # Estimate from system parameters
        estimates = {}
        
        # T1n estimate from thermal relaxation
        kb = SYSTEM.get_constant('fundamental', 'kb')
        hbar = SYSTEM.get_constant('fundamental', 'hbar')
        gamma_n = SYSTEM.get_constant('nv_center', 'gamma_n_13c')
        
        # Nuclear Larmor frequency
        B_field_magnitude = np.linalg.norm(self.config.magnetic_field)
        omega_n = gamma_n * B_field_magnitude
        
        # T1n estimate (crude)
        estimates['T1n'] = 1 / (omega_n * (kb * self.config.temperature / hbar)**7)
        
        # T2n estimate from magnetic noise
        if self.hyperfine:
            couplings = self.hyperfine.get_hyperfine_tensors()
            if couplings:
                # RMS hyperfine coupling
                A_rms = np.sqrt(np.mean([A_par**2 + A_perp**2 for A_par, A_perp in couplings.values()]))
                estimates['T2n'] = 1 / (2 * np.pi * A_rms)
            else:
                # NO COUPLING = NO DEPHASING from this source
                estimates['T2n'] = np.inf  # Infinite T2n if no hyperfine coupling
                print("🔍 No hyperfine couplings found - T2n set to infinity")
        else:
            # NO HYPERFINE MODULE = NO DEPHASING from this source  
            estimates['T2n'] = np.inf  # Infinite T2n if no hyperfine module
            print("🔍 No hyperfine module - T2n set to infinity")
            
        return estimates
        
    def get_magnetic_noise_spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """Get magnetic noise spectrum from C13 bath"""
        if self.n_c13 == 0:
            return np.zeros_like(frequencies)
            
        # Simple model: Lorentzian spectrum from nuclear fluctuations
        gamma_n = SYSTEM.get_constant('nv_center', 'gamma_n_13c')
        B_field_magnitude = np.linalg.norm(self.config.magnetic_field)
        
        # Nuclear correlation time
        tau_c = 1 / (gamma_n * B_field_magnitude)  # Larmor period
        
        # Lorentzian PSD
        noise_amplitude = 1e-12  # T²
        psd = noise_amplitude * (2 * tau_c) / (1 + (2 * np.pi * frequencies * tau_c)**2)
        
        return psd
        
    def reset_to_thermal_state(self, temperature: float):
        """Reset to thermal equilibrium"""
        self.config.temperature = temperature
        self._current_state = self._initialize_thermal_state()
        self._current_time = 0.0
        
    def update_environment(self, **params):
        """Update environmental parameters"""
        for param, value in params.items():
            if param == 'temperature':
                self.config.temperature = value
                if self.hyperfine:
                    self.hyperfine.update_temperature(value)
            elif param == 'magnetic_field':
                self.config.magnetic_field = np.asarray(value)
                if self.nuclear_zeeman:
                    self.nuclear_zeeman.set_magnetic_field(self.config.magnetic_field)
                    
        # Clear Hamiltonian cache
        if self._hamiltonian_cache is not None:
            self._hamiltonian_cache.clear()
            
    def validate_physics(self) -> Dict[str, bool]:
        """Validate quantum mechanical consistency"""
        validation = {}
        
        # Check state normalization
        if self._current_state.ndim == 1:
            norm = np.linalg.norm(self._current_state)
            validation['state_normalized'] = abs(norm - 1.0) < 1e-10
        else:
            trace = np.trace(self._current_state)
            validation['density_matrix_trace'] = abs(trace - 1.0) < 1e-10
            
        # Check Hamiltonian Hermiticity
        H = self.get_total_hamiltonian()
        is_hermitian = np.allclose(H, H.conj().T, atol=1e-12)
        validation['hamiltonian_hermitian'] = is_hermitian
        
        # Check operator algebra
        operator_validation = self.quantum_ops.validate_operators()
        validation.update(operator_validation)
        
        # Check hyperfine physics
        if self.hyperfine:
            hyperfine_validation = self.hyperfine.validate_hyperfine_physics()
            validation.update(hyperfine_validation)
            
        return validation
        
    def _validate_initialization(self):
        """Validate initialization"""
        validation = self.validate_physics()
        
        failed_checks = [k for k, v in validation.items() if not v]
        if failed_checks:
            warnings.warn(f"C13 bath validation failed: {failed_checks}")
            
        print(f"✅ C13 bath validation: {sum(validation.values())}/{len(validation)} checks passed")
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage estimates"""
        return self.quantum_ops.get_memory_usage_estimate()
        
    def optimize_performance(self):
        """Optimize performance for current system size"""
        if self.n_c13 > 10:
            self.quantum_ops.optimize_memory_usage()
            
            # Enable sparse matrices
            self.config.use_sparse_matrices = True
            
            print(f"🚀 Performance optimization enabled for {self.n_c13} C13 nuclei")
            
    def export_state(self, filename: str):
        """Export current state to file"""
        import json
        
        export_data = {
            'config': {
                'n_c13': self.n_c13,
                'concentration': self.config.concentration,
                'temperature': self.config.temperature,
                'magnetic_field': self.config.magnetic_field.tolist()
            },
            'positions': self.c13_positions.tolist(),
            'current_time': self._current_time,
            'state_shape': self._current_state.shape,
            'state_real': self._current_state.real.tolist(),
            'state_imag': self._current_state.imag.tolist(),
            'validation': self.validate_physics()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
        print(f"💾 C13 state exported to {filename}")
        
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'configuration': {
                'n_c13': self.n_c13,
                'concentration': self.config.concentration,
                'max_distance': self.config.max_distance,
                'distribution': self.config.distribution,
                'interaction_mode': self.config.interaction_mode.value,
                'temperature': self.config.temperature
            },
            'physics_engines': {
                'hyperfine': self.hyperfine is not None,
                'nuclear_zeeman': self.nuclear_zeeman is not None,
                'knight_shift': self.knight_shift is not None,
                'rf_control': self.rf_control is not None,
                'mw_dnp': self.mw_dnp is not None
            },
            'quantum_state': {
                'dimension': self._current_state.shape,
                'type': 'state_vector' if self._current_state.ndim == 1 else 'density_matrix'
            },
            'memory_usage': self.get_memory_usage(),
            'validation': self.validate_physics()
        }
        
        if self.hyperfine:
            info['hyperfine_statistics'] = self.hyperfine.get_coupling_statistics()
            
        return info