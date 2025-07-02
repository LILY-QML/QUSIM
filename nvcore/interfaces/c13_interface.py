"""
C13 Nuclear Spin Bath Interface

Ultra-realistic interface for quantum mechanical ¹³C nuclear spin baths.
Provides clean separation between NV center physics and nuclear dynamics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum


class C13InteractionMode(Enum):
    """Modes for handling C13-C13 interactions"""
    ISOLATED = "isolated"        # Single C13, no interactions
    PAIRWISE = "pairwise"       # Only nearest-neighbor dipolar
    CCE = "cce"                 # Cluster Correlation Expansion
    FULL = "full"               # Full many-body (expensive)


@dataclass
class C13Configuration:
    """Configuration for C13 nuclear spin bath"""
    
    # Basic parameters
    concentration: float = 0.011        # Natural abundance (1.1%)
    max_distance: float = 10e-9         # Cutoff distance [m]
    cluster_size: int = 100             # Number of C13 nuclei
    
    # Spatial distribution
    distribution: str = "random"        # "random", "lattice", "experimental"
    explicit_positions: Optional[np.ndarray] = None  # For experimental data
    
    # Physics models
    interaction_mode: C13InteractionMode = C13InteractionMode.CCE
    max_cluster_order: int = 4          # CCE expansion order
    include_fermi_contact: bool = True  # Include contact interactions
    
    # Environmental parameters
    temperature: float = 300.0          # Temperature [K]
    magnetic_field: np.ndarray = None   # Applied B-field [T]
    
    # Performance options
    use_sparse_matrices: bool = True    # For large systems
    cache_hamiltonians: bool = True     # Cache expensive calculations
    
    def __post_init__(self):
        if self.magnetic_field is None:
            self.magnetic_field = np.array([0., 0., 0.01])  # 10 mT default


class C13Interface(ABC):
    """
    Abstract interface for all C13 nuclear spin bath implementations
    
    Provides quantum mechanical interface for:
    - Individual ¹³C nuclear spins (I=½)
    - Anisotropic hyperfine coupling to NV center
    - Nuclear-nuclear dipolar interactions
    - Dynamic environment effects
    - RF and MW control capabilities
    """
    
    @abstractmethod
    def get_c13_hamiltonian(self, nv_state: Optional[np.ndarray] = None, 
                           t: float = 0.0, **params) -> np.ndarray:
        """
        Get total C13 bath Hamiltonian
        
        Args:
            nv_state: Current NV quantum state (for feedback effects)
            t: Current time (for time-dependent effects)
            **params: Additional parameters (B-field, temperature, etc.)
            
        Returns:
            Total C13 Hamiltonian matrix
        """
        pass
    
    @abstractmethod
    def get_nv_c13_coupling_hamiltonian(self, nv_operators: Dict[str, np.ndarray],
                                       nv_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get NV-C13 hyperfine coupling Hamiltonian
        
        Args:
            nv_operators: Dictionary with NV spin operators
            nv_state: Current NV state (for state-dependent coupling)
            
        Returns:
            NV-C13 coupling Hamiltonian in joint Hilbert space
        """
        pass
    
    @abstractmethod
    def evolve_c13_ensemble(self, initial_state: np.ndarray, t_span: Tuple[float, float],
                           nv_trajectory: Optional[Callable] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Evolve C13 ensemble quantum mechanically
        
        Args:
            initial_state: Initial C13 ensemble state
            t_span: Time span (t_start, t_end)
            nv_trajectory: Function providing NV state vs time
            
        Returns:
            (times, states) tuple with evolution trajectory
        """
        pass
    
    @abstractmethod
    def apply_rf_pulse(self, target_nuclei: Union[int, List[int]], 
                      pulse_params: Dict[str, Any]) -> np.ndarray:
        """
        Apply RF pulse to specific C13 nuclei
        
        Args:
            target_nuclei: Index/indices of target C13 nuclei
            pulse_params: Pulse parameters (frequency, amplitude, phase, duration)
            
        Returns:
            RF pulse propagator
        """
        pass
    
    @abstractmethod
    def apply_mw_dnp_sequence(self, dnp_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply microwave-driven dynamic nuclear polarization
        
        Args:
            dnp_params: DNP sequence parameters
            
        Returns:
            DNP transfer results (polarization, efficiency, etc.)
        """
        pass
    
    @abstractmethod
    def get_hyperfine_couplings(self) -> Dict[int, Tuple[float, float]]:
        """
        Get hyperfine coupling constants for all C13 nuclei
        
        Returns:
            Dictionary mapping C13 index to (A_parallel, A_perpendicular) [Hz]
        """
        pass
    
    @abstractmethod
    def get_nuclear_positions(self) -> np.ndarray:
        """
        Get 3D positions of all C13 nuclei
        
        Returns:
            Array of shape (N_c13, 3) with positions in meters
        """
        pass
    
    @abstractmethod
    def get_c13_quantum_state(self) -> np.ndarray:
        """
        Get current quantum state of C13 ensemble
        
        Returns:
            Complex state vector in C13 Hilbert space
        """
        pass
    
    @abstractmethod
    def set_c13_quantum_state(self, state: np.ndarray):
        """
        Set quantum state of C13 ensemble
        
        Args:
            state: New quantum state vector
        """
        pass
    
    @abstractmethod
    def get_nuclear_magnetization(self) -> np.ndarray:
        """
        Get nuclear magnetization vector
        
        Returns:
            3D magnetization vector [Mx, My, Mz]
        """
        pass
    
    @abstractmethod
    def get_nuclear_polarization(self) -> float:
        """
        Get nuclear polarization level
        
        Returns:
            Polarization as fraction of maximum
        """
        pass
    
    @abstractmethod
    def measure_nuclear_observables(self, observables: List[str]) -> Dict[str, float]:
        """
        Measure nuclear observables
        
        Args:
            observables: List of observables to measure
            
        Returns:
            Dictionary with measurement results
        """
        pass
    
    @abstractmethod
    def get_coherence_times(self) -> Dict[str, float]:
        """
        Get nuclear coherence times
        
        Returns:
            Dictionary with T1n, T2n values [s]
        """
        pass
    
    @abstractmethod
    def get_noise_spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Get magnetic noise spectrum from C13 bath
        
        Args:
            frequencies: Frequency array [Hz]
            
        Returns:
            Noise power spectral density [T²/Hz]
        """
        pass
    
    @abstractmethod
    def reset_to_thermal_equilibrium(self, temperature: float):
        """
        Reset C13 bath to thermal equilibrium
        
        Args:
            temperature: Bath temperature [K]
        """
        pass
    
    @abstractmethod
    def set_environment_parameters(self, **params):
        """
        Update environmental parameters
        
        Args:
            **params: Environment parameters (temperature, B-field, etc.)
        """
        pass
    
    @abstractmethod
    def validate_quantum_mechanics(self) -> Dict[str, bool]:
        """
        Validate quantum mechanical consistency
        
        Returns:
            Dictionary with validation results
        """
        pass
    
    # Optional advanced methods
    
    def get_cluster_expansion_terms(self, max_order: int = 4) -> Dict[int, np.ndarray]:
        """
        Get cluster correlation expansion terms
        
        Args:
            max_order: Maximum cluster order
            
        Returns:
            Dictionary mapping cluster order to Hamiltonian terms
        """
        return {}
    
    def get_dipolar_coupling_matrix(self) -> np.ndarray:
        """
        Get C13-C13 dipolar coupling matrix
        
        Returns:
            Coupling matrix [Hz]
        """
        return np.array([])
    
    def compute_dnp_efficiency(self, mw_sequence: List[Dict]) -> float:
        """
        Compute DNP transfer efficiency for given MW sequence
        
        Args:
            mw_sequence: List of MW pulse parameters
            
        Returns:
            DNP efficiency (0 to 1)
        """
        return 0.0
    
    def optimize_rf_sequence(self, target_operation: str, **constraints) -> List[Dict]:
        """
        Optimize RF pulse sequence for target operation
        
        Args:
            target_operation: Target gate/operation
            **constraints: Optimization constraints
            
        Returns:
            Optimized pulse sequence
        """
        return []
    
    def get_real_time_feedback(self) -> Dict[str, Any]:
        """
        Get real-time feedback for adaptive control
        
        Returns:
            Current system state and control recommendations
        """
        return {}


class C13BathEngineAdapter(C13Interface):
    """
    Adapter to integrate C13BathEngine with the interface
    
    This adapter will wrap the detailed C13BathEngine implementation
    and provide a clean interface for the NV system.
    """
    
    def __init__(self, c13_engine):
        """
        Initialize adapter with C13BathEngine instance
        
        Args:
            c13_engine: Instance of C13BathEngine
        """
        self.c13_engine = c13_engine
        self._last_nv_state = None
        self._last_update_time = 0.0
        
    def get_c13_hamiltonian(self, nv_state: Optional[np.ndarray] = None, 
                           t: float = 0.0, **params) -> np.ndarray:
        """Get C13 Hamiltonian using engine"""
        return self.c13_engine.get_total_hamiltonian(t=t, nv_state=nv_state, **params)
    
    def get_nv_c13_coupling_hamiltonian(self, nv_operators: Dict[str, np.ndarray],
                                       nv_state: Optional[np.ndarray] = None) -> np.ndarray:
        """Get NV-C13 coupling using hyperfine engine"""
        return self.c13_engine.hyperfine.get_hyperfine_hamiltonian(nv_operators, nv_state)
    
    def evolve_c13_ensemble(self, initial_state: np.ndarray, t_span: Tuple[float, float],
                           nv_trajectory: Optional[Callable] = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Evolve ensemble using engine"""
        return self.c13_engine.evolve_c13_ensemble(initial_state, t_span, nv_trajectory)
    
    def apply_rf_pulse(self, target_nuclei: Union[int, List[int]], 
                      pulse_params: Dict[str, Any]) -> np.ndarray:
        """Apply RF pulse using RF control engine"""
        return self.c13_engine.rf_control.apply_rf_pulse(target_nuclei, pulse_params)
    
    def apply_mw_dnp_sequence(self, dnp_params: Dict[str, Any]) -> Dict[str, float]:
        """Apply DNP using MW DNP engine"""
        return self.c13_engine.mw_dnp.apply_dnp_sequence(dnp_params)
    
    def get_hyperfine_couplings(self) -> Dict[int, Tuple[float, float]]:
        """Get hyperfine couplings from hyperfine engine"""
        return self.c13_engine.hyperfine.get_hyperfine_tensors()
    
    def get_nuclear_positions(self) -> np.ndarray:
        """Get nuclear positions"""
        return self.c13_engine.get_nuclear_positions()
    
    def get_c13_quantum_state(self) -> np.ndarray:
        """Get current quantum state"""
        return self.c13_engine.get_current_state()
    
    def set_c13_quantum_state(self, state: np.ndarray):
        """Set quantum state"""
        self.c13_engine.set_current_state(state)
    
    def get_nuclear_magnetization(self) -> np.ndarray:
        """Get nuclear magnetization"""
        return self.c13_engine.get_nuclear_magnetization()
    
    def get_nuclear_polarization(self) -> float:
        """Get nuclear polarization"""
        return self.c13_engine.get_hyperpolarization_level()
    
    def measure_nuclear_observables(self, observables: List[str]) -> Dict[str, float]:
        """Measure observables"""
        return self.c13_engine.measure_observables(observables)
    
    def get_coherence_times(self) -> Dict[str, float]:
        """Get coherence times"""
        return self.c13_engine.get_c13_coherence_times()
    
    def get_noise_spectrum(self, frequencies: np.ndarray) -> np.ndarray:
        """Get noise spectrum"""
        return self.c13_engine.get_magnetic_noise_spectrum(frequencies)
    
    def reset_to_thermal_equilibrium(self, temperature: float):
        """Reset to thermal equilibrium"""
        self.c13_engine.reset_to_thermal_state(temperature)
    
    def set_environment_parameters(self, **params):
        """Set environment parameters"""
        self.c13_engine.update_environment(**params)
    
    def validate_quantum_mechanics(self) -> Dict[str, bool]:
        """Validate quantum mechanics"""
        return self.c13_engine.validate_physics()
    
    # Advanced methods
    
    def get_cluster_expansion_terms(self, max_order: int = 4) -> Dict[int, np.ndarray]:
        """Get CCE terms"""
        if hasattr(self.c13_engine, 'cluster_expansion'):
            return self.c13_engine.cluster_expansion.get_terms(max_order)
        return {}
    
    def get_dipolar_coupling_matrix(self) -> np.ndarray:
        """Get dipolar coupling matrix"""
        if hasattr(self.c13_engine, 'dipole_coupling'):
            return self.c13_engine.dipole_coupling.get_coupling_matrix()
        return np.array([])
    
    def compute_dnp_efficiency(self, mw_sequence: List[Dict]) -> float:
        """Compute DNP efficiency"""
        if hasattr(self.c13_engine, 'mw_dnp'):
            return self.c13_engine.mw_dnp.compute_efficiency(mw_sequence)
        return 0.0


# Factory functions for common configurations

def create_single_c13_interface(position: np.ndarray, 
                               nv_position: np.ndarray = None) -> C13Interface:
    """
    Create interface for single C13 nucleus
    
    Args:
        position: C13 position relative to NV [m]
        nv_position: NV center position [m]
        
    Returns:
        C13Interface for single nucleus
    """
    config = C13Configuration(
        concentration=0.0,  # Not applicable for single nucleus
        explicit_positions=position.reshape(1, 3),
        interaction_mode=C13InteractionMode.ISOLATED,
        cluster_size=1
    )
    
    from modules.c13 import C13BathEngine
    engine = C13BathEngine(config, nv_position)
    return C13BathEngineAdapter(engine)


def create_natural_abundance_c13_interface(cluster_size: int = 100,
                                         max_distance: float = 10e-9) -> C13Interface:
    """
    Create interface for natural abundance C13 bath
    
    Args:
        cluster_size: Number of C13 nuclei to include
        max_distance: Maximum distance from NV [m]
        
    Returns:
        C13Interface for natural abundance bath
    """
    config = C13Configuration(
        concentration=0.011,  # Natural abundance
        cluster_size=cluster_size,
        max_distance=max_distance,
        interaction_mode=C13InteractionMode.CCE,
        distribution="random"
    )
    
    from modules.c13 import C13BathEngine
    engine = C13BathEngine(config)
    return C13BathEngineAdapter(engine)


def create_isotopically_pure_c13_interface(cluster_size: int = 50) -> C13Interface:
    """
    Create interface for isotopically pure C13 diamond
    
    Args:
        cluster_size: Number of C13 nuclei
        
    Returns:
        C13Interface for isotopically pure system
    """
    config = C13Configuration(
        concentration=0.999,  # Isotopically pure
        cluster_size=cluster_size,
        interaction_mode=C13InteractionMode.FULL,  # Many-body interactions important
        distribution="lattice"
    )
    
    from modules.c13 import C13BathEngine
    engine = C13BathEngine(config)
    return C13BathEngineAdapter(engine)


def create_experimental_c13_interface(positions: np.ndarray,
                                    coupling_strengths: Optional[Dict] = None) -> C13Interface:
    """
    Create interface from experimental C13 data
    
    Args:
        positions: Measured C13 positions [m]
        coupling_strengths: Measured hyperfine couplings [Hz]
        
    Returns:
        C13Interface with experimental parameters
    """
    config = C13Configuration(
        concentration=0.0,  # Not applicable
        explicit_positions=positions,
        interaction_mode=C13InteractionMode.CCE,
        cluster_size=len(positions)
    )
    
    from modules.c13 import C13BathEngine
    engine = C13BathEngine(config)
    
    # Set experimental coupling strengths if provided
    if coupling_strengths:
        engine.hyperfine.set_experimental_couplings(coupling_strengths)
    
    return C13BathEngineAdapter(engine)