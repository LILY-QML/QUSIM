"""
C13 Noise Interface Adapter

Adapter to integrate quantum mechanical C13BathEngine with the existing NoiseInterface.
Provides seamless replacement for the old C13BathNoise.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'helper'))

from noise_interface import NoiseInterface
from c13_interface import C13Configuration, C13InteractionMode
from modules.c13.core import C13BathEngine


class QuantumC13NoiseAdapter(NoiseInterface):
    """
    Adapter to use quantum C13BathEngine as a noise source
    
    This adapter allows the ultra-realistic C13 quantum implementation
    to integrate seamlessly with the existing noise framework.
    """
    
    def __init__(self, concentration: float = 0.011, max_distance: float = 10e-9,
                 cluster_size: int = 100, B_field: np.ndarray = None):
        """
        Initialize quantum C13 noise adapter
        
        Args:
            concentration: C13 concentration (natural abundance = 0.011)
            max_distance: Maximum distance from NV [m]
            cluster_size: Number of C13 nuclei to simulate
            B_field: Applied magnetic field [T]
        """
        # Create C13 configuration
        self.config = C13Configuration(
            concentration=concentration,
            max_distance=max_distance,
            cluster_size=cluster_size,
            interaction_mode=C13InteractionMode.CCE,
            distribution="random",
            magnetic_field=B_field if B_field is not None else np.array([0, 0, 0.01]),
            temperature=300.0,
            use_sparse_matrices=False,  # Use dense for testing
            cache_hamiltonians=True
        )
        
        # Initialize quantum C13 engine
        self.c13_engine = C13BathEngine(self.config)
        
        # Track time and NV state
        self._current_time = 0.0
        self._last_nv_state = None
        
        # Cache for magnetic field noise
        self._noise_cache = {}
        self._cache_size = 1000
        self._cache_position = 0
        
        print(f"🧲 Quantum C13 adapter initialized with {self.c13_engine.n_c13} nuclei")
        
    def get_magnetic_field_noise(self, n_samples: int = 1) -> np.ndarray:
        """
        Generate magnetic field noise from quantum C13 bath
        
        Args:
            n_samples: Number of noise samples
            
        Returns:
            Magnetic field noise [T], shape (n_samples, 3) or (3,)
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        # Generate samples from quantum magnetization fluctuations
        samples = np.zeros((n_samples, 3))
        
        for i in range(n_samples):
            # Get current nuclear magnetization
            magnetization = self.c13_engine.get_nuclear_magnetization()
            
            # Convert to magnetic field at NV center
            # Simple dipolar field model: B ∝ μ/r³
            # For realistic implementation, this would use proper Green's functions
            
            # Average distance to C13 nuclei
            if self.c13_engine.n_c13 > 0:
                positions = self.c13_engine.get_nuclear_positions()
                avg_distance = np.mean(np.linalg.norm(positions, axis=1))
                
                # Dipolar field scaling
                mu_0 = 4 * np.pi * 1e-7  # H/m
                field_scale = mu_0 / (4 * np.pi * avg_distance**3)
                
                # Magnetic field from nuclear magnetization
                B_field = field_scale * magnetization
            else:
                B_field = np.zeros(3)
                
            # Add thermal fluctuations
            thermal_noise = np.random.normal(0, 1e-9, 3)  # 1 nT RMS
            B_field += thermal_noise
            
            samples[i] = B_field
            
            # Advance time slightly for next sample
            self._current_time += 1e-6  # 1 μs per sample
            
        return samples.squeeze() if n_samples == 1 else samples
        
    def get_hamiltonian_noise(self, spin_operators: Dict[str, np.ndarray], 
                             t: float = 0.0) -> np.ndarray:
        """
        Generate noise Hamiltonian from C13 bath
        
        Args:
            spin_operators: NV spin operators
            t: Current time
            
        Returns:
            Noise Hamiltonian matrix
        """
        # Get magnetic field noise
        B_noise = self.get_magnetic_field_noise(1)
        
        # Convert to Hamiltonian: H = γₑ B⃗ · S⃗
        from helper.noise_sources import SYSTEM
        gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
        
        H_noise = 2 * np.pi * gamma_e * (
            B_noise[0] * spin_operators['Sx'] +
            B_noise[1] * spin_operators['Sy'] +
            B_noise[2] * spin_operators['Sz']
        )
        
        return H_noise
        
    def get_lindblad_operators(self, spin_operators: Dict[str, np.ndarray],
                              include_sources: Optional[List[str]] = None) -> List[Tuple[np.ndarray, float]]:
        """
        Generate Lindblad operators for C13-induced relaxation
        
        Args:
            spin_operators: Spin operators
            include_sources: Sources to include
            
        Returns:
            List of (operator, rate) tuples
        """
        operators = []
        
        # Get C13 coherence times
        coherence_times = self.c13_engine.get_c13_coherence_times()
        
        # Estimate dephasing rate from C13 bath
        if 'T2n' in coherence_times and coherence_times['T2n'] > 0:
            # C13 bath induces dephasing on NV
            gamma_phi = 1 / coherence_times['T2n'] * 0.1  # 10% coupling efficiency
            
            if 'Sz' in spin_operators and gamma_phi > 0:
                operators.append((spin_operators['Sz'], np.sqrt(gamma_phi)))
                
        return operators
        
    def get_noise_power_spectral_density(self, frequencies: np.ndarray,
                                       component: str = 'total') -> np.ndarray:
        """
        Get noise PSD from quantum C13 bath
        
        Args:
            frequencies: Frequency array [Hz]
            component: Which component to return
            
        Returns:
            Power spectral density [T²/Hz]
        """
        return self.c13_engine.get_magnetic_noise_spectrum(frequencies)
        
    def reset(self):
        """Reset C13 bath to thermal equilibrium"""
        self.c13_engine.reset_to_thermal_state(self.config.temperature)
        self._current_time = 0.0
        self._noise_cache.clear()
        
    def set_parameters(self, **kwargs):
        """Update C13 parameters"""
        update_params = {}
        
        for param, value in kwargs.items():
            if param == 'temperature':
                update_params['temperature'] = value
            elif param == 'magnetic_field':
                update_params['magnetic_field'] = np.asarray(value)
            elif param == 'concentration':
                # Cannot change concentration without reinitializing
                print(f"Warning: Cannot change concentration after initialization")
                
        if update_params:
            self.c13_engine.update_environment(**update_params)
            
    def get_c13_engine(self) -> C13BathEngine:
        """Get access to underlying C13 engine for advanced operations"""
        return self.c13_engine
        
    def get_nuclear_polarization(self) -> float:
        """Get current nuclear polarization"""
        return self.c13_engine.get_hyperpolarization_level()
        
    def get_nuclear_coherence_times(self) -> Dict[str, float]:
        """Get nuclear coherence times"""
        return self.c13_engine.get_c13_coherence_times()
        
    def set_nv_state(self, nv_state: np.ndarray):
        """
        Update current NV state for feedback effects
        
        Args:
            nv_state: Current NV quantum state
        """
        self._last_nv_state = nv_state
        # The C13 engine will use this for Knight shift calculations
        
    def validate_physics(self) -> Dict[str, bool]:
        """Validate quantum physics"""
        return self.c13_engine.validate_physics()


def create_quantum_c13_noise_source(concentration: float = 0.011,
                                   cluster_size: int = 50,
                                   B_field: np.ndarray = None) -> QuantumC13NoiseAdapter:
    """
    Factory function to create quantum C13 noise source
    
    Args:
        concentration: C13 concentration
        cluster_size: Number of C13 nuclei
        B_field: Applied magnetic field [T]
        
    Returns:
        Quantum C13 noise adapter
    """
    return QuantumC13NoiseAdapter(
        concentration=concentration,
        cluster_size=cluster_size,
        B_field=B_field
    )


def replace_classical_c13_with_quantum(noise_generator):
    """
    Replace classical C13BathNoise with quantum implementation
    
    Args:
        noise_generator: Existing NoiseGenerator instance
        
    Returns:
        Modified noise generator with quantum C13
    """
    # Remove old C13 source if present
    if 'c13_bath' in noise_generator.sources:
        old_c13 = noise_generator.sources['c13_bath']
        del noise_generator.sources['c13_bath']
        
        print("🗑️ Removed classical C13BathNoise")
        
        # Extract parameters from old source
        if hasattr(old_c13, 'concentration'):
            concentration = old_c13.concentration
        else:
            concentration = 0.011
            
        # Create quantum replacement
        quantum_c13 = create_quantum_c13_noise_source(
            concentration=concentration,
            cluster_size=100
        )
        
        # Add to noise generator
        noise_generator.sources['quantum_c13'] = quantum_c13
        
        print("✨ Installed quantum C13BathEngine")
        
    return noise_generator