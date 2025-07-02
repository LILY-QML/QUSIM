"""
Noise Interface for QUSIM Core

Provides clean interface between NV quantum mechanics and noise sources.
Enables pluggable noise systems while maintaining ultra-realistic physics.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class NoiseInterface(ABC):
    """
    Abstract interface for all noise systems in QUSIM
    
    This interface ensures clean separation between:
    - Pure NV quantum mechanics (core)
    - Realistic noise modeling (modules)
    
    All noise implementations must provide:
    1. Magnetic field noise (Tesla)
    2. Hamiltonian noise contributions  
    3. Lindblad dissipation operators
    4. Power spectral densities for validation
    """
    
    @abstractmethod
    def get_magnetic_field_noise(self, n_samples: int = 1) -> np.ndarray:
        """
        Generate magnetic field noise samples
        
        Args:
            n_samples: Number of noise samples to generate
            
        Returns:
            Array of shape (n_samples, 3) or (3,) if n_samples=1
            Magnetic field noise in Tesla [Bx, By, Bz]
        """
        pass
    
    @abstractmethod
    def get_hamiltonian_noise(self, spin_operators: Dict[str, np.ndarray], 
                             t: float = 0.0) -> np.ndarray:
        """
        Generate noise contribution to Hamiltonian
        
        Args:
            spin_operators: Dictionary with 'Sx', 'Sy', 'Sz' operators
            t: Current time (for time-dependent noise)
            
        Returns:
            Noise Hamiltonian matrix (same shape as spin operators)
        """
        pass
    
    @abstractmethod
    def get_lindblad_operators(self, spin_operators: Dict[str, np.ndarray],
                              include_sources: Optional[List[str]] = None) -> List[Tuple[np.ndarray, float]]:
        """
        Generate Lindblad operators for open system dynamics
        
        Args:
            spin_operators: Dictionary with spin operators (must include 'Sz', may include 'S+', 'S-')
            include_sources: List of noise sources to include (None = all)
            
        Returns:
            List of (operator, sqrt_rate) tuples for Lindblad equation:
            dρ/dt = Σᵢ γᵢ(LᵢρLᵢ† - ½{Lᵢ†Lᵢ, ρ})
            where sqrt_rate = √γᵢ
        """
        pass
    
    @abstractmethod
    def get_noise_power_spectral_density(self, frequencies: np.ndarray,
                                       component: str = 'total') -> np.ndarray:
        """
        Get noise power spectral density for validation
        
        Args:
            frequencies: Frequency array [Hz]  
            component: Which component ('magnetic', 'electric', 'total')
            
        Returns:
            Power spectral density [T²/Hz for magnetic, etc.]
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset all noise sources to initial state"""
        pass
    
    @abstractmethod
    def set_parameters(self, **kwargs):
        """Update noise parameters dynamically"""
        pass
    
    # Optional methods for advanced functionality
    
    def get_correlation_functions(self, tau_values: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get noise correlation functions for analysis
        
        Args:
            tau_values: Time delay values [s]
            
        Returns:
            Dictionary mapping noise type to correlation function
        """
        return {}
    
    def estimate_coherence_times(self) -> Dict[str, float]:
        """
        Estimate T1, T2*, T2 from noise statistics
        
        Returns:
            Dictionary with coherence time estimates [s]
        """
        return {}
    
    def validate_physics(self) -> Dict[str, bool]:
        """
        Validate that noise implementation follows physical constraints
        
        Returns:
            Dictionary of validation results
        """
        return {}


class NoiseGeneratorAdapter(NoiseInterface):
    """
    Adapter to use existing NoiseGenerator with new interface
    
    This adapter allows the existing ultra-realistic noise system
    to work with the new modular architecture without changes.
    """
    
    def __init__(self, noise_generator):
        """
        Initialize adapter with existing NoiseGenerator
        
        Args:
            noise_generator: Instance of modules.noise.NoiseGenerator
        """
        from modules.noise import NoiseGenerator
        
        if not isinstance(noise_generator, NoiseGenerator):
            raise TypeError("noise_generator must be NoiseGenerator instance")
            
        self.noise_gen = noise_generator
        self._last_time = 0.0
        
    def get_magnetic_field_noise(self, n_samples: int = 1) -> np.ndarray:
        """Generate magnetic field noise using existing NoiseGenerator"""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        return self.noise_gen.get_total_magnetic_noise(n_samples)
    
    def get_hamiltonian_noise(self, spin_operators: Dict[str, np.ndarray], 
                             t: float = 0.0) -> np.ndarray:
        """Generate Hamiltonian noise using existing NoiseGenerator"""
        required_ops = {'Sx', 'Sy', 'Sz'}
        if not required_ops.issubset(spin_operators.keys()):
            raise ValueError(f"spin_operators must contain: {required_ops}")
            
        # Update time if needed
        if t != self._last_time:
            dt = t - self._last_time
            if dt > 0:
                for source in self.noise_gen.sources.values():
                    source.update_time(dt)
            self._last_time = t
            
        return self.noise_gen.get_noise_hamiltonian(spin_operators)
    
    def get_lindblad_operators(self, spin_operators: Dict[str, np.ndarray],
                              include_sources: Optional[List[str]] = None) -> List[Tuple[np.ndarray, float]]:
        """Generate Lindblad operators using existing NoiseGenerator"""
        if 'Sz' not in spin_operators:
            raise ValueError("spin_operators must contain 'Sz' for Lindblad operators")
            
        return self.noise_gen.get_lindblad_operators(spin_operators, include_sources)
    
    def get_noise_power_spectral_density(self, frequencies: np.ndarray,
                                       component: str = 'total') -> np.ndarray:
        """Get PSD using existing NoiseGenerator"""
        if component == 'magnetic' or component == 'total':
            return self.noise_gen.get_magnetic_noise_psd(frequencies)
        else:
            raise ValueError(f"Component '{component}' not supported by NoiseGenerator")
    
    def reset(self):
        """Reset using existing NoiseGenerator"""
        self.noise_gen.reset()
        self._last_time = 0.0
    
    def set_parameters(self, **kwargs):
        """Set parameters by updating noise generator configuration"""
        # Update noise generator config
        for param, value in kwargs.items():
            if hasattr(self.noise_gen.config, param):
                setattr(self.noise_gen.config, param, value)
            else:
                # Try updating parameter overrides
                if not hasattr(self.noise_gen.config, 'parameter_overrides'):
                    self.noise_gen.config.parameter_overrides = {}
                    
                # Parse parameter name (e.g., 'c13_bath.concentration')
                if '.' in param:
                    source, subparam = param.split('.', 1)
                    if source not in self.noise_gen.config.parameter_overrides:
                        self.noise_gen.config.parameter_overrides[source] = {}
                    self.noise_gen.config.parameter_overrides[source][subparam] = value
        
        # Reinitialize sources with new parameters
        self.noise_gen._initialize_sources()
    
    # Enhanced functionality using existing NoiseGenerator capabilities
    
    def get_correlation_functions(self, tau_values: np.ndarray) -> Dict[str, np.ndarray]:
        """Get correlation functions from noise samples"""
        correlation_funcs = {}
        
        # Generate long noise trajectory for correlation analysis
        n_samples = len(tau_values) * 10  # Oversample for good statistics
        noise_samples = self.get_magnetic_field_noise(n_samples)
        
        for component, name in enumerate(['Bx', 'By', 'Bz']):
            signal = noise_samples[:, component]
            signal_centered = signal - np.mean(signal)
            
            # Compute autocorrelation
            correlation = np.correlate(signal_centered, signal_centered, mode='full')
            correlation = correlation[len(correlation)//2:][:len(tau_values)]
            correlation /= correlation[0]  # Normalize
            
            correlation_funcs[name] = correlation
            
        return correlation_funcs
    
    def estimate_coherence_times(self) -> Dict[str, float]:
        """Estimate coherence times using existing NoiseGenerator"""
        estimates = {}
        
        try:
            # Use existing T2* estimation method
            t2_star = self.noise_gen.estimate_t2_star()
            estimates['T2_star'] = t2_star
        except:
            estimates['T2_star'] = np.nan
            
        # Basic estimates from noise levels
        try:
            b_noise = self.get_magnetic_field_noise(1000)
            b_rms = np.sqrt(np.mean(b_noise**2))
            
            # Rough estimates (order of magnitude)
            from helper.noise_sources import SYSTEM
            gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
            
            estimates['T2_rough'] = 1 / (gamma_e * np.sqrt(np.mean(b_rms**2)))
            estimates['magnetic_noise_level'] = float(np.sqrt(np.mean(b_rms**2)))
        except:
            estimates['T2_rough'] = np.nan
            estimates['magnetic_noise_level'] = np.nan
            
        return estimates
    
    def validate_physics(self) -> Dict[str, bool]:
        """Validate physics using existing NoiseGenerator"""
        validation = {}
        
        try:
            # Test that noise is finite and reasonable
            b_noise = self.get_magnetic_field_noise(100)
            validation['finite_noise'] = np.all(np.isfinite(b_noise))
            validation['reasonable_amplitude'] = np.all(np.abs(b_noise) < 1.0)  # < 1 Tesla
            
            # Test PSD is positive
            freqs = np.logspace(0, 6, 100)
            psd = self.get_noise_power_spectral_density(freqs)
            validation['positive_psd'] = np.all(psd >= 0)
            
            # Test reset works
            original_noise = self.get_magnetic_field_noise(1)
            self.reset()
            reset_noise = self.get_magnetic_field_noise(1)
            validation['reset_functional'] = True  # If no exception
            
        except Exception as e:
            validation['error'] = str(e)
            validation['finite_noise'] = False
            validation['reasonable_amplitude'] = False
            validation['positive_psd'] = False
            validation['reset_functional'] = False
            
        return validation


# Factory functions for common use cases

def create_realistic_noise_interface(temperature: float = 300.0,
                                   c13_concentration: float = 0.011,
                                   magnetic_field: np.ndarray = None) -> NoiseInterface:
    """
    Create realistic noise interface with common parameters
    
    Args:
        temperature: Sample temperature [K]
        c13_concentration: 13C isotope concentration
        magnetic_field: Applied magnetic field [T]
        
    Returns:
        NoiseInterface configured for realistic conditions
    """
    from modules.noise import NoiseGenerator, NoiseConfiguration
    
    config = NoiseConfiguration()
    config.parameter_overrides = {
        'thermal': {'base_temperature': temperature},
        'c13_bath': {'concentration': c13_concentration},
        'johnson': {'temperature': temperature}
    }
    
    if magnetic_field is not None:
        config.parameter_overrides['c13_bath']['b_field'] = magnetic_field
    
    noise_gen = NoiseGenerator(config)
    return NoiseGeneratorAdapter(noise_gen)


def create_minimal_noise_interface() -> NoiseInterface:
    """
    Create minimal noise interface for testing
    
    Returns:
        NoiseInterface with minimal realistic noise
    """
    from modules.noise import NoiseGenerator, NoiseConfiguration
    
    config = NoiseConfiguration()
    # Enable only essential noise sources
    config.enable_external_field = False
    config.enable_johnson = False
    config.enable_charge_noise = False
    config.enable_strain = False
    config.enable_microwave = False
    config.enable_optical = False
    # Keep C13 bath and temperature for basic realism
    
    noise_gen = NoiseGenerator(config)
    return NoiseGeneratorAdapter(noise_gen)