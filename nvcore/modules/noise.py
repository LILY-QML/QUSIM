"""
High-Level Noise Generator for NV Center Simulations

This module provides a unified interface to compose multiple noise sources
and efficiently generate noise for quantum simulations.
"""

import numpy as np
from typing import Dict, List, Optional, Union, Iterator, Callable
from dataclasses import dataclass, field
import warnings

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'helper'))

from noise_sources import (
    SYSTEM,
    NoiseSource,
    C13BathNoise,
    ExternalFieldNoise,
    JohnsonNoise,
    ChargeStateNoise,
    TemperatureFluctuation,
    StrainNoise,
    MicrowaveNoise,
    OpticalNoise
)


@dataclass
class NoiseConfiguration:
    """Simplified configuration for enabling/disabling noise sources"""
    
    # Enable/disable noise sources
    enable_c13_bath: bool = True
    enable_external_field: bool = True
    enable_johnson: bool = True
    enable_charge_noise: bool = True
    enable_temperature: bool = True
    enable_strain: bool = True
    enable_microwave: bool = True
    enable_optical: bool = True
    
    # Simulation parameters (defaults from system.json)
    dt: float = field(default_factory=lambda: SYSTEM.defaults['timestep'])
    seed: Optional[int] = None
    
    # Parameter overrides (optional)
    parameter_overrides: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    @classmethod
    def from_preset(cls, preset_name: str) -> 'NoiseConfiguration':
        """Create configuration from experimental preset"""
        preset = SYSTEM.get_preset(preset_name)
        config = cls()
        
        # Apply preset-specific settings
        if 'temperature' in preset:
            config.parameter_overrides['thermal'] = {'base_temperature': preset['temperature']}
            config.parameter_overrides['johnson'] = {'temperature': preset['temperature']}
            
        if 'depth' in preset:
            config.parameter_overrides['charge_state'] = {'surface_distance': preset['depth']}
            config.parameter_overrides['johnson'] = {'conductor_distance': preset['depth']}
            
        if 'c13_concentration' in preset:
            config.parameter_overrides['c13_bath'] = {'concentration': preset['c13_concentration']}
            
        return config
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'NoiseConfiguration':
        """Create configuration from dictionary"""
        config = cls()
        
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
        return config


class NoiseGenerator:
    """
    High-performance noise generator with streaming capabilities
    
    Composes multiple noise sources and provides efficient generation methods
    """
    
    def __init__(self, config: Optional[NoiseConfiguration] = None):
        """
        Initialize noise generator
        
        Args:
            config: Noise configuration object
        """
        self.config = config or NoiseConfiguration()
        self.rng = np.random.default_rng(self.config.seed)
        
        # Initialize enabled noise sources
        self.sources: Dict[str, NoiseSource] = {}
        self._initialize_sources()
        
        # Cache for performance
        self._spin_operators_cache = None
        
    def _initialize_sources(self):
        """Initialize all enabled noise sources"""
        if self.config.enable_c13_bath:
            override_params = self.config.parameter_overrides.get('c13_bath', {})
            self.sources['c13_bath'] = C13BathNoise(self.rng, override_params)
            
        if self.config.enable_external_field:
            override_params = self.config.parameter_overrides.get('external_field', {})
            self.sources['external_field'] = ExternalFieldNoise(self.rng, override_params)
            
        if self.config.enable_johnson:
            override_params = self.config.parameter_overrides.get('johnson', {})
            self.sources['johnson'] = JohnsonNoise(self.rng, override_params)
            
        if self.config.enable_charge_noise:
            override_params = self.config.parameter_overrides.get('charge_state', {})
            self.sources['charge_state'] = ChargeStateNoise(self.rng, override_params)
            
        if self.config.enable_temperature:
            override_params = self.config.parameter_overrides.get('thermal', {})
            self.sources['temperature'] = TemperatureFluctuation(self.rng, override_params)
            
        if self.config.enable_strain:
            override_params = self.config.parameter_overrides.get('strain', {})
            self.sources['strain'] = StrainNoise(self.rng, override_params)
            
        if self.config.enable_microwave:
            override_params = self.config.parameter_overrides.get('microwave', {})
            self.sources['microwave'] = MicrowaveNoise(self.rng, override_params)
            
        if self.config.enable_optical:
            override_params = self.config.parameter_overrides.get('optical', {})
            self.sources['optical'] = OpticalNoise(self.rng, override_params)
            
        # Set timestep for all sources
        for source in self.sources.values():
            source._dt = self.config.dt
            
    def reset(self):
        """Reset all noise sources to initial state"""
        for source in self.sources.values():
            source.reset()
            
    def set_seed(self, seed: int):
        """Set random seed for reproducibility"""
        self.config.seed = seed
        self.rng = np.random.default_rng(seed)
        self._initialize_sources()
        
    # Efficient streaming interface
    
    def stream_magnetic_noise(self, n_samples: int, 
                             batch_size: int = 1000) -> Iterator[np.ndarray]:
        """
        Stream magnetic field noise samples in batches
        
        Args:
            n_samples: Total number of samples to generate
            batch_size: Samples per batch (for memory efficiency)
            
        Yields:
            Batches of magnetic field vectors in Tesla
        """
        samples_generated = 0
        
        while samples_generated < n_samples:
            current_batch = min(batch_size, n_samples - samples_generated)
            
            # Combine all magnetic noise sources
            b_total = np.zeros((current_batch, 3))
            
            if 'c13_bath' in self.sources:
                b_total += self.sources['c13_bath'].sample(current_batch)
                
            if 'external_field' in self.sources:
                b_total += self.sources['external_field'].sample(current_batch)
                
            if 'johnson' in self.sources:
                # Johnson noise is generated sample by sample (not vectorized)
                for i in range(current_batch):
                    b_total[i] += self.sources['johnson'].sample(1)
                    
            yield b_total
            samples_generated += current_batch
            
    def get_total_magnetic_noise(self, n_samples: int = 1) -> np.ndarray:
        """
        Get total magnetic field noise from all sources
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of shape (n_samples, 3) or (3,) if n_samples=1
        """
        # For small samples, direct generation
        if n_samples <= 1000:
            b_total = np.zeros((n_samples, 3))
            
            if 'c13_bath' in self.sources:
                b_total += self.sources['c13_bath'].sample(n_samples)
                
            if 'external_field' in self.sources:
                b_total += self.sources['external_field'].sample(n_samples)
                
            if 'johnson' in self.sources:
                for i in range(n_samples):
                    b_total[i] += self.sources['johnson'].sample(1)
                    
            return b_total.squeeze() if n_samples == 1 else b_total
            
        else:
            # Use streaming for large samples
            batches = list(self.stream_magnetic_noise(n_samples))
            return np.vstack(batches)
            
    # Hamiltonian and Lindblad interfaces
    
    def get_noise_hamiltonian(self, spin_operators: Dict[str, np.ndarray],
                             include_sources: Optional[List[str]] = None) -> np.ndarray:
        """
        Generate noise Hamiltonian contribution
        
        Args:
            spin_operators: Dictionary with 'Sx', 'Sy', 'Sz' operators
            include_sources: List of sources to include (None = all enabled)
            
        Returns:
            Noise Hamiltonian matrix
        """
        H_noise = np.zeros_like(spin_operators['Sx'])
        
        # Magnetic field contribution
        if include_sources is None or 'magnetic' in include_sources:
            B_noise = self.get_total_magnetic_noise(1)
            H_noise += SYSTEM.get_constant('nv_center', 'gamma_e') * (
                B_noise[0] * spin_operators['Sx'] +
                B_noise[1] * spin_operators['Sy'] +
                B_noise[2] * spin_operators['Sz']
            )
            
        # Strain contribution to zero-field splitting
        if include_sources is None or 'strain' in include_sources:
            if 'strain' in self.sources:
                strain = self.sources['strain'].sample(1)
                delta_d = self.sources['strain'].get_zfs_shift(strain)
                
                # D term: D(Sz^2 - 2/3)
                S_z2 = spin_operators['Sz'] @ spin_operators['Sz']
                dim = S_z2.shape[0]
                H_noise += delta_d * (S_z2 - 2/3 * np.eye(dim))
                
        return H_noise
        
    def get_lindblad_operators(self, spin_operators: Dict[str, np.ndarray],
                              include_sources: Optional[List[str]] = None) -> List[tuple]:
        """
        Generate Lindblad operators for open system dynamics
        
        Args:
            spin_operators: Dictionary with spin operators including 'S+', 'S-', 'Sz'
            include_sources: List of sources to include (None = all enabled)
            
        Returns:
            List of (operator, rate) tuples for Lindblad equation
        """
        lindblad_ops = []
        
        # T1 relaxation from temperature
        if (include_sources is None or 'thermal' in include_sources) and \
           'temperature' in self.sources:
            temp_source = self.sources['temperature']
            current_temp = temp_source.sample(1)
            
            # Calculate relaxation rate
            energy_gap = SYSTEM.get_constant('nv_center', 'd_gs')  # Ground state ZFS
            phonon_n = temp_source.calculate_phonon_occupation(energy_gap)
            
            # Emission rate (S-)
            phonon_coupling = SYSTEM.get_noise_param('thermal', 'temperature_fluctuation', 'phonon_coupling_strength')
            gamma_down = phonon_coupling * energy_gap**3 * (phonon_n + 1)
            
            # Absorption rate (S+)  
            gamma_up = phonon_coupling * energy_gap**3 * phonon_n
                      
            if 'S-' in spin_operators and gamma_down > 0:
                lindblad_ops.append((spin_operators['S-'], np.sqrt(gamma_down)))
                
            if 'S+' in spin_operators and gamma_up > 0:
                lindblad_ops.append((spin_operators['S+'], np.sqrt(gamma_up)))
                
        # Pure dephasing from magnetic noise
        if include_sources is None or 'dephasing' in include_sources:
            # Estimate dephasing rate from magnetic noise spectrum
            # This is a simplified model - real calculation would integrate PSD
            gamma_phi = SYSTEM.defaults['typical_dephasing_rate']
            
            if 'Sz' in spin_operators:
                lindblad_ops.append((spin_operators['Sz'], np.sqrt(gamma_phi)))
                
        return lindblad_ops
        
    # Specialized noise functions
    
    def process_microwave_pulse(self, nominal_rabi: float, duration: float,
                               phase: float = 0.0) -> Dict[str, np.ndarray]:
        """
        Apply noise to microwave pulse parameters
        
        Args:
            nominal_rabi: Intended Rabi frequency in Hz
            duration: Pulse duration in seconds
            phase: Initial phase in radians
            
        Returns:
            Dictionary with noisy pulse parameters vs time
        """
        n_samples = int(duration / self.config.dt)
        
        result = {
            'time': np.arange(n_samples) * self.config.dt,
            'rabi_frequency': np.full(n_samples, nominal_rabi),
            'phase': np.full(n_samples, phase),
            'frequency_offset': np.zeros(n_samples)
        }
        
        if 'microwave' in self.sources:
            mw_source = self.sources['microwave']
            
            # Apply amplitude noise
            amp_factors = mw_source.sample_amplitude_factor(n_samples)
            result['rabi_frequency'] *= amp_factors
            
            # Apply phase noise (cumulative)
            phase_noise = mw_source.sample_phase_noise(n_samples)
            result['phase'] += phase_noise
            
            # Apply frequency drift
            result['frequency_offset'] = mw_source.sample_frequency_offset(n_samples)
            
        return result
        
    def process_optical_readout(self, state_populations: Dict[str, float],
                               readout_duration: float,
                               n_shots: int = 1) -> np.ndarray:
        """
        Simulate noisy optical readout
        
        Args:
            state_populations: Dict mapping state names to populations
            readout_duration: Integration time in seconds
            n_shots: Number of readout repetitions
            
        Returns:
            Array of photon counts for each shot
        """
        if 'optical' not in self.sources:
            warnings.warn("Optical noise not enabled, returning ideal readout")
            # Simple Poisson noise only
            bright_rate = 1e6  # Typical bright state rate
            counts = self.rng.poisson(bright_rate * readout_duration, n_shots)
            return counts
            
        optical = self.sources['optical']
        
        # Define state-dependent photon rates from system.json
        photon_rates = {
            'ms=0': SYSTEM.get_noise_param('optical', 'readout', 'bright_state_rate'),
            'ms=+1': SYSTEM.get_noise_param('optical', 'readout', 'dark_state_rate'),
            'ms=-1': SYSTEM.get_noise_param('optical', 'readout', 'dark_state_rate')
        }
        
        counts = np.zeros(n_shots, dtype=int)
        
        for shot in range(n_shots):
            # Calculate expected signal from state populations
            expected_rate = sum(
                pop * photon_rates.get(state, 0) 
                for state, pop in state_populations.items()
            )
            
            # Apply intensity noise
            intensity_factor = optical.sample_intensity_factor(1)
            noisy_rate = expected_rate * intensity_factor
            
            # Generate photon counts
            counts[shot] = optical.sample_photon_counts(noisy_rate, readout_duration)
            
        return counts
        
    def estimate_t2_star(self, evolution_time: float = 10e-6,
                        n_samples: int = 1000) -> float:
        """
        Estimate T2* from magnetic noise statistics
        
        Uses autocorrelation of magnetic noise to estimate dephasing time
        
        Args:
            evolution_time: Time window for correlation analysis
            n_samples: Number of noise samples
            
        Returns:
            Estimated T2* in seconds
        """
        # Generate magnetic noise trajectory
        b_noise = self.get_total_magnetic_noise(n_samples)
        
        # Focus on z-component (most relevant for dephasing)
        b_z = b_noise[:, 2]
        
        # Calculate autocorrelation
        correlation = np.correlate(b_z - np.mean(b_z), b_z - np.mean(b_z), mode='full')
        correlation = correlation[len(correlation)//2:]
        correlation /= correlation[0]
        
        # Find 1/e decay time
        try:
            decay_idx = np.where(correlation < 1/np.e)[0][0]
            correlation_time = decay_idx * self.config.dt
            
            # T2* ≈ 1 / (γ * σ_B * sqrt(τ_c))
            # where σ_B is RMS field noise and τ_c is correlation time
            sigma_b = np.std(b_z)
            t2_star = 1 / (SYSTEM.get_constant('nv_center', 'gamma_e') * sigma_b * np.sqrt(correlation_time))
            
            return t2_star
            
        except IndexError:
            # Correlation doesn't decay - return lower bound
            return evolution_time
            
    def save_configuration(self, filename: str):
        """Save current noise configuration to file"""
        import json
        
        config_dict = {
            'enable_c13_bath': self.config.enable_c13_bath,
            'enable_external_field': self.config.enable_external_field,
            'enable_johnson': self.config.enable_johnson,
            'enable_charge_noise': self.config.enable_charge_noise,
            'enable_temperature': self.config.enable_temperature,
            'enable_strain': self.config.enable_strain,
            'enable_microwave': self.config.enable_microwave,
            'enable_optical': self.config.enable_optical,
            'dt': self.config.dt,
            'seed': self.config.seed
        }
        
        # Add parameter overrides
        if self.config.parameter_overrides:
            config_dict['parameter_overrides'] = self.config.parameter_overrides
        # ... continue for other parameters
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
            
    @classmethod
    def load_configuration(cls, filename: str) -> 'NoiseGenerator':
        """Load noise generator from saved configuration"""
        import json
        
        with open(filename, 'r') as f:
            config_dict = json.load(f)
            
        config = NoiseConfiguration.from_dict(config_dict)
        return cls(config)


# Convenience functions for common use cases

def create_realistic_noise_generator(temperature: float = 300.0,
                                   magnetic_field: float = 0.0,
                                   sample_depth: float = 10e-9,
                                   c13_concentration: Optional[float] = None) -> NoiseGenerator:
    """
    Create noise generator with realistic parameters
    
    Args:
        temperature: Sample temperature in Kelvin
        magnetic_field: Applied field in Tesla
        sample_depth: NV depth below surface in meters
        c13_concentration: 13C isotope concentration (None = natural)
        
    Returns:
        Configured NoiseGenerator instance
    """
    config = NoiseConfiguration()
    
    # Set temperature-dependent parameters
    config.parameter_overrides['thermal'] = {'base_temperature': temperature}
    config.parameter_overrides['johnson'] = {'temperature': temperature}
    
    # Set depth-dependent parameters
    config.parameter_overrides['charge_state'] = {'surface_distance': sample_depth}
    if 'johnson' not in config.parameter_overrides:
        config.parameter_overrides['johnson'] = {}
    config.parameter_overrides['johnson']['conductor_distance'] = sample_depth
    
    # Set concentration if specified
    if c13_concentration is not None:
        config.parameter_overrides['c13_bath'] = {'concentration': c13_concentration}
        
    # Adjust noise levels based on conditions
    if temperature > 77:  # Room temperature
        if 'thermal' not in config.parameter_overrides:
            config.parameter_overrides['thermal'] = {}
        config.parameter_overrides['thermal']['phonon_coupling_strength'] = 1e-3
    else:  # Cryogenic
        if 'thermal' not in config.parameter_overrides:
            config.parameter_overrides['thermal'] = {}
        config.parameter_overrides['thermal']['phonon_coupling_strength'] = 1e-5
        
    if magnetic_field > 0.01:  # High field
        config.enable_external_field = False  # Less relevant
        
    return NoiseGenerator(config)


def create_low_noise_generator() -> NoiseGenerator:
    """Create noise generator for ideal/low-noise conditions"""
    config = NoiseConfiguration()
    
    # Disable most noise sources
    config.enable_johnson = False
    config.enable_charge_noise = False
    config.enable_external_field = False
    
    # Reduce remaining noise levels
    config.parameter_overrides = {
        'c13_bath': {'concentration': 1e-4},  # Isotopically pure
        'thermal': {'base_temperature': 4.0},  # Cryogenic
        'microwave': {'amplitude_noise': 1e-4},
        'optical': {'laser_rin': 1e-5}
    }
    
    return NoiseGenerator(config)