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
from filter_functions import FilterFunctionCalculator
from charge_dynamics import MultiLevelChargeNoise, create_charge_state_model
from strain_tensor import StrainTensorNoise, create_bulk_diamond_strain, create_nanodiamond_strain, create_surface_nv_strain
from non_markovian import NonMarkovianBath, create_c13_non_markovian_bath, create_phonon_non_markovian_bath, create_charge_non_markovian_bath
from leeson_microwave import LeesonMicrowaveNoise, create_lab_microwave_source, create_precision_microwave_source


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
    
    # Phase 2 improvements
    enable_tensor_strain: bool = False  # Advanced strain model
    enable_multi_level_charge: bool = False  # Advanced charge dynamics
    enable_non_markovian: bool = False  # Memory effects
    enable_leeson_microwave: bool = False  # Advanced MW model
    
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
        
        # Filter function calculator
        self.filter_calc = FilterFunctionCalculator()
        
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
            
        # Phase 2 advanced implementations
        if self.config.enable_multi_level_charge:
            override_params = self.config.parameter_overrides.get('multi_level_charge', {})
            # Replace simple charge noise with multi-level model
            if 'charge_state' in self.sources:
                del self.sources['charge_state']
            self.sources['multi_level_charge'] = MultiLevelChargeNoise(self.rng, override_params)
            
        if self.config.enable_tensor_strain:
            override_params = self.config.parameter_overrides.get('tensor_strain', {})
            # Replace simple strain with tensor model
            if 'strain' in self.sources:
                del self.sources['strain'] 
            self.sources['tensor_strain'] = StrainTensorNoise(self.rng, override_params)
            
        if self.config.enable_leeson_microwave:
            override_params = self.config.parameter_overrides.get('leeson_microwave', {})
            # Replace simple MW noise with Leeson model
            if 'microwave' in self.sources:
                del self.sources['microwave']
            carrier_freq = override_params.get('carrier_frequency', SYSTEM.get_constant('nv_center', 'd_gs'))
            self.sources['leeson_microwave'] = LeesonMicrowaveNoise(carrier_freq, self.rng, override_params)
            
        if self.config.enable_non_markovian:
            override_params = self.config.parameter_overrides.get('non_markovian', {})
            bath_type = override_params.get('bath_type', 'c13')
            
            if bath_type == 'c13':
                concentration = override_params.get('c13_concentration', 0.011)
                self.sources['non_markovian_c13'] = create_c13_non_markovian_bath(concentration)
            elif bath_type == 'phonon':
                temperature = override_params.get('temperature', 300.0)
                self.sources['non_markovian_phonon'] = create_phonon_non_markovian_bath(temperature)
            elif bath_type == 'charge':
                depth = override_params.get('depth_nm', 10.0)
                self.sources['non_markovian_charge'] = create_charge_non_markovian_bath(depth)
            
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
                
            elif 'tensor_strain' in self.sources:
                # Advanced tensor strain model
                H_strain = self.sources['tensor_strain'].get_hamiltonian_perturbation(spin_operators)
                H_noise += H_strain
                
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
                
        # Pure dephasing from magnetic noise - ULTRA REALISTIC CALCULATION
        if include_sources is None or 'dephasing' in include_sources:
            # Calculate dephasing rate from actual magnetic noise power spectral density
            # Integrate γ_φ = γ_e² ∫ S_B(ω) dω over relevant frequency range
            try:
                # Define frequency range relevant for dephasing (DC to ~100 MHz)
                frequencies = np.logspace(-1, 8, 2000)  # 0.1 Hz to 100 MHz, high resolution
                omega = 2 * np.pi * frequencies
                
                # Get total magnetic noise PSD from all enabled sources
                noise_psd = self.get_magnetic_noise_psd(frequencies)
                
                # Calculate dephasing rate: γ_φ = γ_e² ∫ S_B(ω) dω
                # Focus on z-component which dominates pure dephasing
                gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
                
                # Integrate PSD over frequency (using trapezoid rule for accuracy)
                # Factor of 2 accounts for positive/negative frequency contributions
                gamma_phi = 2 * gamma_e**2 * np.trapz(noise_psd, frequencies)
                
                # Ensure minimum numerical stability (avoid zero dephasing)
                gamma_phi = max(gamma_phi, 1e3)  # Minimum 1 kHz dephasing rate
                
            except Exception as e:
                # Only fall back if PSD calculation completely fails
                import warnings
                warnings.warn(f"PSD integration failed ({e}), using conservative estimate")
                
                # Conservative estimate based on typical room temperature magnetic noise
                gamma_phi = 1e6  # 1 MHz as absolute fallback
            
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
            
        elif 'leeson_microwave' in self.sources:
            # Advanced Leeson model
            mw_source = self.sources['leeson_microwave']
            
            # Generate comprehensive noise samples
            for i in range(n_samples):
                noise_sample = mw_source.sample(1)
                result['rabi_frequency'][i] *= noise_sample['amplitude_factor']
                result['phase'][i] += noise_sample['phase_noise']
                result['frequency_offset'][i] = noise_sample['frequency_offset']
            
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
            raise RuntimeError("Optical noise source required for realistic readout simulation. "
                             "Enable optical noise in NoiseConfiguration.")
            
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
            
    def get_magnetic_noise_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Get total magnetic noise power spectral density
        
        Args:
            frequencies: Frequency array [Hz]
            
        Returns:
            Total magnetic noise PSD [T²/Hz]
        """
        omega = 2 * np.pi * frequencies
        total_psd = np.zeros_like(frequencies)
        
        # Add contribution from each magnetic noise source
        if 'c13_bath' in self.sources:
            total_psd += self.sources['c13_bath'].get_power_spectral_density(omega)
            
        if 'external_field' in self.sources:
            total_psd += self.sources['external_field'].get_power_spectral_density(omega)
            
        if 'johnson' in self.sources:
            total_psd += self.sources['johnson'].get_power_spectral_density(omega)
            
        return total_psd
        
    def calculate_t2_for_sequence(self, sequence_type: str, 
                                 frequencies: np.ndarray = None,
                                 **sequence_params) -> float:
        """
        Calculate T2 for specific pulse sequence using filter functions
        
        Args:
            sequence_type: Type of pulse sequence ('ramsey', 'echo', 'cpmg', etc.)
            frequencies: Frequency array [Hz] (default: auto-generate)
            **sequence_params: Sequence-specific parameters
            
        Returns:
            T2 time in seconds
        """
        # Generate frequency array if not provided
        if frequencies is None:
            frequencies = np.logspace(-3, 6, 1000)  # 1 mHz to 1 MHz
        
        # Get magnetic noise spectrum
        noise_psd = self.get_magnetic_noise_psd(frequencies)
        
        # Calculate T2 using filter functions
        t2 = self.filter_calc.calculate_t2_from_spectrum(
            sequence_type, noise_psd, frequencies, **sequence_params
        )
        
        return t2
        
    def predict_sequence_performance(self, sequence_configs: Dict[str, Dict]) -> Dict[str, float]:
        """
        Predict T2 times for multiple pulse sequences
        
        Args:
            sequence_configs: Dict of {sequence_name: {params}}
            
        Returns:
            Dict of {sequence_name: t2_time}
        """
        frequencies = np.logspace(-3, 6, 1000)
        results = {}
        
        for seq_name, params in sequence_configs.items():
            try:
                t2 = self.calculate_t2_for_sequence(seq_name, frequencies, **params)
                results[seq_name] = t2
            except Exception as e:
                print(f"Warning: Could not calculate T2 for {seq_name}: {e}")
                results[seq_name] = np.nan
                
        return results
        
    def optimize_sequence_parameters(self, sequence_type: str, 
                                   param_ranges: Dict[str, np.ndarray],
                                   frequencies: np.ndarray = None) -> Dict[str, float]:
        """
        Find optimal parameters for given sequence type
        
        Args:
            sequence_type: Pulse sequence type
            param_ranges: Dict of {param_name: array_of_values}
            frequencies: Frequency array for calculation
            
        Returns:
            Dict with optimal parameters and resulting T2
        """
        if frequencies is None:
            frequencies = np.logspace(-3, 6, 1000)
            
        best_t2 = 0
        best_params = {}
        
        # Grid search over parameter space
        param_names = list(param_ranges.keys())
        param_arrays = list(param_ranges.values())
        
        # Generate all combinations
        from itertools import product
        for param_combo in product(*param_arrays):
            params = dict(zip(param_names, param_combo))
            
            try:
                t2 = self.calculate_t2_for_sequence(sequence_type, frequencies, **params)
                if t2 > best_t2:
                    best_t2 = t2
                    best_params = params.copy()
            except:
                continue
                
        best_params['optimal_t2'] = best_t2
        return best_params
            
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


def create_cryogenic_low_noise_generator(temperature: float = 4.0,
                                        c13_concentration: float = 1e-4) -> NoiseGenerator:
    """
    Create noise generator for cryogenic low-noise conditions with realistic parameters.
    
    Args:
        temperature: Cryogenic temperature in Kelvin (must be < 77K)
        c13_concentration: Measured isotopic purification level
        
    Returns:
        NoiseGenerator with experimentally achievable low-noise parameters
    """
    if temperature >= 77:
        raise ValueError("Temperature must be < 77K for cryogenic operation")
    if c13_concentration < 1e-5:
        raise ValueError("C13 concentration below 1e-5 is not experimentally achievable")
        
    config = NoiseConfiguration()
    
    # Use empirically validated low-noise parameters
    config.parameter_overrides = {
        'c13_bath': {'concentration': c13_concentration},
        'thermal': {'base_temperature': temperature},
        'microwave': {
            'amplitude_noise': SYSTEM.get_empirical_param('microwave_system', 'mw_amplitude_stability') * 0.1
        },
        'optical': {
            'laser_rin': SYSTEM.get_empirical_param('optical_system', 'laser_rin') * 0.1
        },
        'johnson': {'temperature': temperature}  # Keep Johnson noise but at low temp
    }
    
    # Disable environmental noise sources that can be controlled
    config.enable_external_field = False  # Magnetic shielding
    config.enable_charge_noise = False    # Good surface treatment
    
    return NoiseGenerator(config)


def create_advanced_realistic_generator(nv_type: str = 'bulk',
                                      temperature: float = 300.0,
                                      enable_memory_effects: bool = True) -> NoiseGenerator:
    """
    Create advanced noise generator with Phase 2 improvements
    
    Args:
        nv_type: Type of NV ('bulk', 'nanodiamond', 'surface')
        temperature: Operating temperature [K]
        enable_memory_effects: Include non-Markovian effects
        
    Returns:
        NoiseGenerator with advanced models enabled
    """
    config = NoiseConfiguration()
    
    # Enable Phase 2 advanced models
    config.enable_tensor_strain = True
    config.enable_multi_level_charge = True
    config.enable_leeson_microwave = True
    if enable_memory_effects:
        config.enable_non_markovian = True
    
    # Configure based on NV type
    if nv_type == 'bulk':
        config.parameter_overrides = {
            'tensor_strain': {},  # Use defaults for bulk diamond
            'multi_level_charge': {
                'setup_type': 'room_temperature' if temperature > 77 else 'cryogenic'
            },
            'leeson_microwave': {
                'q_factor': 1e4,
                'noise_figure_db': 12.0
            },
            'non_markovian': {
                'bath_type': 'c13',
                'c13_concentration': 0.011
            }
        }
    elif nv_type == 'nanodiamond':
        config.parameter_overrides = {
            'tensor_strain': {
                'strain_amplitude': 1e-6,  # Higher strain
                'correlation_time': 1e-4,   # Faster dynamics
                'resonance_frequency': 1000.0
            },
            'multi_level_charge': {
                'setup_type': 'surface_nv',
                'surface_distance': 5e-9
            },
            'leeson_microwave': {
                'q_factor': 5e3,  # Lower Q
                'noise_figure_db': 15.0
            }
        }
    elif nv_type == 'surface':
        config.parameter_overrides = {
            'tensor_strain': {
                'strain_amplitude': 2e-6,  # Highest strain
                'correlation_time': 1e-3,
                'resonance_frequency': 200.0,
                'static_strain_tensor': np.array([
                    [1e-5, 0, 0],
                    [0, 1e-5, 0], 
                    [0, 0, -2e-5]
                ])
            },
            'multi_level_charge': {
                'setup_type': 'surface_nv',
                'surface_distance': 2e-9,
                'electric_field': 1e5
            },
            'non_markovian': {
                'bath_type': 'charge',
                'depth_nm': 5.0
            }
        }
    
    # Temperature-dependent parameters
    config.parameter_overrides['thermal'] = {'base_temperature': temperature}
    config.parameter_overrides['johnson'] = {'temperature': temperature}
    
    return NoiseGenerator(config)


def create_precision_experiment_generator() -> NoiseGenerator:
    """Create noise generator optimized for precision experiments"""
    config = NoiseConfiguration()
    
    # Enable all advanced models for maximum realism
    config.enable_tensor_strain = True
    config.enable_multi_level_charge = True
    config.enable_leeson_microwave = True
    config.enable_non_markovian = True
    
    # Use precision MW source
    config.parameter_overrides = {
        'leeson_microwave': {
            'q_factor': 1e5,
            'noise_figure_db': 8.0,
            'power_level_dbm': 15.0,
            'flicker_corner_hz': 100.0
        },
        'tensor_strain': {
            'strain_amplitude': 1e-7,  # Low strain
            'correlation_time': 10e-3
        },
        'multi_level_charge': {
            'setup_type': 'cryogenic'
        },
        'non_markovian': {
            'bath_type': 'c13',
            'c13_concentration': 0.001  # Isotopically purified
        },
        'thermal': {'base_temperature': 4.0},  # Cryogenic
        'optical': {'laser_rin': 1e-6}  # Low RIN laser
    }
    
    # Disable some environmental noise
    config.enable_external_field = False
    config.enable_johnson = False
    
    return NoiseGenerator(config)