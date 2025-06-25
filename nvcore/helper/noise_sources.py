"""
Modular Noise Sources for NV Center Simulations

Each noise source is implemented as a separate class with a common interface.
All parameters are loaded from system.json configuration.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union, Optional, Dict, Any
import numpy as np
from scipy import signal
import json
import os


class SystemConfig:
    """Load and access system configuration from system.json"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to system.json in nvcore directory
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'system.json')
            
        with open(config_path, 'r') as f:
            self._config = json.load(f)
            
        # Create convenient accessors
        self.constants = self._config['physical_constants']
        self.noise_params = self._config['noise_parameters']
        self.defaults = self._config['simulation_defaults']
        self.presets = self._config['experimental_presets']
        
    def get_constant(self, category: str, name: str) -> float:
        """Get a physical constant value"""
        return self.constants[category][name]
    
    def get_noise_param(self, category: str, subcategory: str, name: str) -> float:
        """Get a noise parameter value"""
        return self.noise_params[category][subcategory][name]
    
    def get_preset(self, preset_name: str) -> Dict[str, Any]:
        """Get an experimental preset configuration"""
        return self.presets[preset_name]


# Global system configuration instance
SYSTEM = SystemConfig()


class NoiseSource(ABC):
    """Abstract base class for all noise sources"""
    
    def __init__(self, rng: Optional[np.random.Generator] = None):
        """
        Initialize noise source with random number generator
        
        Args:
            rng: NumPy random generator for reproducibility
        """
        self.rng = rng or np.random.default_rng()
        self._time = 0.0
        self._dt = SYSTEM.defaults['timestep']  # Default timestep from config
        
    @abstractmethod
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate noise samples"""
        pass
    
    @abstractmethod
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """Return theoretical PSD for this noise source"""
        pass
    
    def update_time(self, dt: float):
        """Update internal time by dt"""
        self._time += dt
        self._dt = dt
        
    def reset(self):
        """Reset noise source to initial state"""
        self._time = 0.0


# Magnetic Noise Sources

class C13BathNoise(NoiseSource):
    """
    13C nuclear spin bath noise using Ornstein-Uhlenbeck process
    
    Models the quasi-static magnetic field from nuclear spin flip-flops
    """
    
    def __init__(self, rng: Optional[np.random.Generator] = None, override_params: Optional[dict] = None):
        if override_params is None:
            override_params = {}
        super().__init__(rng)
        
        # Load parameters from system.json or use overrides
        self.concentration = (override_params.get('concentration') if override_params and 'concentration' in override_params
                            else SYSTEM.get_noise_param('magnetic', 'c13_bath', 'concentration'))
        self.correlation_time = (override_params.get('correlation_time') if override_params and 'correlation_time' in override_params
                               else SYSTEM.get_noise_param('magnetic', 'c13_bath', 'correlation_time'))
        self.coupling_strength = (override_params.get('coupling_strength') if override_params and 'coupling_strength' in override_params
                                else SYSTEM.get_noise_param('magnetic', 'c13_bath', 'coupling_strength'))
        
        self._state = np.zeros(3)  # Current magnetic field state
        
        # Calculate noise strength from concentration
        self.sigma = np.sqrt(self.concentration) * self.coupling_strength
        
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate magnetic field noise samples [Bx, By, Bz] in Tesla"""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        tau = self.correlation_time
        samples = np.zeros((n_samples, 3))
        
        for i in range(n_samples):
            # Ornstein-Uhlenbeck update
            decay = np.exp(-self._dt / tau)
            noise_term = self.sigma * np.sqrt(2 * self._dt / tau) * self.rng.standard_normal(3)
            self._state = decay * self._state + np.sqrt(1 - decay**2) * noise_term
            samples[i] = self._state
            
        return samples.squeeze() if n_samples == 1 else samples
    
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """Lorentzian PSD for Ornstein-Uhlenbeck process"""
        tau = self.correlation_time
        return (2 * self.sigma**2 * tau) / (1 + (2 * np.pi * frequencies * tau)**2)
    
    def reset(self):
        """Reset to initial state"""
        super().reset()
        self._state = np.zeros(3)


class ExternalFieldNoise(NoiseSource):
    """External magnetic field fluctuations with 1/f^α spectrum"""
    
    def __init__(self, rng: Optional[np.random.Generator] = None, override_params: Optional[dict] = None):
        if override_params is None:
            override_params = {}
        super().__init__(rng)
        
        self.drift_amplitude = (override_params.get('drift_amplitude') if override_params and 'drift_amplitude' in override_params
                              else SYSTEM.get_noise_param('magnetic', 'external_field', 'drift_amplitude'))
        self.noise_exponent = (override_params.get('noise_exponent') if override_params and 'noise_exponent' in override_params
                             else SYSTEM.get_noise_param('magnetic', 'external_field', 'noise_exponent'))
        self.high_freq_cutoff = (override_params.get('high_freq_cutoff') if override_params and 'high_freq_cutoff' in override_params
                               else SYSTEM.get_noise_param('magnetic', 'external_field', 'high_freq_cutoff'))
        
        self._buffer = None
        self._buffer_pos = 0
        self._buffer_size = SYSTEM.defaults['batch_size']
        
    def _generate_colored_noise(self, n_samples: int, alpha: float) -> np.ndarray:
        """Generate 1/f^α colored noise using spectral method"""
        white = self.rng.standard_normal((n_samples, 3))
        freqs = np.fft.fftfreq(n_samples, self._dt)
        
        mask = (freqs != 0) & (np.abs(freqs) < self.high_freq_cutoff)
        f_filter = np.ones_like(freqs)
        f_filter[mask] = 1 / np.abs(freqs[mask])**(alpha / 2)
        f_filter[~mask] = 0
        
        fft_white = np.fft.fft(white, axis=0)
        fft_filtered = fft_white * f_filter[:, np.newaxis]
        colored = np.real(np.fft.ifft(fft_filtered, axis=0))
        
        colored *= self.drift_amplitude / np.std(colored)
        return colored
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate external field noise samples in Tesla"""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        if self._buffer is None or self._buffer_pos + n_samples > len(self._buffer):
            self._buffer = self._generate_colored_noise(self._buffer_size, self.noise_exponent)
            self._buffer_pos = 0
            
        samples = self._buffer[self._buffer_pos:self._buffer_pos + n_samples]
        self._buffer_pos += n_samples
        
        return samples.squeeze() if n_samples == 1 else samples
    
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """Power spectral density ∝ 1/f^α"""
        psd = np.zeros_like(frequencies)
        mask = (frequencies != 0) & (frequencies < self.high_freq_cutoff)
        psd[mask] = self.drift_amplitude**2 / np.abs(frequencies[mask])**self.noise_exponent
        return psd


class JohnsonNoise(NoiseSource):
    """Magnetic Johnson noise from thermal fluctuations in nearby conductors"""
    
    def __init__(self, rng: Optional[np.random.Generator] = None, override_params: Optional[dict] = None):
        if override_params is None:
            override_params = {}
        super().__init__(rng)
        
        self.temperature = (override_params.get('temperature') if override_params and 'temperature' in override_params 
                           else SYSTEM.get_noise_param('magnetic', 'johnson', 'temperature'))
        self.conductor_distance = (override_params.get('conductor_distance') if override_params and 'conductor_distance' in override_params
                                  else SYSTEM.get_noise_param('magnetic', 'johnson', 'conductor_distance'))
        self.conductor_resistivity = (override_params.get('conductor_resistivity') if override_params and 'conductor_resistivity' in override_params
                                     else SYSTEM.get_noise_param('magnetic', 'johnson', 'conductor_resistivity'))
        self.conductor_thickness = (override_params.get('conductor_thickness') if override_params and 'conductor_thickness' in override_params
                                   else SYSTEM.get_noise_param('magnetic', 'johnson', 'conductor_thickness'))
        
    def _calculate_noise_amplitude(self) -> float:
        """Calculate RMS magnetic field from Johnson noise"""
        mu_0 = SYSTEM.get_constant('fundamental', 'mu_0')
        kb = SYSTEM.get_constant('fundamental', 'kb')
        
        prefactor = np.sqrt(mu_0 * kb * self.temperature * self.conductor_resistivity)
        distance_factor = self.conductor_distance**1.5
        thickness_factor = np.sqrt(self.conductor_thickness)
        
        return prefactor / (np.sqrt(np.pi) * distance_factor * np.sqrt(thickness_factor))
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate Johnson noise magnetic field samples in Tesla"""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        amplitude = self._calculate_noise_amplitude()
        samples = amplitude * self.rng.standard_normal((n_samples, 3))
        
        return samples.squeeze() if n_samples == 1 else samples
    
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """White noise PSD"""
        amplitude = self._calculate_noise_amplitude()
        return np.full_like(frequencies, amplitude**2)


# Electric and Charge Noise Sources

class ChargeStateNoise(NoiseSource):
    """Stochastic charge state transitions between NV- and NV0"""
    
    def __init__(self, rng: Optional[np.random.Generator] = None, override_params: Optional[dict] = None):
        if override_params is None:
            override_params = {}
        super().__init__(rng)
        
        self.jump_rate = (override_params.get('jump_rate') if override_params and 'jump_rate' in override_params
                        else SYSTEM.get_noise_param('electric', 'charge_state', 'jump_rate'))
        self.laser_power = (override_params.get('laser_power') if override_params and 'laser_power' in override_params
                          else SYSTEM.get_noise_param('electric', 'charge_state', 'laser_power'))
        self.surface_distance = (override_params.get('surface_distance') if override_params and 'surface_distance' in override_params
                               else SYSTEM.get_noise_param('electric', 'charge_state', 'surface_distance'))
        
        self._current_state = -1  # Start in NV- state
        self._last_flip_time = 0.0
        
    def _get_effective_rate(self) -> float:
        """Calculate effective transition rate including laser effects"""
        base_rate = self.jump_rate
        laser_factor = 1 + 0.1 * self.laser_power  # Empirical scaling
        surface_factor = np.exp(-self.surface_distance / 10e-9)
        return base_rate * laser_factor * surface_factor
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate charge state samples (-1 for NV-, 0 for NV0)"""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        rate = self._get_effective_rate()
        samples = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            if self.rng.random() < rate * self._dt:
                self._current_state = 0 if self._current_state == -1 else -1
                self._last_flip_time = self._time
                
            samples[i] = self._current_state
            self.update_time(self._dt)
            
        return samples[0] if n_samples == 1 else samples
    
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """Telegraph noise PSD"""
        rate = self._get_effective_rate()
        return (4 * rate) / (1 + (2 * np.pi * frequencies / rate)**2)
    
    def reset(self):
        """Reset to initial NV- state"""
        super().reset()
        self._current_state = -1
        self._last_flip_time = 0.0


# Thermal Noise Sources

class TemperatureFluctuation(NoiseSource):
    """Temperature fluctuations affecting phonon populations and relaxation rates"""
    
    def __init__(self, rng: Optional[np.random.Generator] = None, override_params: Optional[dict] = None):
        if override_params is None:
            override_params = {}
        super().__init__(rng)
        
        self.base_temperature = (override_params.get('base_temperature') if override_params and 'base_temperature' in override_params
                               else SYSTEM.get_noise_param('thermal', 'temperature_fluctuation', 'base_temperature'))
        self.fluctuation_amplitude = (override_params.get('fluctuation_amplitude') if override_params and 'fluctuation_amplitude' in override_params
                                     else SYSTEM.get_noise_param('thermal', 'temperature_fluctuation', 'fluctuation_amplitude'))
        self.correlation_time = (override_params.get('correlation_time') if override_params and 'correlation_time' in override_params
                               else SYSTEM.get_noise_param('thermal', 'temperature_fluctuation', 'correlation_time'))
        
        self._current_temp = self.base_temperature
        
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate temperature samples in Kelvin"""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        tau = self.correlation_time
        sigma = self.fluctuation_amplitude
        samples = np.zeros(n_samples)
        
        for i in range(n_samples):
            decay = np.exp(-self._dt / tau)
            noise = sigma * np.sqrt(2 * self._dt / tau) * self.rng.standard_normal()
            
            self._current_temp = (
                self.base_temperature + 
                decay * (self._current_temp - self.base_temperature) +
                np.sqrt(1 - decay**2) * noise
            )
            samples[i] = self._current_temp
            
        return samples[0] if n_samples == 1 else samples
    
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """Lorentzian PSD for temperature fluctuations"""
        tau = self.correlation_time
        sigma = self.fluctuation_amplitude
        return (2 * sigma**2 * tau) / (1 + (2 * np.pi * frequencies * tau)**2)
    
    def calculate_phonon_occupation(self, energy_hz: float) -> float:
        """Calculate phonon occupation number for given energy"""
        if energy_hz <= 0:
            return 0.0
            
        h = SYSTEM.get_constant('fundamental', 'h')
        kb = SYSTEM.get_constant('fundamental', 'kb')
        
        energy_j = h * energy_hz
        thermal_energy = kb * self._current_temp
        
        return 1 / (np.exp(energy_j / thermal_energy) - 1)
    
    def reset(self):
        """Reset to base temperature"""
        super().reset()
        self._current_temp = self.base_temperature


# Mechanical Noise Sources

class StrainNoise(NoiseSource):
    """Mechanical strain affecting the NV center through crystal field"""
    
    def __init__(self, rng: Optional[np.random.Generator] = None, override_params: Optional[dict] = None):
        if override_params is None:
            override_params = {}
        super().__init__(rng)
        
        self.static_strain = (override_params.get('static_strain') if override_params and 'static_strain' in override_params
                            else SYSTEM.get_noise_param('mechanical', 'strain', 'static_strain'))
        self.dynamic_amplitude = (override_params.get('dynamic_amplitude') if override_params and 'dynamic_amplitude' in override_params
                                else SYSTEM.get_noise_param('mechanical', 'strain', 'dynamic_amplitude'))
        self.oscillation_frequency = (override_params.get('oscillation_frequency') if override_params and 'oscillation_frequency' in override_params
                                     else SYSTEM.get_noise_param('mechanical', 'strain', 'oscillation_frequency'))
        self.random_amplitude = (override_params.get('random_amplitude') if override_params and 'random_amplitude' in override_params
                               else SYSTEM.get_noise_param('mechanical', 'strain', 'random_amplitude'))
        self.strain_coupling = (override_params.get('strain_coupling') if override_params and 'strain_coupling' in override_params
                              else SYSTEM.get_noise_param('mechanical', 'strain', 'strain_coupling'))
        
        self._phase = 0.0
        
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Generate strain samples (dimensionless)"""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        samples = np.zeros(n_samples)
        
        for i in range(n_samples):
            self._phase += 2 * np.pi * self.oscillation_frequency * self._dt
            dynamic = self.dynamic_amplitude * np.sin(self._phase)
            random = self.random_amplitude * self.rng.standard_normal()
            
            samples[i] = self.static_strain + dynamic + random
            
        return samples[0] if n_samples == 1 else samples
    
    def get_zfs_shift(self, strain: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Convert strain to zero-field splitting shift"""
        return self.strain_coupling * strain
    
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """PSD with delta function at oscillation frequency + white noise"""
        psd = np.full_like(frequencies, self.random_amplitude**2)
        
        freq_idx = np.argmin(np.abs(frequencies - self.oscillation_frequency))
        if freq_idx < len(frequencies):
            psd[freq_idx] += self.dynamic_amplitude**2 / 2
            
        return psd
    
    def reset(self):
        """Reset oscillation phase"""
        super().reset()
        self._phase = 0.0


# Microwave Noise Sources

class MicrowaveNoise(NoiseSource):
    """Noise in microwave control fields"""
    
    def __init__(self, rng: Optional[np.random.Generator] = None, override_params: Optional[dict] = None):
        if override_params is None:
            override_params = {}
        super().__init__(rng)
        
        self.amplitude_noise = (override_params.get('amplitude_noise') if override_params and 'amplitude_noise' in override_params
                              else SYSTEM.get_noise_param('microwave', 'noise', 'amplitude_noise'))
        self.phase_noise = (override_params.get('phase_noise') if override_params and 'phase_noise' in override_params
                          else SYSTEM.get_noise_param('microwave', 'noise', 'phase_noise'))
        self.frequency_drift = (override_params.get('frequency_drift') if override_params and 'frequency_drift' in override_params
                              else SYSTEM.get_noise_param('microwave', 'noise', 'frequency_drift'))
        self.drift_time_constant = (override_params.get('drift_time_constant') if override_params and 'drift_time_constant' in override_params
                                  else SYSTEM.get_noise_param('microwave', 'noise', 'drift_time_constant'))
        
        self._phase_accumulation = 0.0
        self._frequency_offset = 0.0
        
    def sample_amplitude_factor(self, n_samples: int = 1) -> np.ndarray:
        """Generate multiplicative amplitude noise factors"""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        sigma = self.amplitude_noise
        factors = np.exp(sigma * self.rng.standard_normal(n_samples))
        
        return factors[0] if n_samples == 1 else factors
    
    def sample_phase_noise(self, n_samples: int = 1) -> np.ndarray:
        """Generate phase noise samples in radians"""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        samples = np.zeros(n_samples)
        phase_increment_std = self.phase_noise * np.sqrt(self._dt)
        
        for i in range(n_samples):
            self._phase_accumulation += phase_increment_std * self.rng.standard_normal()
            samples[i] = self._phase_accumulation
            
        return samples[0] if n_samples == 1 else samples
    
    def sample_frequency_offset(self, n_samples: int = 1) -> np.ndarray:
        """Generate frequency offset samples in Hz"""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        tau = self.drift_time_constant
        samples = np.zeros(n_samples)
        
        for i in range(n_samples):
            decay = np.exp(-self._dt / tau)
            noise = self.frequency_drift * np.sqrt(2 * self._dt / tau) * self.rng.standard_normal()
            
            self._frequency_offset = decay * self._frequency_offset + np.sqrt(1 - decay**2) * noise
            samples[i] = self._frequency_offset
            
        return samples[0] if n_samples == 1 else samples
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Default sample method (returns amplitude factors)"""
        return self.sample_amplitude_factor(n_samples)
    
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """Combined PSD for MW noise"""
        phase_psd = np.zeros_like(frequencies)
        nonzero = frequencies != 0
        phase_psd[nonzero] = self.phase_noise**2 / frequencies[nonzero]**2
        
        tau = self.drift_time_constant  
        freq_psd = (2 * self.frequency_drift**2 * tau) / (1 + (2 * np.pi * frequencies * tau)**2)
        
        return phase_psd + freq_psd
    
    def reset(self):
        """Reset accumulated phase and frequency offset"""
        super().reset()
        self._phase_accumulation = 0.0
        self._frequency_offset = 0.0


# Optical Noise Sources

class OpticalNoise(NoiseSource):
    """Noise in optical pumping and readout"""
    
    def __init__(self, rng: Optional[np.random.Generator] = None, override_params: Optional[dict] = None):
        if override_params is None:
            override_params = {}
        super().__init__(rng)
        
        self.laser_rin = (override_params.get('laser_rin') if override_params and 'laser_rin' in override_params
                        else SYSTEM.get_noise_param('optical', 'readout', 'laser_rin'))
        self.rin_corner_frequency = (override_params.get('rin_corner_frequency') if override_params and 'rin_corner_frequency' in override_params
                                   else SYSTEM.get_noise_param('optical', 'readout', 'rin_corner_frequency'))
        self.detector_dark_rate = (override_params.get('detector_dark_rate') if override_params and 'detector_dark_rate' in override_params
                                 else SYSTEM.get_noise_param('optical', 'readout', 'detector_dark_rate'))
        self.detector_efficiency = (override_params.get('detector_efficiency') if override_params and 'detector_efficiency' in override_params
                                  else SYSTEM.get_noise_param('optical', 'readout', 'detector_efficiency'))
        self.readout_fidelity = (override_params.get('readout_fidelity') if override_params and 'readout_fidelity' in override_params
                               else SYSTEM.get_noise_param('optical', 'readout', 'readout_fidelity'))
        
        self._intensity_factor = 1.0
        
    def sample_intensity_factor(self, n_samples: int = 1) -> np.ndarray:
        """Generate laser intensity noise factors"""
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        white = self.rng.standard_normal(n_samples)
        factors = np.zeros(n_samples)
        tau = 1 / (2 * np.pi * self.rin_corner_frequency)
        decay = np.exp(-self._dt / tau)
        
        for i in range(n_samples):
            noise = self.laser_rin * white[i]
            self._intensity_factor = (
                1.0 + decay * (self._intensity_factor - 1.0) + 
                np.sqrt(1 - decay**2) * noise
            )
            factors[i] = max(0.1, self._intensity_factor)
            
        return factors[0] if n_samples == 1 else factors
    
    def sample_photon_counts(self, expected_signal: float, integration_time: float) -> int:
        """Generate realistic photon counts including all noise sources"""
        detected_signal_rate = expected_signal * self.detector_efficiency
        total_rate = detected_signal_rate + self.detector_dark_rate
        expected_counts = total_rate * integration_time
        
        if expected_counts > 0:
            return self.rng.poisson(expected_counts)
        else:
            return 0
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Default sample method (returns intensity factors)"""
        return self.sample_intensity_factor(n_samples)
    
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """RIN spectrum: 1/f at low frequencies, white at high frequencies"""
        f_c = self.rin_corner_frequency
        psd = self.laser_rin**2 * f_c / (frequencies + f_c)
        return psd
    
    def reset(self):
        """Reset intensity factor"""
        super().reset()
        self._intensity_factor = 1.0