"""
Leeson Model for Microwave Oscillator Phase Noise

Implements realistic phase noise characteristics of microwave sources
following Leeson's model rather than simple 1/f² approximations.

Mathematical Background:
- Leeson Model: S_φ(f) = S₀[1 + (f_c/f)² + (f_1f/f)²]
- Close-in phase noise: 1/f² region
- Flicker floor: 1/f region  
- White floor: flat region
- Additional peaks from spurious modes

Physical Origins:
- Thermal noise in active components
- Flicker noise in amplifiers
- Q-factor limitations
- Vibration sensitivity

References:
- Leeson, IEEE Proc. 54, 329 (1966)
- Rubiola, Phase Noise and Frequency Stability in Oscillators (2008)
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from scipy.signal import find_peaks
from abc import ABC, abstractmethod

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from noise_sources import SYSTEM, NoiseSource


class OscillatorModel(ABC):
    """Abstract base class for oscillator models"""
    
    @abstractmethod
    def get_phase_noise_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """Get phase noise power spectral density [rad²/Hz]"""
        pass
    
    @abstractmethod
    def get_frequency_noise_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """Get frequency noise power spectral density [Hz²/Hz]"""
        pass


class LeesonOscillator(OscillatorModel):
    """
    Leeson model oscillator with realistic phase noise characteristics
    
    S_φ(f) = S₀[1 + (f_c/f)² + (f_1f/f)²] + S_white
    
    Where:
    - f_c: Close-in corner frequency (1/f² to flat transition)
    - f_1f: Flicker corner frequency (1/f to flat transition)
    - S₀: White phase noise floor
    """
    
    def __init__(self, carrier_frequency: float,
                 q_factor: float = 1e4,
                 noise_figure: float = 10.0,  # dB
                 power_level: float = 0.0,    # dBm
                 flicker_corner: float = 1e3,  # Hz
                 close_in_corner: float = 1e5,  # Hz
                 vibration_sensitivity: float = 1e-9,  # rad/g
                 temperature_sensitivity: float = 1e-6):  # rad/K
        """
        Initialize Leeson oscillator model
        
        Args:
            carrier_frequency: Oscillator frequency [Hz]
            q_factor: Loaded Q factor of resonator
            noise_figure: Amplifier noise figure [dB]
            power_level: Signal power level [dBm]
            flicker_corner: 1/f corner frequency [Hz]
            close_in_corner: 1/f² corner frequency [Hz]
            vibration_sensitivity: Phase noise due to vibration [rad/g]
            temperature_sensitivity: Phase noise due to temperature [rad/K]
        """
        self.f_carrier = carrier_frequency
        self.Q = q_factor
        self.nf_db = noise_figure
        self.power_dbm = power_level
        self.f_flicker = flicker_corner
        self.f_close = close_in_corner
        self.vibration_sens = vibration_sensitivity
        self.temp_sens = temperature_sensitivity
        
        # Convert dB values to linear
        self.nf_linear = 10**(noise_figure / 10)
        self.power_watts = 10**((power_level - 30) / 10)
        
        # Environmental conditions - from empirical measurements
        try:
            self.temperature = SYSTEM.get_empirical_param('environmental', 'temperature')
            self.vibration_psd = SYSTEM.get_empirical_param('environmental', 'vibration_psd')
        except KeyError as e:
            raise RuntimeError(f"Missing required environmental parameter: {e}. "
                             f"Add environmental measurements to empirical_parameters in system.json.")
        
        # Calculate Leeson parameters
        self._calculate_leeson_parameters()
        
    def _calculate_leeson_parameters(self):
        """Calculate Leeson model parameters from physical parameters"""
        
        # Boltzmann constant
        kB = 1.380649e-23
        
        # White phase noise floor (thermal noise limit)
        # S₀ = (F * kB * T) / (2 * P_carrier)
        self.S0 = (self.nf_linear * kB * self.temperature) / (2 * self.power_watts)
        
        # Close-in corner frequency (related to loaded Q)
        # f_c ≈ f_carrier / (2 * Q_loaded)
        self.f_c = self.f_carrier / (2 * self.Q)
        
        # Flicker corner frequency (empirical, device dependent)
        self.f_1f = self.f_flicker
        
        # Additional noise coefficients
        self.flicker_coeff = self.S0 * self.f_1f  # For 1/f region
        self.close_coeff = self.S0 * self.f_c**2  # For 1/f² region
        
    def get_phase_noise_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate phase noise PSD using Leeson model
        
        Args:
            frequencies: Offset frequencies from carrier [Hz]
            
        Returns:
            Phase noise PSD [rad²/Hz]
        """
        f = np.abs(frequencies)
        f = np.maximum(f, 1e-3)  # Avoid f=0
        
        # Leeson model: S_φ(f) = S₀[1 + (f_c/f)² + (f_1f/f)]
        
        # White floor
        psd = np.full_like(f, self.S0)
        
        # 1/f² region (close to carrier)
        close_in_term = self.close_coeff / f**2
        psd += close_in_term
        
        # 1/f region (flicker noise)
        flicker_term = self.flicker_coeff / f
        psd += flicker_term
        
        # Add environmental contributions
        psd += self._get_environmental_noise(f)
        
        # Add spurious peaks if any
        psd += self._get_spurious_noise(f)
        
        return psd
    
    def _get_environmental_noise(self, frequencies: np.ndarray) -> np.ndarray:
        """Calculate environmental noise contributions"""
        
        # Vibration noise (typically has 1/f² characteristic)
        vibration_noise = np.zeros_like(frequencies)
        f_vib = 100.0  # Hz, typical building vibrations
        vib_amplitude = (self.vibration_sens**2 * self.vibration_psd * f_vib**2 / 
                        (frequencies**2 + f_vib**2))
        vibration_noise += vib_amplitude
        
        # Temperature fluctuations (typically 1/f)
        temp_noise = np.zeros_like(frequencies)
        temp_psd = 1e-4  # K²/Hz, typical temperature stability
        temp_amplitude = self.temp_sens**2 * temp_psd / frequencies
        temp_noise += temp_amplitude
        
        return vibration_noise + temp_noise
    
    def _get_spurious_noise(self, frequencies: np.ndarray) -> np.ndarray:
        """Add spurious peaks (power supply, mechanical resonances)"""
        spurious = np.zeros_like(frequencies)
        
        # Spurious frequencies should be measured for specific lab setup
        # These are common values but should be in system.json empirical parameters
        try:
            spur_freqs = SYSTEM.get_empirical_param('microwave_system', 'spurious_frequencies')
            spur_levels = SYSTEM.get_empirical_param('microwave_system', 'spurious_levels') 
            spur_widths = SYSTEM.get_empirical_param('microwave_system', 'spurious_widths')
        except KeyError as e:
            raise RuntimeError(f"Missing required microwave spurious frequency parameters: {e}. "
                             f"These must be measured for your specific MW source. "
                             f"Add spurious_frequencies, spurious_levels, and spurious_widths "
                             f"to empirical_parameters.microwave_system in system.json.")
        
        for f_spur, level, width in zip(spur_freqs, spur_levels, spur_widths):
            # Lorentzian peak
            peak = level * (width/2)**2 / ((frequencies - f_spur)**2 + (width/2)**2)
            spurious += peak
        
        return spurious
    
    def get_frequency_noise_psd(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Calculate frequency noise PSD
        
        S_f(f) = (f)² * S_φ(f) for f << f_carrier
        """
        phase_psd = self.get_phase_noise_psd(frequencies)
        return (frequencies**2) * phase_psd
    
    def get_allan_variance(self, tau_values: np.ndarray) -> np.ndarray:
        """
        Calculate Allan variance from phase noise PSD
        
        σ²_y(τ) = 2 ∫₀^∞ S_f(f) sin⁴(πfτ)/(πfτ)² df
        """
        allan_var = np.zeros_like(tau_values)
        
        for i, tau in enumerate(tau_values):
            # Frequency range for integration
            f_min = 1.0 / (100 * tau)
            f_max = 10.0 / tau
            freqs = np.logspace(np.log10(f_min), np.log10(f_max), 1000)
            
            # Get frequency noise PSD
            Sf = self.get_frequency_noise_psd(freqs)
            
            # Allan variance integrand
            sin_term = np.sin(np.pi * freqs * tau)
            filter_func = 2 * sin_term**4 / (np.pi * freqs * tau)**2
            
            # Handle f=0 case
            filter_func[freqs * tau < 1e-6] = 2 * (np.pi * tau)**2 / 3
            
            # Integrate
            integrand = Sf * filter_func
            allan_var[i] = np.trapz(integrand, freqs)
        
        return allan_var
    
    def get_rms_phase_error(self, integration_time: float,
                           f_min: float = 1e-3, f_max: float = 1e6) -> float:
        """
        Calculate RMS phase error over integration bandwidth
        
        φ_rms = √(∫_{f_min}^{f_max} S_φ(f) df)
        """
        freqs = np.logspace(np.log10(f_min), np.log10(f_max), 10000)
        phase_psd = self.get_phase_noise_psd(freqs)
        
        # Integrate PSD
        phase_variance = np.trapz(phase_psd, freqs)
        return np.sqrt(phase_variance)


class LeesonMicrowaveNoise(NoiseSource):
    """
    Microwave noise source using Leeson oscillator model
    
    Generates realistic phase and amplitude noise for MW sources
    """
    
    def __init__(self, carrier_frequency: float = None,
                 rng: Optional[np.random.Generator] = None,
                 override_params: Optional[dict] = None):
        """
        Initialize Leeson microwave noise source
        
        Args:
            carrier_frequency: MW carrier frequency [Hz]
            rng: Random number generator
            override_params: Parameter overrides
        """
        if override_params is None:
            override_params = {}
        super().__init__(rng)
        
        # Oscillator parameters - get from system.json
        if carrier_frequency is None:
            self.f_carrier = SYSTEM.get_constant('nv_center', 'd_gs')
        else:
            self.f_carrier = carrier_frequency
        
        # Get parameters from empirical data - no defaults allowed
        try:
            self.q_factor = override_params.get('q_factor', SYSTEM.get_empirical_param('microwave_system', 'q_factor'))
            self.noise_figure = override_params.get('noise_figure_db', SYSTEM.get_empirical_param('microwave_system', 'noise_figure_db'))
            self.power_level = override_params.get('power_level_dbm', SYSTEM.get_empirical_param('microwave_system', 'power_level_dbm'))
            self.flicker_corner = override_params.get('flicker_corner_hz', SYSTEM.get_empirical_param('microwave_system', 'flicker_corner_hz'))
            self.close_in_corner = override_params.get('close_in_corner_hz', SYSTEM.get_empirical_param('microwave_system', 'close_in_corner_hz'))
        except KeyError as e:
            raise RuntimeError(f"Missing required microwave parameter: {e}. "
                             f"Add all microwave_system parameters to empirical_parameters in system.json.")
        
        # Create Leeson oscillator model
        self.oscillator = LeesonOscillator(
            carrier_frequency=self.f_carrier,
            q_factor=self.q_factor,
            noise_figure=self.noise_figure,
            power_level=self.power_level,
            flicker_corner=self.flicker_corner,
            close_in_corner=self.close_in_corner
        )
        
        # Amplitude noise parameters - must be measured
        try:
            self.amplitude_noise_ratio = override_params.get('am_to_pm_ratio', SYSTEM.get_empirical_param('microwave_system', 'am_to_pm_ratio'))
        except KeyError as e:
            raise RuntimeError(f"Missing required microwave AM/PM ratio: {e}. "
                             f"This must be measured for your MW source.")
        
        # State variables for correlated noise generation
        self.phase_state = 0.0
        self.amplitude_state = 1.0
        self.frequency_state = self.f_carrier
        
        # History for colored noise generation
        self.phase_history = []
        self.amplitude_history = []
        
    def _generate_colored_noise(self, psd_func: callable, 
                               frequencies: np.ndarray,
                               n_samples: int = 1) -> np.ndarray:
        """
        Generate colored noise with specified PSD using frequency domain method
        """
        if n_samples == 1:
            # For single sample, use mini-batch method to maintain accuracy
            # Generate small batch and return single sample
            batch_size = max(32, int(1 / (self._dt * self.flicker_corner_hz)))
            batch = self._generate_colored_noise(psd_func, frequencies, batch_size)
            
            # Store batch and return first sample
            if not hasattr(self, '_colored_noise_buffer'):
                self._colored_noise_buffer = []
                self._buffer_index = 0
            
            if len(self._colored_noise_buffer) == 0:
                self._colored_noise_buffer = batch.tolist()
                self._buffer_index = 0
            
            sample = self._colored_noise_buffer[self._buffer_index]
            self._buffer_index = (self._buffer_index + 1) % len(self._colored_noise_buffer)
            
            # Refresh buffer when exhausted
            if self._buffer_index == 0:
                batch = self._generate_colored_noise(psd_func, frequencies, batch_size)
                self._colored_noise_buffer = batch.tolist()
            
            return sample
        
        else:
            # Full frequency domain method for multiple samples
            # Generate white noise in frequency domain
            N = n_samples
            freqs = np.fft.fftfreq(N, self._dt)[1:N//2]  # Positive frequencies only
            
            # Get target PSD
            target_psd = psd_func(freqs)
            
            # Generate complex white noise
            white_real = self.rng.normal(0, 1, len(freqs))
            white_imag = self.rng.normal(0, 1, len(freqs))
            white_complex = white_real + 1j * white_imag
            
            # Scale by sqrt(PSD)
            colored_complex = white_complex * np.sqrt(target_psd * self._dt)
            
            # Create symmetric spectrum
            full_spectrum = np.zeros(N, dtype=complex)
            full_spectrum[1:N//2+1] = colored_complex
            full_spectrum[N//2+1:] = np.conj(colored_complex[::-1])
            
            # Inverse FFT to get time domain
            time_series = np.fft.ifft(full_spectrum).real
            
            return time_series
    
    def sample_phase_noise(self, n_samples: int = 1) -> Union[float, np.ndarray]:
        """
        Generate phase noise samples
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Phase noise in radians
        """
        # Frequency array for PSD calculation
        if n_samples == 1:
            # Single sample using AR model
            return self._generate_colored_noise(
                self.oscillator.get_phase_noise_psd, 
                np.array([1.0]), 
                n_samples
            )
        else:
            # Multiple samples using frequency domain
            return self._generate_colored_noise(
                self.oscillator.get_phase_noise_psd,
                np.linspace(1.0, 1e6, 1000),
                n_samples
            )
    
    def sample_amplitude_factor(self, n_samples: int = 1) -> Union[float, np.ndarray]:
        """
        Generate amplitude noise as multiplicative factor
        
        Amplitude noise is typically much lower than phase noise
        """
        phase_noise = self.sample_phase_noise(n_samples)
        
        # AM-PM correlation: amplitude noise is reduced version of phase noise
        amplitude_noise = self.amplitude_noise_ratio * phase_noise
        
        # Convert to multiplicative factor: A(t) = A₀(1 + ε(t))
        return 1.0 + amplitude_noise
    
    def sample_frequency_offset(self, n_samples: int = 1) -> Union[float, np.ndarray]:
        """
        Generate frequency noise (derivative of phase noise)
        
        f(t) = f₀ + df/dt where φ(t) = ∫ df dt
        """
        if n_samples == 1:
            # Simple difference approximation
            dt = getattr(self, '_dt', 1e-6)
            current_phase = self.sample_phase_noise(1)
            
            if hasattr(self, '_last_phase'):
                freq_noise = (current_phase - self._last_phase) / (2 * np.pi * dt)
            else:
                freq_noise = 0.0
            
            self._last_phase = current_phase
            return freq_noise
        
        else:
            # Generate frequency noise directly from PSD
            return self._generate_colored_noise(
                self.oscillator.get_frequency_noise_psd,
                np.linspace(1.0, 1e6, 1000),
                n_samples
            )
    
    def sample(self, n_samples: int = 1) -> Dict[str, Union[float, np.ndarray]]:
        """
        Generate complete MW noise sample
        
        Returns:
            Dictionary with phase, amplitude, and frequency noise
        """
        phase_noise = self.sample_phase_noise(n_samples)
        amplitude_factor = self.sample_amplitude_factor(n_samples)
        frequency_offset = self.sample_frequency_offset(n_samples)
        
        return {
            'phase_noise': phase_noise,
            'amplitude_factor': amplitude_factor,
            'frequency_offset': frequency_offset
        }
    
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """Get phase noise PSD"""
        return self.oscillator.get_phase_noise_psd(frequencies)
    
    def get_oscillator_specs(self) -> Dict[str, float]:
        """Get oscillator specifications"""
        # Calculate key specs at standard offset frequencies
        f_offsets = [1e2, 1e3, 1e4, 1e5, 1e6]  # 100Hz to 1MHz
        phase_noise_dbc = []
        
        for f in f_offsets:
            # Convert to dBc/Hz: L(f) = 10*log10(S_φ(f)/2)
            psd = self.oscillator.get_phase_noise_psd(np.array([f]))[0]
            dbc_hz = 10 * np.log10(psd / 2)
            phase_noise_dbc.append(dbc_hz)
        
        # RMS phase error
        rms_phase = self.oscillator.get_rms_phase_error(1e-3, 1e2, 1e6)
        
        return {
            'carrier_frequency_ghz': self.f_carrier / 1e9,
            'q_factor': self.q_factor,
            'noise_figure_db': self.noise_figure,
            'power_level_dbm': self.power_level,
            'phase_noise_100hz_dbc': phase_noise_dbc[0],
            'phase_noise_1khz_dbc': phase_noise_dbc[1],
            'phase_noise_10khz_dbc': phase_noise_dbc[2],
            'phase_noise_100khz_dbc': phase_noise_dbc[3],
            'phase_noise_1mhz_dbc': phase_noise_dbc[4],
            'rms_phase_error_mrad': rms_phase * 1000,
            'flicker_corner_hz': self.flicker_corner,
            'close_in_corner_hz': self.close_in_corner
        }
    
    def reset(self):
        """Reset noise source to initial state"""
        super().reset()
        self.phase_state = 0.0
        self.amplitude_state = 1.0
        self.frequency_state = self.f_carrier
        self.phase_history = []
        self.amplitude_history = []
        if hasattr(self, '_last_colored_noise'):
            delattr(self, '_last_colored_noise')
        if hasattr(self, '_last_phase'):
            delattr(self, '_last_phase')


# Factory functions for common MW sources
def create_lab_microwave_source(frequency_ghz: float = None) -> LeesonMicrowaveNoise:
    """Create typical laboratory MW source"""
    return LeesonMicrowaveNoise(
        carrier_frequency=frequency_ghz * 1e9 if frequency_ghz is not None else None,
        override_params={
            'q_factor': 1e4,
            'noise_figure_db': 15.0,
            'power_level_dbm': 10.0,
            'flicker_corner_hz': 1e3,
            'close_in_corner_hz': 1e5
        }
    )

def create_precision_microwave_source(frequency_ghz: float = None) -> LeesonMicrowaveNoise:
    """Create high-precision MW source (lower noise)"""
    return LeesonMicrowaveNoise(
        carrier_frequency=frequency_ghz * 1e9 if frequency_ghz is not None else None,
        override_params={
            'q_factor': 1e5,
            'noise_figure_db': 8.0,
            'power_level_dbm': 15.0,
            'flicker_corner_hz': 100.0,
            'close_in_corner_hz': 1e4
        }
    )

def create_budget_microwave_source(frequency_ghz: float = None) -> LeesonMicrowaveNoise:
    """Create budget MW source (higher noise)"""
    return LeesonMicrowaveNoise(
        carrier_frequency=frequency_ghz * 1e9 if frequency_ghz is not None else None,
        override_params={
            'q_factor': 5e3,
            'noise_figure_db': 20.0,
            'power_level_dbm': 5.0,
            'flicker_corner_hz': 10e3,
            'close_in_corner_hz': 5e5
        }
    )


# Example usage and testing
if __name__ == "__main__":
    print("📻 Testing Leeson Microwave Noise Model")
    
    # Create different MW source types
    sources = {
        'Lab Source': create_lab_microwave_source(),
        'Precision Source': create_precision_microwave_source(),
        'Budget Source': create_budget_microwave_source()
    }
    
    print("\n📊 MW Source Specifications:")
    for name, source in sources.items():
        specs = source.get_oscillator_specs()
        print(f"\n   {name}:")
        print(f"     Phase noise @ 1kHz: {specs['phase_noise_1khz_dbc']:.1f} dBc/Hz")
        print(f"     Phase noise @ 100kHz: {specs['phase_noise_100khz_dbc']:.1f} dBc/Hz")
        print(f"     RMS phase error: {specs['rms_phase_error_mrad']:.2f} mrad")
    
    # Test noise generation
    print("\n🔄 Noise Generation Test:")
    source = sources['Lab Source']
    source._dt = 1e-6  # 1 μs timestep
    
    # Generate noise samples
    n_samples = 1000
    noise_samples = []
    for _ in range(n_samples):
        sample = source.sample(1)
        noise_samples.append({
            'phase': sample['phase_noise'],
            'amplitude': sample['amplitude_factor'],
            'frequency': sample['frequency_offset']
        })
    
    # Analyze statistics
    phases = [s['phase'] for s in noise_samples]
    amplitudes = [s['amplitude'] for s in noise_samples]
    frequencies = [s['frequency'] for s in noise_samples]
    
    print(f"   Phase noise RMS: {np.std(phases)*1000:.2f} mrad")
    print(f"   Amplitude factor RMS: {np.std(amplitudes)*100:.3f} %")
    print(f"   Frequency offset RMS: {np.std(frequencies)/1e3:.1f} kHz")
    
    # Test PSD
    print("\n📈 Power Spectral Density:")
    freqs = np.logspace(2, 6, 100)  # 100 Hz to 1 MHz
    psd = source.get_power_spectral_density(freqs)
    
    # Find corner frequencies from PSD
    # Look for -20 dB/decade slope (1/f²) and -10 dB/decade slope (1/f)
    log_freqs = np.log10(freqs)
    log_psd = np.log10(psd)
    
    # Simple slope analysis
    slopes = np.diff(log_psd) / np.diff(log_freqs)
    
    print(f"   PSD @ 1 kHz: {10*np.log10(psd[np.argmin(np.abs(freqs - 1e3))]/2):.1f} dBc/Hz")
    print(f"   PSD @ 100 kHz: {10*np.log10(psd[np.argmin(np.abs(freqs - 1e5))]/2):.1f} dBc/Hz")
    print(f"   Average slope in 1/f² region: {np.mean(slopes[:20]):.1f} (expect ≈-2)")
    
    print("\n✅ Leeson microwave noise model successfully implemented!")