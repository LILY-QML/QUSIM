"""
Filter Functions for Pulse Sequence Dependent Decoherence

Implements filter functions F(ω,t) for various pulse sequences to calculate
sequence-specific T2 times via: Γ_φ = γ_e² ∫ S_B(ω) F(ω) dω

Mathematical Background:
- Free precession: F(ω,t) = 4sin²(ωt/2)/ω²
- Spin echo: F(ω,τ) = 8sin⁴(ωτ/4)/ω²  
- CPMG: F(ω,τ,N) = (sin(Nωτ/2)/sin(ωτ/2))² × sin⁴(ωτ/4)/ω²

References:
- Cywiński et al., Phys. Rev. B 77, 174509 (2008)
- Biercuk et al., Nature 458, 996 (2009)
"""

import numpy as np
from typing import Dict, Any, Union
from abc import ABC, abstractmethod


class FilterFunction(ABC):
    """Abstract base class for filter functions"""
    
    @abstractmethod
    def evaluate(self, frequencies: np.ndarray, **params) -> np.ndarray:
        """Evaluate filter function at given frequencies"""
        pass
    
    @abstractmethod
    def get_required_params(self) -> list:
        """Return list of required parameters"""
        pass


class FreePrecessionFilter(FilterFunction):
    """
    Filter function for free precession (Ramsey experiment)
    
    F(ω,t) = 4sin²(ωt/2)/ω²
    
    This gives the sensitivity to noise at frequency ω during 
    free evolution time t.
    """
    
    def evaluate(self, frequencies: np.ndarray, evolution_time: float) -> np.ndarray:
        """
        Calculate free precession filter function
        
        Args:
            frequencies: Angular frequencies in rad/s
            evolution_time: Free evolution time in seconds
            
        Returns:
            Filter function values
        """
        # Handle ω=0 case separately to avoid division by zero
        omega_t_half = frequencies * evolution_time / 2
        
        # For small ω, use Taylor expansion: sin(x)/x ≈ 1 - x²/6
        mask_small = np.abs(omega_t_half) < 1e-6
        mask_large = ~mask_small
        
        result = np.zeros_like(frequencies)
        
        # Small frequency approximation
        if np.any(mask_small):
            x = omega_t_half[mask_small]
            sinc_approx = 1 - x**2/6 + x**4/120  # Taylor expansion of sinc
            result[mask_small] = (2 * evolution_time * sinc_approx)**2
            
        # Large frequency exact calculation
        if np.any(mask_large):
            omega = frequencies[mask_large]
            sinc_term = np.sin(omega * evolution_time / 2) / (omega / 2)
            result[mask_large] = sinc_term**2
            
        return result
    
    def get_required_params(self) -> list:
        return ['evolution_time']


class SpinEchoFilter(FilterFunction):
    """
    Filter function for spin echo (π-pulse at τ/2)
    
    F(ω,τ) = 8sin⁴(ωτ/4)/ω²
    
    The π-pulse at τ/2 refocuses low-frequency noise but maintains
    sensitivity to high-frequency noise.
    """
    
    def evaluate(self, frequencies: np.ndarray, echo_time: float) -> np.ndarray:
        """
        Calculate spin echo filter function
        
        Args:
            frequencies: Angular frequencies in rad/s
            echo_time: Total echo time (2τ) in seconds
            
        Returns:
            Filter function values
        """
        omega_tau_quarter = frequencies * echo_time / 4
        
        # Handle small frequency case
        mask_small = np.abs(omega_tau_quarter) < 1e-6
        mask_large = ~mask_small
        
        result = np.zeros_like(frequencies)
        
        # Small frequency: F ≈ (ωτ/2)⁴/12
        if np.any(mask_small):
            x = omega_tau_quarter[mask_small]
            result[mask_small] = 8 * (x**4) / 3  # Approximation for small ω
            
        # Large frequency exact calculation
        if np.any(mask_large):
            omega = frequencies[mask_large]
            sin_term = np.sin(omega * echo_time / 4)
            result[mask_large] = 8 * sin_term**4 / omega**2
            
        return result
    
    def get_required_params(self) -> list:
        return ['echo_time']


class CPMGFilter(FilterFunction):
    """
    Filter function for CPMG sequence (Carr-Purcell-Meiboom-Gill)
    
    F(ω,τ,N) = (sin(Nωτ/2)/sin(ωτ/2))² × sin⁴(ωτ/4)/ω²
    
    N π-pulses with spacing τ provide enhanced noise suppression
    at specific frequencies.
    """
    
    def evaluate(self, frequencies: np.ndarray, tau: float, n_pulses: int) -> np.ndarray:
        """
        Calculate CPMG filter function
        
        Args:
            frequencies: Angular frequencies in rad/s
            tau: Interpulse spacing in seconds
            n_pulses: Number of π-pulses
            
        Returns:
            Filter function values
        """
        omega_tau_half = frequencies * tau / 2
        omega_tau_quarter = frequencies * tau / 4
        
        # Handle small frequency case
        mask_small = np.abs(omega_tau_half) < 1e-6
        mask_large = ~mask_small
        
        result = np.zeros_like(frequencies)
        
        # Small frequency approximation
        if np.any(mask_small):
            x = omega_tau_quarter[mask_small]
            # For small ω: (sin(Nx)/sin(x))² ≈ N² for x→0
            interference_factor = n_pulses**2
            echo_factor = 8 * x**4 / 3
            result[mask_small] = interference_factor * echo_factor
            
        # Large frequency exact calculation  
        if np.any(mask_large):
            omega = frequencies[mask_large]
            
            # Interference factor from N pulses
            sin_n_omega_tau_half = np.sin(n_pulses * omega * tau / 2)
            sin_omega_tau_half = np.sin(omega * tau / 2)
            
            # Avoid division by zero
            safe_mask = np.abs(sin_omega_tau_half) > 1e-12
            interference_factor = np.ones_like(omega) * n_pulses**2
            interference_factor[safe_mask] = (
                sin_n_omega_tau_half[safe_mask] / sin_omega_tau_half[safe_mask]
            )**2
            
            # Echo filter factor
            echo_factor = 8 * np.sin(omega * tau / 4)**4 / omega**2
            
            result[mask_large] = interference_factor * echo_factor
            
        return result
    
    def get_required_params(self) -> list:
        return ['tau', 'n_pulses']


class DDSequenceFilter(FilterFunction):
    """
    Filter function for general Dynamical Decoupling sequences
    
    Supports XY4, XY8, and other composite pulse sequences
    """
    
    def __init__(self, sequence_type: str = 'xy4'):
        """
        Initialize DD sequence filter
        
        Args:
            sequence_type: Type of DD sequence ('xy4', 'xy8', 'uhrig')
        """
        self.sequence_type = sequence_type.lower()
        
        # Define pulse timings for different sequences
        self.pulse_patterns = {
            'xy4': [1/8, 3/8, 5/8, 7/8],  # Normalized pulse positions
            'xy8': [1/16, 3/16, 5/16, 7/16, 9/16, 11/16, 13/16, 15/16],
            'uhrig': self._uhrig_timings(4)  # Default 4 pulses
        }
    
    def _uhrig_timings(self, n: int) -> np.ndarray:
        """Generate Uhrig DD sequence timings"""
        k = np.arange(1, n+1)
        return np.sin(np.pi * k / (2 * (n + 1)))**2
    
    def evaluate(self, frequencies: np.ndarray, total_time: float, 
                n_pulses: int = None) -> np.ndarray:
        """
        Calculate DD sequence filter function
        
        Args:
            frequencies: Angular frequencies in rad/s
            total_time: Total sequence time in seconds
            n_pulses: Number of pulses (for Uhrig sequence)
            
        Returns:
            Filter function values
        """
        if self.sequence_type == 'uhrig' and n_pulses is not None:
            pulse_positions = self._uhrig_timings(n_pulses)
        else:
            pulse_positions = np.array(self.pulse_patterns[self.sequence_type])
        
        # Calculate filter function using Magnus expansion
        result = np.zeros_like(frequencies)
        
        for omega in frequencies:
            # Sum over all pulse positions
            phase_accumulation = 0
            current_phase = 0
            
            for i, pos in enumerate(pulse_positions):
                # Evolution to pulse position
                evolution_time = pos * total_time
                if i == 0:
                    prev_time = 0
                else:
                    prev_time = pulse_positions[i-1] * total_time
                    
                dt = evolution_time - prev_time
                current_phase += omega * dt
                
                # π-pulse flips the phase
                current_phase *= -1
                
            # Final evolution
            final_dt = total_time - pulse_positions[-1] * total_time
            current_phase += omega * final_dt
            
            # Filter function is |∫₀ᵀ e^(iφ(t)) dt|²
            result[np.where(frequencies == omega)[0][0]] = np.abs(current_phase)**2
            
        return result / total_time**2  # Normalize
    
    def get_required_params(self) -> list:
        params = ['total_time']
        if self.sequence_type == 'uhrig':
            params.append('n_pulses')
        return params


class FilterFunctionCalculator:
    """
    Main calculator class for filter functions
    
    Provides easy interface to calculate filter functions for common
    pulse sequences and integrate with noise spectra.
    """
    
    def __init__(self):
        """Initialize with available filter functions"""
        self.filters = {
            'free_precession': FreePrecessionFilter(),
            'ramsey': FreePrecessionFilter(),  # Alias
            'spin_echo': SpinEchoFilter(),
            'echo': SpinEchoFilter(),  # Alias
            'cpmg': CPMGFilter(),
            'xy4': DDSequenceFilter('xy4'),
            'xy8': DDSequenceFilter('xy8'),
            'uhrig': DDSequenceFilter('uhrig')
        }
    
    def get_filter_function(self, sequence_type: str, frequencies: np.ndarray,
                           **params) -> np.ndarray:
        """
        Calculate filter function for given sequence
        
        Args:
            sequence_type: Type of pulse sequence
            frequencies: Frequencies to evaluate at (Hz or rad/s)
            **params: Sequence-specific parameters
            
        Returns:
            Filter function values
        """
        if sequence_type not in self.filters:
            raise ValueError(f"Unknown sequence type: {sequence_type}")
        
        filter_obj = self.filters[sequence_type]
        
        # Check required parameters
        required = filter_obj.get_required_params()
        missing = [p for p in required if p not in params]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")
        
        # Convert frequencies to rad/s if needed
        if 'frequencies_in_hz' in params and params['frequencies_in_hz']:
            omega = 2 * np.pi * frequencies
        else:
            omega = frequencies
            
        return filter_obj.evaluate(omega, **params)
    
    def calculate_t2_from_spectrum(self, sequence_type: str, 
                                  noise_psd: np.ndarray, 
                                  frequencies: np.ndarray,
                                  gamma_e: float = 2.8024925e10,
                                  **sequence_params) -> float:
        """
        Calculate T2 time from noise spectrum and filter function
        
        Args:
            sequence_type: Pulse sequence type
            noise_psd: Noise power spectral density [T²/Hz]
            frequencies: Frequency array [Hz]
            gamma_e: Electron gyromagnetic ratio [Hz/T]
            **sequence_params: Sequence-specific parameters
            
        Returns:
            T2 time in seconds
        """
        # Get filter function
        omega = 2 * np.pi * frequencies
        filter_func = self.get_filter_function(
            sequence_type, omega, **sequence_params
        )
        
        # Calculate dephasing rate: Γ_φ = γ_e² ∫ S_B(ω) F(ω) dω
        integrand = noise_psd * filter_func
        dephasing_rate = gamma_e**2 * np.trapz(integrand, omega)
        
        # T2 = 1/Γ_φ
        if dephasing_rate > 0:
            return 1.0 / dephasing_rate
        else:
            return np.inf
    
    def plot_filter_functions(self, frequencies: np.ndarray, 
                             sequences: Dict[str, Dict[str, Any]],
                             save_path: str = None):
        """
        Plot comparison of different filter functions
        
        Args:
            frequencies: Frequency array for plotting
            sequences: Dict of {sequence_name: {params}}
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 8))
        
        for seq_name, params in sequences.items():
            try:
                filter_vals = self.get_filter_function(seq_name, frequencies, **params)
                plt.loglog(frequencies, filter_vals, label=seq_name, linewidth=2)
            except Exception as e:
                print(f"Could not plot {seq_name}: {e}")
        
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Filter Function')
        plt.title('Pulse Sequence Filter Functions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


# Convenience functions for easy access
def calculate_ramsey_t2(noise_psd: np.ndarray, frequencies: np.ndarray, 
                       evolution_time: float) -> float:
    """Quick calculation of Ramsey T2"""
    calc = FilterFunctionCalculator()
    return calc.calculate_t2_from_spectrum(
        'ramsey', noise_psd, frequencies, evolution_time=evolution_time
    )


def calculate_echo_t2(noise_psd: np.ndarray, frequencies: np.ndarray,
                     echo_time: float) -> float:
    """Quick calculation of spin echo T2"""
    calc = FilterFunctionCalculator()
    return calc.calculate_t2_from_spectrum(
        'echo', noise_psd, frequencies, echo_time=echo_time
    )


def calculate_cpmg_t2(noise_psd: np.ndarray, frequencies: np.ndarray,
                     tau: float, n_pulses: int) -> float:
    """Quick calculation of CPMG T2"""
    calc = FilterFunctionCalculator()
    return calc.calculate_t2_from_spectrum(
        'cpmg', noise_psd, frequencies, tau=tau, n_pulses=n_pulses
    )


# Example usage and validation
if __name__ == "__main__":
    # Test the filter functions
    frequencies = np.logspace(-3, 6, 1000)  # 1 mHz to 1 MHz
    
    # Example sequences to compare
    sequences = {
        'Free Precession (1μs)': {'evolution_time': 1e-6},
        'Spin Echo (2μs)': {'echo_time': 2e-6},
        'CPMG-4 (τ=0.5μs)': {'tau': 0.5e-6, 'n_pulses': 4},
        'XY4 (10μs total)': {'total_time': 10e-6}
    }
    
    calc = FilterFunctionCalculator()
    calc.plot_filter_functions(frequencies, sequences)
    
    print("✅ Filter functions module successfully loaded!")
    print("Available sequences:", list(calc.filters.keys()))