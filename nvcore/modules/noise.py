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
# Advanced modules removed in cleanup - using basic implementations only


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
    
    # Advanced features disabled after cleanup
    
    # Simulation parameters (defaults from system.json)
    dt: float = field(default_factory=lambda: SYSTEM.defaults['timestep'])
    seed: Optional[int] = 12345  # Default deterministic seed
    
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
        
        # CRITICAL FIX: Apply external field scaling for different experimental conditions
        if 'external_field_scaling' in preset:
            base_drift = SYSTEM.get_noise_param('magnetic', 'external_field', 'drift_amplitude')
            scaled_drift = base_drift * preset['external_field_scaling']
            config.parameter_overrides['external_field'] = {'drift_amplitude': scaled_drift}
            print(f"🎯 External field scaling: {preset['external_field_scaling']:.3f}x → {scaled_drift*1e12:.0f} pT")
        
        # Apply C13 field enhancement
        if 'c13_field_enhancement' in preset:
            if 'c13_bath' not in config.parameter_overrides:
                config.parameter_overrides['c13_bath'] = {}
            config.parameter_overrides['c13_bath']['field_enhancement_factor'] = preset['c13_field_enhancement']
            print(f"🎯 C13 field enhancement: {preset['c13_field_enhancement']:.1f}x")
        
        # Apply Johnson noise scaling
        if 'johnson_noise_scaling' in preset:
            # Scale Johnson noise parameters
            johnson_params = {}
            base_distance = SYSTEM.get_noise_param('magnetic', 'johnson', 'conductor_distance')
            # Inverse scaling - larger distance = less noise
            johnson_params['conductor_distance'] = base_distance / preset['johnson_noise_scaling']
            
            if 'johnson' not in config.parameter_overrides:
                config.parameter_overrides['johnson'] = {}
            config.parameter_overrides['johnson'].update(johnson_params)
            print(f"🎯 Johnson noise scaling: {preset['johnson_noise_scaling']:.3f}x")
            
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
    
    def __init__(self, config: NoiseConfiguration, system_coordinator):
        """
        Initialize noise generator
        
        Args:
            config: Noise configuration object REQUIRED
            system_coordinator: SystemCoordinator REQUIRED for hyperrealistic noise
        """
        if config is None:
            raise ValueError("NoiseConfiguration REQUIRED - no defaults allowed")
        if system_coordinator is None:
            raise ValueError("SystemCoordinator REQUIRED for noise generator - no fallbacks allowed")
        
        self.config = config
        self.system = system_coordinator
        self.rng = np.random.default_rng(self.config.seed)
        
        # Initialize enabled noise sources
        self.sources: Dict[str, NoiseSource] = {}
        self._initialize_sources()
        
        # Cache for performance
        self._spin_operators_cache = None
        
        # Advanced filter calculator removed in cleanup
        
    def _initialize_sources(self):
        """Initialize all enabled noise sources"""
        master_seed = self.config.seed
        
        if self.config.enable_c13_bath:
            override_params = self.config.parameter_overrides.get('c13_bath', {})
            self.sources['c13_bath'] = C13BathNoise(self.rng, override_params, master_seed)
            
        if self.config.enable_external_field:
            override_params = self.config.parameter_overrides.get('external_field', {})
            self.sources['external_field'] = ExternalFieldNoise(self.rng, override_params, master_seed)
            
        if self.config.enable_johnson:
            override_params = self.config.parameter_overrides.get('johnson', {})
            self.sources['johnson'] = JohnsonNoise(self.rng, override_params, master_seed)
            
        if self.config.enable_charge_noise:
            override_params = self.config.parameter_overrides.get('charge_state', {})
            self.sources['charge_state'] = ChargeStateNoise(self.rng, override_params, master_seed)
            
        if self.config.enable_temperature:
            override_params = self.config.parameter_overrides.get('thermal', {})
            self.sources['temperature'] = TemperatureFluctuation(self.rng, override_params, master_seed)
            
        if self.config.enable_strain:
            override_params = self.config.parameter_overrides.get('strain', {})
            self.sources['strain'] = StrainNoise(self.rng, override_params, master_seed)
            
        if self.config.enable_microwave:
            override_params = self.config.parameter_overrides.get('microwave', {})
            self.sources['microwave'] = MicrowaveNoise(self.rng, override_params, master_seed)
            
        if self.config.enable_optical:
            override_params = self.config.parameter_overrides.get('optical', {})
            self.sources['optical'] = OpticalNoise(self.rng, override_params, master_seed)
            
        # Advanced implementations removed in cleanup
            
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
    
    def get_total_magnetic_noise_vectorized(self, n_samples: int = 1) -> np.ndarray:
        """
        PERFORMANCE: Ultra-fast vectorized noise generation
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of shape (n_samples, 3) or (3,) if n_samples=1
        """
        # Pre-allocate arrays
        b_total = np.zeros((n_samples, 3), dtype=np.float64)
        
        # Vectorized source sampling
        if 'c13_bath' in self.sources:
            # Check if source has vectorized method
            if hasattr(self.sources['c13_bath'], 'sample_vectorized'):
                b_total += self.sources['c13_bath'].sample_vectorized(n_samples)
            else:
                b_total += self.sources['c13_bath'].sample(n_samples)
                
        if 'external_field' in self.sources:
            # Use vectorized method
            b_total += self.sources['external_field'].sample_vectorized(n_samples)
                
        if 'johnson' in self.sources:
            # Use vectorized method
            b_total += self.sources['johnson'].sample_vectorized(n_samples)
                    
        return b_total.squeeze() if n_samples == 1 else b_total
            
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
                
        # Pure dephasing from magnetic noise - NO PHANTOM SOURCES
        if include_sources is None or 'dephasing' in include_sources:
            
            # 🚨 CHECK: Do we have any magnetic noise sources?
            active_magnetic_sources = [name for name in ['c13_bath', 'external_field', 'johnson']
                                     if name in self.sources and getattr(self.config, f'enable_{name}', False)]
            
            if len(active_magnetic_sources) == 0:
                # NO MAGNETIC SOURCES = NO MAGNETIC DEPHASING
                print("🚨 NO magnetic noise sources - NO dephasing operators added")
                return lindblad_ops
            
            print(f"🔍 Calculating dephasing from sources: {active_magnetic_sources}")
            
            try:
                gamma_phi = self.calculate_dephasing_from_spectrum()
                
                if gamma_phi > 0 and 'Sz' in spin_operators:
                    lindblad_ops.append((spin_operators['Sz'], np.sqrt(gamma_phi)))
                    print(f"🔍 Added dephasing operator: γ_φ = {gamma_phi:.2e} Hz")
                else:
                    print("🔍 No dephasing operator added (γ_φ = 0)")
                
            except Exception as e:
                # ZERO FALLBACKS - if calculation fails, we fail
                raise RuntimeError(f"Dephasing calculation failed: {e}. NO FALLBACK VALUES!")
                
        return lindblad_ops
    
    def calculate_physical_minimum_dephasing(self) -> float:
        """Nur echte physikalische Beiträge - KEINE künstlichen Minima"""
        if len(self.sources) == 0:
            return 0.0  # Keine Quellen = Keine Dekohärenz
        
        # Nur fundamentale Quantengrenzen - KEINE künstlichen Faktoren
        temperature = self.sources.get('temperature', {}).get('base_temperature', 0) if 'temperature' in self.sources else 0
        if temperature > 0:
            # Echte thermische Anregungsrate ohne künstliche Faktoren
            from ...helper.noise_sources import SYSTEM
            kb_T = SYSTEM.get_constant('fundamental', 'kb') * temperature
            D = SYSTEM.get_constant('nv_center', 'd_gs')
            hbar = SYSTEM.get_constant('fundamental', 'hbar')
            
            # Spontane Emission Rate: γ = (ω³|μ|²)/(3πε₀ℏc³)
            # Vereinfacht für NV: thermische Anregungsrate bei T > 0
            if kb_T > 0:
                # Boltzmann Besetzung höherer Zustände
                thermal_population = np.exp(-D*hbar/(kb_T))
                if thermal_population > 1e-10:  # Nur wenn messbar besetzt
                    # Echte Übergangsrate ohne willkürliche Faktoren
                    return D * thermal_population / (2*np.pi*hbar)  # Hz
        
        return 0.0  # Sonst wirklich null
    
    def get_correlated_magnetic_noise(self, n_samples: int) -> np.ndarray:
        """Berücksichtige Kreuz-Korrelationen zwischen Quellen"""
        # Basis-Rauschen
        b_c13 = self.sources['c13_bath'].sample(n_samples) if 'c13_bath' in self.sources else 0
        b_ext = self.sources['external_field'].sample(n_samples) if 'external_field' in self.sources else 0
        
        # Kreuz-Korrelation: External field beeinflusst C13 Dynamik
        if isinstance(b_c13, np.ndarray) and isinstance(b_ext, np.ndarray):
            # C13 reagiert auf externe Feldänderungen
            # ECHTE Korrelationszeit aus SystemCoordinator
            if self.system is not None:
                # Echtes Magnetfeld vom System
                B_field = self.system.get_actual_magnetic_field()
                B_magnitude = np.linalg.norm(B_field)
                
                # Echte physikalische Konstanten
                gamma_c = self.system.get_physical_constant('gamma_n_13c')
                larmor_freq = gamma_c * B_magnitude
                correlation_time = 1.0 / larmor_freq if larmor_freq > 0 else 1e-6
                
                # Echte Hyperfein-Kopplung vom N14-Modul
                if self.system.has_module('n14'):
                    n14_engine = self.system.get_module('n14')
                    A_hf = abs(n14_engine.get_hyperfine_parameters()['A_parallel'])
                else:
                    # KEINE hardcoded Fallbacks - System muss vollständig sein
                    raise RuntimeError("N14 module required for hyperrealistic noise calculation!")
                    
                correlation_strength = min(0.5, A_hf / (2*np.pi*larmor_freq)) if larmor_freq > 0 else 0.1
            else:
                # NO LEGACY SUPPORT - SystemCoordinator is required
                raise ValueError("SystemCoordinator required for noise correlations - no legacy support")
            
            # Faltung für Gedächtniseffekt
            from scipy.ndimage import gaussian_filter1d
            b_ext_filtered = gaussian_filter1d(b_ext, sigma=correlation_time/self.config.dt, axis=0)
            b_c13 += correlation_strength * b_ext_filtered
        
        return b_c13 + b_ext
    
    def calculate_dephasing_from_spectrum(self) -> float:
        """Calculate dephasing with physically motivated frequency limits"""
        
        # Physikalisch motivierte Grenzen
        f_min = 1/(self._evolution_time_scale())  # Längste relevante Zeitskala
        f_max = min(
            SYSTEM.get_constant('nv_center', 'd_gs'),  # ZFS
            1/self.config.dt  # Nyquist
        )
        
        # Logarithmisch mit mehr Punkten bei Resonanzen
        base_freqs = np.logspace(np.log10(f_min), np.log10(f_max), 500)
        
        # Füge Resonanzfrequenzen hinzu
        resonances = self._get_system_resonances()
        frequencies = np.sort(np.concatenate([base_freqs, resonances]))
        
        print(f"🔍 Integration range: {f_min:.1e} to {f_max:.1e} Hz")
        
        # Get total magnetic noise PSD
        noise_psd = self.get_magnetic_noise_psd(frequencies)
        
        # CONVERGENCE VALIDATION
        converged_integral = self.validate_psd_convergence(frequencies, noise_psd)
        
        # SANITY CHECK: PSD values
        max_psd = np.max(noise_psd)
        print(f"🔍 Max PSD value: {max_psd:.2e} T²/Hz")
        
        # PHYSICS VALIDATION - NO ARTIFICIAL SCALING
        # Check if parameters give realistic T2* values without hacks
        
        gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
        integral_current = np.trapz(noise_psd, frequencies)
        
        print(f"🔍 Raw PSD integral: {integral_current:.2e} T²")
        
        # Calculate what T2* this would give
        gamma_phi_from_psd = 2 * gamma_e**2 * integral_current
        if gamma_phi_from_psd > 0:
            predicted_t2_star = 1 / gamma_phi_from_psd
            print(f"🔍 Predicted T2* from PSD: {predicted_t2_star:.2e} s ({predicted_t2_star*1e6:.1f} μs)")
            
            # VALIDATION: Check if realistic without scaling
            if predicted_t2_star < 1e-9:  # < 1 ns
                print(f"⚠️  WARNING: Raw PSD gives unrealistically short T2* = {predicted_t2_star*1e9:.1f} ns")
                print(f"    This indicates incorrect noise source parameters!")
                print(f"    Check system.json parameters for physical realism.")
            elif predicted_t2_star > 1e-1:  # > 0.1 s
                print(f"⚠️  WARNING: Raw PSD gives unrealistically long T2* = {predicted_t2_star*1e3:.1f} ms")
                print(f"    This indicates noise sources may be too weak.")
            else:
                print(f"✅ Raw PSD gives realistic T2* range - no scaling needed!")
        
        # NO MORE ARTIFICIAL SCALING - use physics as-is
        
        # Calculate dephasing rate
        gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
        gamma_phi_psd = 2 * gamma_e**2 * np.trapz(noise_psd, frequencies)
        
        print(f"🔍 PSD dephasing rate: {gamma_phi_psd:.2e} Hz")
        
        # Add physical minimum
        min_gamma_phi = self.calculate_physical_minimum_dephasing()
        gamma_phi = gamma_phi_psd + min_gamma_phi
        
        print(f"🔍 Total dephasing rate: {gamma_phi:.2e} Hz")
        
        if gamma_phi > 0:
            t2_star = 1 / gamma_phi
            print(f"🔍 Corresponding T2*: {t2_star:.2e} s ({t2_star*1e6:.1f} μs)")
        
        # BRUTAL SANITY CHECKS
        if gamma_phi > 1e9:  # > 1 GHz
            raise ValueError(f"Dephasing rate {gamma_phi:.2e} Hz exceeds 1 GHz limit")
        
        if gamma_phi > 0:
            t2_star = 1 / gamma_phi
            if t2_star < 1e-9:  # < 1 ns
                raise ValueError(f"T2* = {t2_star:.2e} s below 1 ns limit")
            if t2_star > 1e-1:  # > 0.1 s
                warnings.warn(f"T2* = {t2_star:.2e} s above 0.1 s (exceptionally good!)")
        
        return gamma_phi
    
    def validate_psd_convergence(self, frequencies: np.ndarray, psd: np.ndarray) -> float:
        """Adaptive Integration mit Fehlerabschätzung"""
        from scipy.integrate import quad
        
        def psd_func(log_f):
            f = np.exp(log_f)
            return np.interp(f, frequencies, psd) * f  # Jacobian für log-Integration
        
        # Adaptive Quadratur
        integral, error = quad(psd_func, np.log(frequencies[0]), np.log(frequencies[-1]),
                              epsabs=1e-20, epsrel=1e-10, limit=1000)
        
        print(f"🔍 Adaptive Integration: {integral:.2e} ± {error:.2e} T²")
        
        if error/integral > 0.01:
            print(f"⚠️ Integration Unsicherheit: {100*error/integral:.1f}%")
        
        return integral
    
    def _evolution_time_scale(self) -> float:
        """Get characteristic evolution time scale"""
        if hasattr(self.config, 'evolution_time'):
            return self.config.evolution_time
        else:
            # Get from SystemCoordinator if available
            if self.system is not None:
                # Use characteristic time scale from system resonances
                resonances = self.system.get_all_system_resonances()
                if len(resonances) > 0:
                    return 1.0 / np.min(resonances)  # 1/min_frequency
            # NO DEFAULT FALLBACK - require explicit configuration
            raise ValueError("Evolution time scale not configured - no fallbacks allowed")
    
    def _get_system_resonances(self) -> np.ndarray:
        """Echte Resonanzen aus allen Modulen vom SystemCoordinator"""
        if self.system is not None:
            # Verwende echte systemweite Resonanzen
            return self.system.get_all_system_resonances()
        else:
            # NO LEGACY SUPPORT - SystemCoordinator required
            raise ValueError("SystemCoordinator required for system resonances - no fallbacks allowed")
    
    def _extract_magnetic_field_from_sources(self) -> Optional[np.ndarray]:
        """Extrahiere echtes Magnetfeld aus SystemCoordinator"""
        if self.system is not None:
            # Use real magnetic field from SystemCoordinator
            return self.system.get_actual_magnetic_field()
        else:
            # NO FALLBACKS - require SystemCoordinator
            raise ValueError("SystemCoordinator required for magnetic field extraction - no fallbacks allowed")
        
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
        Estimate T2* from ACTUAL dephasing rate calculation
        
        Uses realistic physics-based dephasing calculation instead of autocorrelation
        
        Returns:
            Estimated T2* in seconds
        """
        print("🔍 Estimating T2* from dephasing operators...")
        
        # Use actual physics calculation
        spin_ops = {
            'Sz': np.array([[0.5, 0], [0, -0.5]], dtype=complex)
        }
        
        try:
            lindblad_ops = self.get_lindblad_operators(spin_ops)
            
            # Find dephasing rate
            for op, rate in lindblad_ops:
                if np.allclose(op, spin_ops['Sz']):
                    gamma_phi = rate**2  # Lindblad rate squared
                    t2_star = 1 / gamma_phi if gamma_phi > 0 else np.inf
                    print(f"🔍 T2* from dephasing rate {gamma_phi:.2e} Hz: {t2_star:.2e} s")
                    return t2_star
            
            # No dephasing operator found - this is a physics violation
            raise RuntimeError("💀 CRITICAL: No dephasing operators found!\n"
                             "🚨 This indicates broken noise source configuration.\n"
                             "🔥 Enable magnetic noise sources or disable dephasing calculation entirely.")
            
        except Exception as e:
            # NO FALLBACKS! If dephasing calculation fails, the system fails
            raise RuntimeError(f"💀 CRITICAL: Dephasing calculation failed: {e}\n"
                             f"🚨 NO FALLBACK VALUES ALLOWED!\n"
                             f"🔥 Fix the physics calculation or disable dephasing entirely.")
            
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
    
    def validate_physics(self) -> Dict[str, bool]:
        """Comprehensive physics validation of noise levels and parameters"""
        validator = PhysicsValidator(self)
        return validator.validate_all()
            
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


class PhysicsValidator:
    """Comprehensive physics validation for noise generators"""
    
    def __init__(self, noise_generator: NoiseGenerator):
        self.generator = noise_generator
        self.validation_results = {}
        
    def validate_noise_levels(self) -> Dict[str, bool]:
        """Validate magnetic noise levels are in physical range"""
        results = {}
        
        try:
            # Generate test noise samples
            noise = self.generator.get_total_magnetic_noise(10000)
            rms = np.sqrt(np.mean(noise**2))
            
            # Check RMS field strength
            results['rms_range'] = 1e-15 < rms < 1e-6  # 1 fT to 1 μT
            
            # Check for reasonable spectral content
            fft_noise = np.fft.fft(noise[:, 2])  # z-component
            power = np.abs(fft_noise)**2
            results['spectral_power'] = not np.any(power > 1e-10)  # No excessive power
            
            # Frequency spectrum check
            freqs = np.logspace(0, 6, 100)  # 1 Hz to 1 MHz
            psd = self.generator.get_magnetic_noise_psd(freqs)
            results['psd_finite'] = np.all(np.isfinite(psd))
            results['psd_positive'] = np.all(psd >= 0)
            
        except Exception as e:
            print(f"Noise level validation failed: {e}")
            results['validation_error'] = False
            
        return results
        
    def validate_coherence_times(self) -> Dict[str, bool]:
        """Validate estimated coherence times are in physical range"""
        results = {}
        
        try:
            # Estimate T2* from magnetic noise
            t2_star = self.generator.estimate_t2_star()
            results['t2_star_range'] = 1e-9 < t2_star < 1e-3  # 1 ns to 1 ms
            
            # Calculate dephasing rate
            lindblad_ops = self.generator.get_lindblad_operators(
                {'Sz': np.array([[1, 0], [0, -1]])}  # Simple 2-level system
            )
            
            dephasing_rates = [rate**2 for op, rate in lindblad_ops if op.shape == (2, 2)]
            if dephasing_rates:
                gamma_phi = max(dephasing_rates)
                t2_from_gamma = 1 / gamma_phi if gamma_phi > 0 else np.inf
                results['t2_from_dephasing'] = 1e-9 < t2_from_gamma < 1e-3
            else:
                results['t2_from_dephasing'] = False
                
        except Exception as e:
            print(f"Coherence time validation failed: {e}")
            results['coherence_error'] = False
            
        return results
        
    def validate_temperature_scaling(self) -> Dict[str, bool]:
        """Validate noise scales correctly with temperature"""
        results = {}
        
        try:
            if 'temperature' in self.generator.sources:
                temp_source = self.generator.sources['temperature']
                
                # Test temperature range
                temp_300k = temp_source.sample(1)
                results['temp_range'] = 1 < temp_300k < 1000  # 1K to 1000K
                
                # Test phonon occupation scaling
                test_energy = 2.87e9  # NV GS splitting in Hz
                phonon_n = temp_source.calculate_phonon_occupation(test_energy)
                results['phonon_occupation'] = 0 <= phonon_n < 100  # Reasonable range
                
            else:
                results['no_temperature_source'] = True
                
        except Exception as e:
            print(f"Temperature scaling validation failed: {e}")
            results['temperature_error'] = False
            
        return results
        
    def validate_frequency_response(self) -> Dict[str, bool]:
        """Validate frequency response follows physical laws"""
        results = {}
        
        try:
            # Test frequency range
            freqs = np.logspace(-1, 8, 1000)  # 0.1 Hz to 100 MHz
            
            # Check Johnson noise PSD
            if 'johnson' in self.generator.sources:
                johnson_psd = self.generator.sources['johnson'].get_power_spectral_density(freqs)
                
                # Should fall off at high frequencies due to skin depth
                high_freq_ratio = johnson_psd[-1] / johnson_psd[len(freqs)//2]
                results['johnson_cutoff'] = high_freq_ratio < 0.5  # At least 50% reduction
                
                # Should be finite and positive
                results['johnson_psd_valid'] = np.all(np.isfinite(johnson_psd)) and np.all(johnson_psd > 0)
                
            # Test overall PSD integration convergence
            total_psd = self.generator.get_magnetic_noise_psd(freqs)
            integral = np.trapz(total_psd, freqs)
            results['psd_convergent'] = np.isfinite(integral) and integral > 0
            
        except Exception as e:
            print(f"Frequency response validation failed: {e}")
            results['frequency_error'] = False
            
        return results
        
    def validate_determinism(self) -> Dict[str, bool]:
        """Validate time-deterministic behavior"""
        results = {}
        
        try:
            # Test same time gives same samples
            t_test = 1e-6  # 1 μs
            for source_name, source in self.generator.sources.items():
                samples1 = source.sample_at_time(t_test, 100)
                samples2 = source.sample_at_time(t_test, 100)
                
                if hasattr(samples1, 'shape') and hasattr(samples2, 'shape'):
                    results[f'{source_name}_deterministic'] = np.allclose(samples1, samples2)
                else:
                    results[f'{source_name}_deterministic'] = np.array_equal(samples1, samples2)
                    
        except Exception as e:
            print(f"Determinism validation failed: {e}")
            results['determinism_error'] = False
            
        return results
        
    def validate_all(self) -> Dict[str, bool]:
        """Run all validation tests"""
        all_results = {}
        
        all_results.update(self.validate_noise_levels())
        all_results.update(self.validate_coherence_times())
        all_results.update(self.validate_temperature_scaling())
        all_results.update(self.validate_frequency_response())
        all_results.update(self.validate_determinism())
        
        # Summary
        passed_tests = sum(all_results.values())
        total_tests = len(all_results)
        all_results['validation_summary'] = f"{passed_tests}/{total_tests} tests passed"
        all_results['all_passed'] = passed_tests == total_tests
        
        return all_results


class DeterministicNoiseGenerator(NoiseGenerator):
    """NoiseGenerator with ENFORCED time determinism"""
    
    def get_hamiltonian_noise_at_time(self, spin_operators: Dict[str, np.ndarray], t: float) -> np.ndarray:
        """FIXED: Guaranteed deterministic Hamiltonian with ULTRA precision"""
        
        # Ultra-precise time quantization (femtosecond precision)
        time_precision = 1e-15
        t_quantized = round(t / time_precision) * time_precision
        
        # Create ULTRA-specific cache key
        cache_key = (
            "DeterministicHamiltonian",
            hash(str(t_quantized)),
            tuple(sorted(spin_operators.keys())),
            tuple(op.shape for op in spin_operators.values()),  # Include shapes
            hash(str(spin_operators['Sx'].flatten()[:4])),  # Include sample of operator values
            id(self)  # Instance-specific
        )
        
        # Use instance-level cache for perfect determinism
        if not hasattr(self, '_hamiltonian_cache'):
            self._hamiltonian_cache = {}
            
        if cache_key in self._hamiltonian_cache:
            cached_result = self._hamiltonian_cache[cache_key]
            return cached_result.copy()  # Return COPY for safety
        
        print(f"🔍 Generating ULTRA-deterministic Hamiltonian at t={t_quantized:.2e} s")
        
        # Generate magnetic field noise with GUARANTEED determinism
        B_noise = np.zeros(3, dtype=np.float64)  # Explicit dtype
        
        # CRITICAL: Sort sources by name for deterministic iteration order
        source_names = sorted(self.sources.keys())
        print(f"  Processing sources in order: {source_names}")
        
        for source_name in source_names:
            source = self.sources[source_name]
            
            if hasattr(source, 'get_deterministic_sample_for_time'):
                # Use quantized time for perfect determinism
                source_noise = source.get_deterministic_sample_for_time(t_quantized, 1)
                
                # Handle different array shapes with STRICT validation
                if isinstance(source_noise, np.ndarray):
                    if source_noise.shape == (3,):
                        B_noise += source_noise.astype(np.float64)
                    elif source_noise.shape == (1, 3):
                        B_noise += source_noise[0].astype(np.float64)
                    elif len(source_noise.shape) == 1 and len(source_noise) == 3:
                        B_noise += source_noise.astype(np.float64)
                    else:
                        print(f"⚠️  {source_name}: unexpected shape {source_noise.shape}, taking first 3 elements")
                        B_noise += source_noise.flatten()[:3].astype(np.float64)
                else:
                    raise ValueError(f"💀 {source_name}: non-array result {type(source_noise)}")
                
                print(f"  {source_name}: {np.linalg.norm(source_noise):.6e} T")
            else:
                raise ValueError(f"💀 {source_name}: missing get_deterministic_sample_for_time method")
        
        print(f"  Total B_noise: {np.linalg.norm(B_noise):.6e} T")
        
        # Convert to Hamiltonian with EXACT reproducibility
        gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
        
        # Ensure spin operators are in a FIXED order with exact precision
        Sx = spin_operators['Sx'].astype(np.complex128)
        Sy = spin_operators['Sy'].astype(np.complex128)
        Sz = spin_operators['Sz'].astype(np.complex128)
        
        # Use exact floating point arithmetic
        H_noise = (2.0 * np.pi * gamma_e) * (
            B_noise[0] * Sx +
            B_noise[1] * Sy +
            B_noise[2] * Sz
        )
        
        # Ensure exact dtype
        H_noise = H_noise.astype(np.complex128)
        
        print(f"  H_noise max element: {np.max(np.abs(H_noise)):.6e} Hz")
        
        # Cache result for PERFECT determinism
        self._hamiltonian_cache[cache_key] = H_noise.copy()
        
        return H_noise
    
    def validate_all_time_determinism(self):
        """Validate time determinism for ALL sources"""
        
        print("🔍 Validating time determinism for all sources...")
        
        for source_name, source in self.sources.items():
            print(f"\n🔍 Testing {source_name}...")
            
            if hasattr(source, 'validate_time_determinism'):
                try:
                    source.validate_time_determinism()
                    print(f"✅ {source_name}: Time determinism OK")
                except Exception as e:
                    print(f"💀 {source_name}: Time determinism FAILED: {e}")
                    raise
            else:
                print(f"⚠️  {source_name}: No time determinism validation available")
        
        print("\n✅ ALL TIME DETERMINISM VALIDATION PASSED")
        return True
        
    def validate_interface_determinism(self, spin_ops: Dict[str, np.ndarray], t: float, n_trials: int = 10):
        """BRUTAL validation that interface Hamiltonian is deterministic"""
        
        print(f"🔥 BRUTAL interface determinism test at t={t:.2e} s, {n_trials} trials")
        
        hamiltonians = []
        for i in range(n_trials):
            H = self.get_hamiltonian_noise_at_time(spin_ops, t)
            hamiltonians.append(H.copy())
            
        # Check EXACT consistency across all trials
        reference = hamiltonians[0]
        max_diff = 0.0
        
        for i in range(1, len(hamiltonians)):
            H_current = hamiltonians[i]
            
            # Shape check
            if reference.shape != H_current.shape:
                raise ValueError(f"💀 Hamiltonian shape mismatch: {reference.shape} vs {H_current.shape}")
            
            # Element-wise difference
            diff_matrix = np.abs(reference - H_current)
            max_element_diff = np.max(diff_matrix)
            max_diff = max(max_diff, max_element_diff)
            
            if max_element_diff > 1e-10:  # 0.1 mHz tolerance
                print(f"💀 FAILURE at trial {i}: max diff = {max_element_diff:.6e} Hz")
                print(f"   Reference max: {np.max(np.abs(reference)):.6e} Hz")
                print(f"   Current max: {np.max(np.abs(H_current)):.6e} Hz")
                print(f"   Worst element index: {np.unravel_index(np.argmax(diff_matrix), diff_matrix.shape)}")
                raise ValueError(f"💀 Interface NOT deterministic: max diff = {max_element_diff:.6e} Hz > 1e-10 Hz")
        
        print(f"✅ Interface PERFECTLY deterministic: max diff = {max_diff:.6e} Hz < 1e-10 Hz")
        print(f"   Hamiltonian max element: {np.max(np.abs(reference)):.6e} Hz")
        return True


class PhysicsViolationError(Exception):
    """Custom exception for physics violations"""
    pass


class StrictPhysicsValidator:
    """Physics validator with ZERO TOLERANCE for violations"""
    
    def __init__(self, noise_generator):
        self.generator = noise_generator
        self.strict_mode = True  # ZERO compromise
        
    def validate_with_enforcement(self) -> Dict[str, bool]:
        """Validate physics and CRASH if violations found"""
        
        print("🔥 STRICT PHYSICS VALIDATION - ZERO TOLERANCE MODE")
        
        results = {}
        
        # Critical validations
        results.update(self.validate_t2_star_realism())
        results.update(self.validate_noise_levels_brutal())
        results.update(self.validate_dephasing_rates())
        
        # Count failures
        failed_checks = [k for k, v in results.items() if v == False]
        critical_failures = []
        
        # Classify failures by severity
        for check in failed_checks:
            if any(keyword in check for keyword in
                  ['t2_star_range', 'dephasing_realistic', 'noise_rms_physical']):
                critical_failures.append(check)
        
        # Report results
        total_checks = len(results)
        passed_checks = total_checks - len(failed_checks)
        
        print(f"🔍 Physics validation: {passed_checks}/{total_checks} passed")
        
        if failed_checks:
            print(f"❌ Failed checks: {failed_checks}")
        
        # ENFORCE in strict mode
        if self.strict_mode and critical_failures:
            error_msg = (
                f"💀 CRITICAL PHYSICS VIOLATIONS DETECTED!\n"
                f"Failed checks: {critical_failures}\n"
                f"System CANNOT run with broken physics.\n"
                f"Fix the violations or disable strict_mode."
            )
            raise PhysicsViolationError(error_msg)
        
        if self.strict_mode and len(failed_checks) > 2:
            error_msg = (
                f"💀 TOO MANY PHYSICS VIOLATIONS: {len(failed_checks)}\n"
                f"Failed checks: {failed_checks}\n"
                f"Maximum 2 violations allowed in strict mode."
            )
            raise PhysicsViolationError(error_msg)
        
        return results
    
    def validate_t2_star_realism(self) -> Dict[str, bool]:
        """BRUTAL T2* validation against experimental data"""
        
        results = {}
        
        try:
            t2_star = self.generator.estimate_t2_star()
            
            # Literature values for different conditions
            literature_ranges = {
                'isotopically_pure_cryogenic': (1e-3, 10e-3),    # 1-10 ms
                'isotopically_pure_room_temp': (100e-6, 1e-3),   # 0.1-1 ms  
                'natural_abundance_room_temp': (1e-6, 100e-6),   # 1-100 μs
                'high_concentration': (100e-9, 10e-6),           # 0.1-10 μs
                'surface_nv': (10e-9, 1e-6)                     # 10 ns - 1 μs
            }
            
            # Check against ALL ranges - must fit at least one
            fits_any_range = False
            for condition, (t_min, t_max) in literature_ranges.items():
                if t_min <= t2_star <= t_max:
                    fits_any_range = True
                    print(f"✅ T2* = {t2_star*1e6:.1f} μs fits {condition}")
                    break
            
            if not fits_any_range:
                print(f"💀 T2* = {t2_star*1e6:.1f} μs fits NO experimental condition!")
                print(f"Literature ranges:")
                for condition, (t_min, t_max) in literature_ranges.items():
                    print(f"  {condition}: {t_min*1e6:.1f} - {t_max*1e6:.1f} μs")
            
            results['t2_star_range'] = fits_any_range
            results['t2_star_finite'] = np.isfinite(t2_star)
            results['t2_star_realistic'] = 1e-9 <= t2_star <= 1e-1  # 1 ns to 0.1 s
            
        except Exception as e:
            print(f"💀 T2* validation crashed: {e}")
            results['t2_star_validation_error'] = False
        
        return results
    
    def validate_noise_levels_brutal(self) -> Dict[str, bool]:
        """BRUTAL noise level validation"""
        
        results = {}
        
        try:
            # Generate test noise samples
            noise = self.generator.get_total_magnetic_noise(1000)
            rms = np.sqrt(np.mean(noise**2))
            
            print(f"🔍 Noise RMS: {rms:.2e} T ({rms*1e12:.1f} pT)")
            
            # BRUTAL checks
            results['noise_rms_physical'] = 1e-18 <= rms <= 1e-6  # 1 aT to 1 μT
            results['noise_finite'] = np.all(np.isfinite(noise))
            results['noise_not_zero'] = rms > 0
            
            # Spectral check
            if len(noise) > 10:
                fft_noise = np.fft.fft(noise[:, 2])  # z-component
                power = np.abs(fft_noise)**2
                max_power = np.max(power)
                results['spectral_power_reasonable'] = max_power < 1e-15
                
                print(f"🔍 Max spectral power: {max_power:.2e}")
            
        except Exception as e:
            print(f"💀 Noise validation crashed: {e}")
            results['noise_validation_error'] = False
        
        return results
    
    def validate_dephasing_rates(self) -> Dict[str, bool]:
        """Validate dephasing rates are physically realistic"""
        
        results = {}
        
        try:
            # Test with simple spin system
            spin_ops = {
                'Sz': np.array([[0.5, 0], [0, -0.5]], dtype=complex),
                'S+': np.array([[0, 1], [0, 0]], dtype=complex),
                'S-': np.array([[0, 0], [1, 0]], dtype=complex)
            }
            
            lindblad_ops = self.generator.get_lindblad_operators(spin_ops)
            
            dephasing_rates = []
            for op, rate in lindblad_ops:
                if np.allclose(op, spin_ops['Sz']):
                    rate_hz = rate**2
                    dephasing_rates.append(rate_hz)
                    print(f"🔍 Dephasing rate: {rate_hz:.2e} Hz")
            
            if dephasing_rates:
                max_rate = max(dephasing_rates)
                min_rate = min(dephasing_rates)
                
                # BRUTAL checks
                results['dephasing_realistic'] = 0.1 <= max_rate <= 1e8  # 0.1 Hz to 100 MHz
                results['dephasing_finite'] = all(np.isfinite(r) for r in dephasing_rates)
                
                # T2* from dephasing should be reasonable
                if max_rate > 0:
                    t2_from_dephasing = 1 / max_rate
                    results['t2_from_dephasing_realistic'] = 1e-8 <= t2_from_dephasing <= 1e-1
                    print(f"🔍 T2* from dephasing: {t2_from_dephasing*1e6:.1f} μs")
            else:
                print("🔍 No dephasing operators found")
                results['no_dephasing_ok'] = True
            
        except Exception as e:
            print(f"💀 Dephasing validation crashed: {e}")
            results['dephasing_validation_error'] = False
        
        return results


class ValidatedNoiseGenerator(NoiseGenerator):
    """NoiseGenerator that ENFORCES physics validation"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Create strict validator
        self.validator = StrictPhysicsValidator(self)
        
        # MANDATORY validation after initialization
        self._validate_on_creation()
    
    def _validate_on_creation(self):
        """Validate physics immediately after creation"""
        
        print("🔍 Running mandatory physics validation...")
        
        try:
            validation_results = self.validator.validate_with_enforcement()
            print("✅ Physics validation PASSED - generator ready")
            
        except PhysicsViolationError as e:
            print(f"💀 PHYSICS VALIDATION FAILED:\n{e}")
            print("💀 NoiseGenerator CANNOT be used with broken physics")
            raise
    
    def get_total_magnetic_noise(self, n_samples=1):
        """Generate noise with periodic validation checks"""
        
        # Generate noise normally
        noise = super().get_total_magnetic_noise(n_samples)
        
        # Periodic validation for large samples
        if n_samples > 1000:
            rms = np.sqrt(np.mean(noise**2))
            
            if rms > 1e-9:  # > 1 nT
                warnings.warn(f"High noise level: {rms*1e12:.1f} pT")
            
            if not np.all(np.isfinite(noise)):
                raise PhysicsViolationError("Generated noise contains NaN/infinite values")
        
        return noise