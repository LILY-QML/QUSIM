#!/usr/bin/env python3
"""
Photon Detection Module for NV Center Simulations
Handles photon counting and optical readout modeling
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class PhotonDetector:
    """Photon detection model for NV centers"""
    
    def __init__(self, optical_params: Dict[str, Any], readout_params: Dict[str, Any]):
        """
        Initialize photon detector with optical and readout parameters
        
        Args:
            optical_params: Dictionary containing optical parameters (Tau_rad_s, Tau_MS_s, etc.)
            readout_params: Dictionary containing readout parameters (Beta_max_Hz, W_ms0_late, W_ms1_late)
        """
        # Optical parameters
        self.Tau_rad_s = optical_params['Tau_rad_s']
        self.Tau_MS_s = optical_params['Tau_MS_s']
        self.k_ISC_ms0 = optical_params['k_ISC_ms0_factor'] / self.Tau_rad_s
        self.k_ISC_ms1 = optical_params['k_ISC_ms1_factor'] / self.Tau_rad_s
        
        # Readout parameters
        self.Beta_max_Hz = readout_params['Beta_max_Hz']
        self.W_ms0_late = readout_params['W_ms0_late']
        self.W_ms1_late = readout_params['W_ms1_late']
        
        # Advanced model parameters (can be extended)
        self.pump_time_constant_ns = 20.0  # Time constant for optical pumping
        self.decay_time_constant_ns = 100.0  # Time constant for fluorescence decay
        
    def calculate_photon_rate(self, p_ms0: float, t_rel: float) -> float:
        """
        Calculate photon emission rate based on ms=0 population and relative time
        
        Args:
            p_ms0: Population in ms=0 state
            t_rel: Relative time since laser turn-on (in seconds)
            
        Returns:
            Photon emission rate in Hz
        """
        # Optical pumping dynamics
        pump = 1 - jnp.exp(-t_rel / (self.pump_time_constant_ns * 1e-9))
        
        # Time-dependent fluorescence contrast
        W1t = self.W_ms1_late + (1 - self.W_ms1_late) * jnp.exp(-t_rel / (self.decay_time_constant_ns * 1e-9))
        
        # Total photon rate
        rate = self.Beta_max_Hz * pump * (p_ms0 * self.W_ms0_late + (1 - p_ms0) * W1t)
        
        return rate
    
    def generate_counts(self, 
                       populations: Dict[str, np.ndarray],
                       photon_counter: Dict[str, Any],
                       dt_simulation: float,
                       rng_key: jax.random.PRNGKey) -> Dict[str, Any]:
        """
        Generate photon counts for a given measurement window
        
        Args:
            populations: Dictionary with 'times_ns' and 'ms0' population arrays
            photon_counter: Configuration for photon counting
            dt_simulation: Simulation time step in seconds
            rng_key: JAX random key for Poisson sampling
            
        Returns:
            Dictionary with photon count data
        """
        counter_start = photon_counter['start_ns']
        counter_duration = photon_counter['duration_ns']
        bin_width = photon_counter['bin_width_ns']
        shots = photon_counter.get('shots', 1000)
        
        # Time bins for photon counting
        count_times = jnp.arange(counter_start, counter_start + counter_duration, bin_width) * 1e-9
        counts = []
        
        for i, t in enumerate(count_times):
            # Find population at this time
            idx = int(t / dt_simulation)
            if idx < len(populations['ms0']):
                p0 = populations['ms0'][idx]
            else:
                p0 = populations['ms0'][-1]
            
            # Calculate photon rate
            t_rel = t - counter_start * 1e-9
            rate = self.calculate_photon_rate(p0, t_rel)
            
            # Generate Poisson counts
            lambda_counts = rate * bin_width * 1e-9 * shots
            rng_key, sub = jax.random.split(rng_key)
            count = jax.random.poisson(sub, lambda_counts)
            counts.append(float(count))
        
        return {
            'times_ns': count_times * 1e9,
            'counts': np.array(counts),
            'bin_width_ns': bin_width,
            'shots': shots,
            'mean_rate_Hz': float(np.mean(counts) / (bin_width * 1e-9 * shots))
        }
    
    def advanced_photon_model(self, 
                            p_ms0: float, 
                            p_ms1: float,
                            p_ms_minus1: float,
                            t_rel: float,
                            laser_power: float = 1.0) -> Dict[str, float]:
        """
        Advanced photon emission model with more physics
        Can be extended for more realistic simulations
        
        Args:
            p_ms0, p_ms1, p_ms_minus1: Populations in different ms states
            t_rel: Relative time since laser turn-on
            laser_power: Normalized laser power (0-1)
            
        Returns:
            Dictionary with emission rates and other optical properties
        """
        # This is a placeholder for more advanced modeling
        # Could include:
        # - Shelving state dynamics
        # - Ionization effects
        # - Power-dependent saturation
        # - Spectral diffusion
        # - Charge state dynamics
        
        # Basic implementation
        rate_ms0 = self.calculate_photon_rate(p_ms0, t_rel) * laser_power
        rate_ms1 = self.calculate_photon_rate(1 - p_ms0, t_rel) * laser_power
        
        return {
            'total_rate_Hz': rate_ms0 * p_ms0 + rate_ms1 * (p_ms1 + p_ms_minus1),
            'rate_ms0_Hz': rate_ms0,
            'rate_ms1_Hz': rate_ms1,
            'contrast': (rate_ms0 - rate_ms1) / (rate_ms0 + rate_ms1),
            'collection_efficiency': 0.03,  # Typical collection efficiency
            'detected_rate_Hz': (rate_ms0 * p_ms0 + rate_ms1 * (p_ms1 + p_ms_minus1)) * 0.03
        }