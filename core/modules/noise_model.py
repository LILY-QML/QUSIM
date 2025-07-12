#!/usr/bin/env python3
"""
Noise Model Module for NV Center Simulations
Handles decoherence, relaxation, and environmental noise
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any, List, Tuple, Optional, Callable


class NoiseModel:
    """Comprehensive noise model for NV centers"""
    
    def __init__(self, relaxation_params: Dict[str, Any], dimension: int):
        """
        Initialize noise model with relaxation parameters
        
        Args:
            relaxation_params: Dictionary containing T1_s, Tphi_s, and other noise parameters
            dimension: Dimension of the Hilbert space
        """
        self.T1_s = relaxation_params['T1_s']
        self.Tphi_s = relaxation_params.get('Tphi_s', relaxation_params.get('T2star_s', 1e-6))
        self.dimension = dimension
        
        # Additional noise parameters (can be extended)
        self.T2_s = relaxation_params.get('T2_s', 2 * self.T1_s)  # T2 <= 2*T1
        self.spectral_diffusion_rate = relaxation_params.get('spectral_diffusion_rate_Hz', 0.0)
        self.charge_noise_amplitude = relaxation_params.get('charge_noise_amplitude_Hz', 0.0)
        
        # Pre-compute decay rates
        self.gamma_1 = 1.0 / self.T1_s if self.T1_s > 0 else 0.0
        self.gamma_phi = 1.0 / self.Tphi_s if self.Tphi_s > 0 else 0.0
        self.gamma_2 = 1.0 / self.T2_s if self.T2_s > 0 else 0.0
        
    def basic_dissipator(self, rho: jnp.ndarray, operators: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Apply basic Lindblad dissipator for T1 and T2* relaxation
        
        Args:
            rho: Density matrix
            operators: Dictionary containing required operators (Sz, Pg, etc.)
            
        Returns:
            Dissipator contribution to drho/dt
        """
        out = jnp.zeros_like(rho, dtype=jnp.complex64)
        
        # Pure dephasing (T2*)
        if self.gamma_phi > 0:
            Sz = operators['Sz']
            L_phi = jnp.sqrt(self.gamma_phi) * (Sz - jnp.trace(Sz) / self.dimension)
            out += self._lindblad_term(rho, L_phi)
        
        # Energy relaxation (T1) - simplified for ground state
        if self.gamma_1 > 0 and 'Pg' in operators:
            Pg = operators['Pg']
            L_1 = jnp.sqrt(self.gamma_1) * Pg
            out += self._lindblad_term(rho, L_1)
        
        return out
    
    def advanced_dissipator(self, 
                          rho: jnp.ndarray, 
                          operators: Dict[str, jnp.ndarray],
                          t: float = 0.0) -> jnp.ndarray:
        """
        Advanced dissipator with multiple noise sources
        
        Args:
            rho: Density matrix
            operators: Dictionary containing all required operators
            t: Current time (for time-dependent noise)
            
        Returns:
            Total dissipator contribution
        """
        out = self.basic_dissipator(rho, operators)
        
        # Additional noise channels can be added here:
        
        # 1. Spectral diffusion (time-dependent dephasing)
        if self.spectral_diffusion_rate > 0 and 'Sz' in operators:
            # Simplified model - could be made more sophisticated
            gamma_sd = self.spectral_diffusion_rate * (1 + 0.1 * jnp.sin(2 * jnp.pi * 1e6 * t))
            L_sd = jnp.sqrt(gamma_sd) * operators['Sz']
            out += self._lindblad_term(rho, L_sd)
        
        # 2. Charge noise
        if self.charge_noise_amplitude > 0 and 'Sx' in operators and 'Sy' in operators:
            # Random telegraph noise model (simplified)
            L_charge_x = jnp.sqrt(self.charge_noise_amplitude) * operators['Sx']
            L_charge_y = jnp.sqrt(self.charge_noise_amplitude) * operators['Sy']
            out += 0.1 * (self._lindblad_term(rho, L_charge_x) + self._lindblad_term(rho, L_charge_y))
        
        return out
    
    def _lindblad_term(self, rho: jnp.ndarray, L: jnp.ndarray) -> jnp.ndarray:
        """
        Calculate single Lindblad term: L*rho*L† - 0.5*(L†L*rho + rho*L†L)
        
        Args:
            rho: Density matrix
            L: Lindblad operator
            
        Returns:
            Lindblad term contribution
        """
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        return L @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L)
    
    def magnetic_field_noise(self, B_vec_G: jnp.ndarray, noise_amplitude_G: float = 0.1) -> jnp.ndarray:
        """
        Model magnetic field noise
        
        Args:
            B_vec_G: Static magnetic field vector in Gauss
            noise_amplitude_G: RMS noise amplitude in Gauss
            
        Returns:
            Noisy magnetic field vector
        """
        # This is a placeholder - in reality would need time correlation
        noise = jax.random.normal(jax.random.PRNGKey(0), shape=(3,)) * noise_amplitude_G
        return B_vec_G + noise
    
    def nuclear_spin_bath_dynamics(self, 
                                 rho: jnp.ndarray,
                                 bath_params: Dict[str, Any]) -> jnp.ndarray:
        """
        Model dynamics due to nuclear spin bath (C13 environment)
        
        Args:
            rho: System density matrix
            bath_params: Parameters for the nuclear spin bath
            
        Returns:
            Modified density matrix
        """
        # Placeholder for nuclear spin bath effects
        # Could implement:
        # - Cluster correlation functions
        # - Dynamical decoupling effects
        # - Non-Markovian dynamics
        return rho
    
    def get_noise_spectrum(self, 
                         frequencies: jnp.ndarray,
                         noise_type: str = 'dephasing') -> jnp.ndarray:
        """
        Calculate noise power spectral density
        
        Args:
            frequencies: Array of frequencies in Hz
            noise_type: Type of noise ('dephasing', 'relaxation', 'charge')
            
        Returns:
            Power spectral density S(f)
        """
        if noise_type == 'dephasing':
            # Lorentzian spectrum
            tau_c = 1e-6  # Correlation time
            return (2 * tau_c * self.gamma_phi) / (1 + (2 * jnp.pi * frequencies * tau_c)**2)
        elif noise_type == 'relaxation':
            # Ohmic spectrum with cutoff
            cutoff = 1e9  # Hz
            return self.gamma_1 * frequencies / (frequencies**2 + cutoff**2)
        elif noise_type == 'charge':
            # 1/f noise
            f0 = 1.0  # Hz
            return self.charge_noise_amplitude**2 * f0 / (frequencies + 1e-10)
        else:
            return jnp.zeros_like(frequencies)
    
    def filter_function(self, 
                       pulse_sequence: List[Dict[str, Any]],
                       total_time: float) -> Callable:
        """
        Calculate filter function for dynamical decoupling sequences
        
        Args:
            pulse_sequence: List of pulses in the sequence
            total_time: Total sequence time
            
        Returns:
            Filter function F(omega)
        """
        # Placeholder - would calculate actual filter function
        # based on pulse sequence for noise suppression analysis
        def F(omega):
            return jnp.ones_like(omega)
        return F