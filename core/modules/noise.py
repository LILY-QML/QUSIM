#!/usr/bin/env python3
"""
Noise and Decoherence Module
T1 relaxation, T2 dephasing, spectral diffusion, magnetic noise
"""

import jax
import jax.numpy as jnp
from typing import Dict, List, Tuple, Optional, Any, Callable
import numpy as np


class DecoherenceModel:
    """Decoherence processes for NV centers"""
    
    def __init__(self, relaxation_params: Dict[str, float], dimension: int):
        """
        Args:
            relaxation_params: T1_s, Tphi_s, spectral_diffusion_rate_Hz, etc.
            dimension: Hilbert space dimension
        """
        # Basic relaxation
        self.T1_s = relaxation_params['T1_s']
        self.Tphi_s = relaxation_params.get('Tphi_s', 1e-6)
        self.dimension = dimension
        
        # Extended decoherence
        self.T2_s = relaxation_params.get('T2_s', 2 * self.T1_s)
        self.spectral_diffusion_rate_Hz = relaxation_params.get('spectral_diffusion_rate_Hz', 0.0)
        self.charge_noise_amplitude_Hz = relaxation_params.get('charge_noise_amplitude_Hz', 0.0)
        self.magnetic_noise_amplitude_G = relaxation_params.get('magnetic_noise_amplitude_G', 0.0)
        self.correlation_time_s = relaxation_params.get('correlation_time_s', 1e-6)
        self.nuclear_bath_coupling_Hz = relaxation_params.get('nuclear_bath_coupling_Hz', 0.0)
        
        # Decay rates
        self.gamma_1 = 1.0 / self.T1_s if self.T1_s > 0 else 0.0
        self.gamma_phi = 1.0 / self.Tphi_s if self.Tphi_s > 0 else 0.0
        self.gamma_2 = 1.0 / self.T2_s if self.T2_s > 0 else 0.0
    
    def lindblad_term(self, rho: jnp.ndarray, L: jnp.ndarray) -> jnp.ndarray:
        """Lindblad superoperator: L*rho*L† - 0.5*(L†L*rho + rho*L†L)"""
        L_dag = L.conj().T
        L_dag_L = L_dag @ L
        return L @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L)
    
    def basic_dissipator(self, rho: jnp.ndarray, operators: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Basic T1 and T2* dissipator"""
        out = jnp.zeros_like(rho, dtype=jnp.complex64)
        
        # Pure dephasing (T2*)
        if self.gamma_phi > 0 and 'Sz' in operators:
            Sz = operators['Sz']
            L_phi = jnp.sqrt(self.gamma_phi) * (Sz - jnp.trace(Sz) / self.dimension)
            out += self.lindblad_term(rho, L_phi)
        
        # Energy relaxation (T1)
        if self.gamma_1 > 0 and 'Pg' in operators:
            Pg = operators['Pg']
            L_1 = jnp.sqrt(self.gamma_1) * Pg
            out += self.lindblad_term(rho, L_1)
        
        return out
    
    def spectral_diffusion_dissipator(self, rho: jnp.ndarray, operators: Dict[str, jnp.ndarray], 
                                    t: float = 0.0) -> jnp.ndarray:
        """Time-dependent spectral diffusion"""
        if self.spectral_diffusion_rate_Hz <= 0 or 'Sz' not in operators:
            return jnp.zeros_like(rho)
        
        # Modulated spectral diffusion
        modulation = 1 + 0.1 * jnp.sin(2 * jnp.pi * 1e6 * t)
        gamma_sd = self.spectral_diffusion_rate_Hz * modulation
        L_sd = jnp.sqrt(gamma_sd) * operators['Sz']
        
        return self.lindblad_term(rho, L_sd)
    
    def charge_noise_dissipator(self, rho: jnp.ndarray, operators: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Random telegraph charge noise"""
        if self.charge_noise_amplitude_Hz <= 0:
            return jnp.zeros_like(rho)
        
        out = jnp.zeros_like(rho)
        
        if 'Sx' in operators and 'Sy' in operators:
            L_charge_x = jnp.sqrt(self.charge_noise_amplitude_Hz * 0.1) * operators['Sx']
            L_charge_y = jnp.sqrt(self.charge_noise_amplitude_Hz * 0.1) * operators['Sy']
            out += self.lindblad_term(rho, L_charge_x) + self.lindblad_term(rho, L_charge_y)
        
        return out
    
    def nuclear_bath_dissipator(self, rho: jnp.ndarray, operators: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        """Nuclear spin bath dephasing"""
        if self.nuclear_bath_coupling_Hz <= 0 or 'Sz' not in operators:
            return jnp.zeros_like(rho)
        
        L_nuclear = jnp.sqrt(self.nuclear_bath_coupling_Hz * 0.01) * operators['Sz']
        return self.lindblad_term(rho, L_nuclear)
    
    def complete_dissipator(self, rho: jnp.ndarray, operators: Dict[str, jnp.ndarray], 
                          t: float = 0.0) -> jnp.ndarray:
        """Complete decoherence model"""
        dissipator = self.basic_dissipator(rho, operators)
        dissipator += self.spectral_diffusion_dissipator(rho, operators, t)
        dissipator += self.charge_noise_dissipator(rho, operators)
        dissipator += self.nuclear_bath_dissipator(rho, operators)
        
        return dissipator
    
    def magnetic_field_noise(self, B_vec_G: jnp.ndarray, 
                           noise_key: jax.random.PRNGKey) -> jnp.ndarray:
        """Magnetic field fluctuations"""
        if self.magnetic_noise_amplitude_G <= 0:
            return B_vec_G
        
        noise = jax.random.normal(noise_key, shape=(3,)) * self.magnetic_noise_amplitude_G
        correlation_factor = jnp.exp(-1 / (self.correlation_time_s * 1e6))
        noise = noise * correlation_factor
        
        return B_vec_G + noise
    
    def noise_spectrum(self, frequencies: jnp.ndarray, noise_type: str = 'dephasing') -> jnp.ndarray:
        """Power spectral density of noise"""
        if noise_type == 'dephasing':
            tau_c = self.correlation_time_s
            base = (2 * tau_c * self.gamma_phi) / (1 + (2 * jnp.pi * frequencies * tau_c)**2)
            sd = (self.spectral_diffusion_rate_Hz * tau_c) / (1 + (2 * jnp.pi * frequencies * tau_c)**2)
            return base + sd
            
        elif noise_type == 'relaxation':
            cutoff = 1e9
            return self.gamma_1 * frequencies / (frequencies**2 + cutoff**2)
            
        elif noise_type == 'charge':
            f0 = 1.0
            f_cutoff = 1e9
            return (self.charge_noise_amplitude_Hz**2 * f0) / ((frequencies + 1e-10) * (1 + frequencies/f_cutoff))
            
        elif noise_type == 'magnetic':
            return self.magnetic_noise_amplitude_G**2 * jnp.exp(-frequencies * self.correlation_time_s)
        
        return jnp.zeros_like(frequencies)
    
    def dynamical_decoupling_filter(self, sequence_type: str, n_pulses: int, 
                                  total_time: float) -> Callable:
        """Filter function for dynamical decoupling"""
        
        def filter_function(omega: jnp.ndarray) -> jnp.ndarray:
            if sequence_type == "CPMG":
                tau = total_time / (2 * n_pulses)
                omega_tau = omega * tau
                return jnp.where(jnp.abs(omega_tau) > 1e-10, 
                               4 * jnp.sin(omega_tau)**2 / (omega_tau)**2, 
                               jnp.ones_like(omega))
            
            elif sequence_type == "XY8":
                tau = total_time / (8 * n_pulses)
                omega_tau = omega * tau
                return jnp.sin(4 * omega_tau)**2 / (4 * omega_tau)**2
            
            else:
                return jnp.ones_like(omega)
        
        return filter_function
    
    def coherence_time(self, sequence_type: str = "free", 
                     n_pulses: int = 0, total_time: float = 1e-6) -> float:
        """Effective coherence time with dynamical decoupling"""
        frequencies = jnp.logspace(-1, 9, 1000)
        S_dephasing = self.noise_spectrum(frequencies, 'dephasing')
        
        if sequence_type == "free":
            integral = jnp.trapz(S_dephasing, frequencies)
            T2_eff = 1.0 / (jnp.pi * integral) if integral > 0 else self.Tphi_s
        else:
            filter_func = self.dynamical_decoupling_filter(sequence_type, n_pulses, total_time)
            F_omega = filter_func(2 * jnp.pi * frequencies)
            filtered_spectrum = S_dephasing * F_omega
            integral = jnp.trapz(filtered_spectrum, frequencies)
            T2_eff = 1.0 / (jnp.pi * integral) if integral > 0 else 10 * self.Tphi_s
        
        return float(T2_eff)
    
    def get_parameters(self) -> Dict[str, float]:
        """All decoherence parameters"""
        return {
            'T1_s': self.T1_s,
            'T2_s': self.T2_s,
            'Tphi_s': self.Tphi_s,
            'spectral_diffusion_rate_Hz': self.spectral_diffusion_rate_Hz,
            'charge_noise_amplitude_Hz': self.charge_noise_amplitude_Hz,
            'magnetic_noise_amplitude_G': self.magnetic_noise_amplitude_G,
            'correlation_time_s': self.correlation_time_s,
            'nuclear_bath_coupling_Hz': self.nuclear_bath_coupling_Hz,
            'gamma_1': self.gamma_1,
            'gamma_phi': self.gamma_phi,
            'gamma_2': self.gamma_2
        }