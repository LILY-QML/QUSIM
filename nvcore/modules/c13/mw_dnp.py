"""
Microwave Dynamic Nuclear Polarization Engine

Implementation of electron-driven DNP via MW pulses and Hartmann-Hahn conditions.
"""

import numpy as np
from typing import Dict, List, Optional, Any
import sys
import os

# Add path for system constants
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'helper'))
from noise_sources import SYSTEM


class MicrowaveDNPEngine:
    """
    Microwave-driven dynamic nuclear polarization
    
    Uses Hartmann-Hahn conditions to transfer polarization from NV electron spin
    to ¹³C nuclear spins via MW irradiation.
    """
    
    def __init__(self, hyperfine_engine, zeeman_engine):
        """
        Initialize MW DNP engine
        
        Args:
            hyperfine_engine: Reference to hyperfine engine
            zeeman_engine: Reference to Zeeman engine
        """
        self.hyperfine = hyperfine_engine
        self.zeeman = zeeman_engine
        
        # Physical constants
        self.gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
        self.gamma_n = SYSTEM.get_constant('nv_center', 'gamma_n_13c')
        self.D = SYSTEM.get_constant('nv_center', 'd_gs')
        
        # DNP sequence queue
        self.dnp_sequences = []
        
    def check_hartmann_hahn_condition(self, mw_frequency: float, 
                                    c13_index: int, B_field: float) -> bool:
        """
        Check if Hartmann-Hahn condition is satisfied
        
        Args:
            mw_frequency: MW frequency [Hz]
            c13_index: Target C13 index
            B_field: Magnetic field magnitude [T]
            
        Returns:
            True if condition is satisfied
        """
        # NV transition frequencies
        nv_freq_plus = self.D + self.gamma_e * B_field
        nv_freq_minus = self.D - self.gamma_e * B_field
        
        # C13 Larmor frequency
        c13_freq = self.gamma_n * B_field
        
        # Hyperfine coupling
        hyperfine_tensors = self.hyperfine.get_hyperfine_tensors()
        if c13_index in hyperfine_tensors:
            A_par, A_perp = hyperfine_tensors[c13_index]
        else:
            return False
            
        # Hartmann-Hahn conditions: |ω_mw - ω_nv| ≈ ω_n ± A/2
        resonance_threshold = 1e6  # 1 MHz threshold
        
        detuning_plus = abs(mw_frequency - nv_freq_plus)
        detuning_minus = abs(mw_frequency - nv_freq_minus)
        
        condition_1 = abs(detuning_plus - c13_freq) < resonance_threshold
        condition_2 = abs(detuning_minus - c13_freq) < resonance_threshold
        condition_3 = abs(detuning_plus - abs(A_par)/2) < resonance_threshold
        condition_4 = abs(detuning_minus - abs(A_par)/2) < resonance_threshold
        
        return condition_1 or condition_2 or condition_3 or condition_4
        
    def apply_dnp_sequence(self, dnp_params: Dict[str, Any]) -> Dict[str, float]:
        """
        Apply DNP sequence
        
        Args:
            dnp_params: DNP parameters
            
        Returns:
            DNP results
        """
        # Placeholder implementation
        results = {
            'polarization_transfer': 0.1,  # 10% transfer efficiency
            'transfer_time': dnp_params.get('duration', 1e-3),  # 1 ms
            'efficiency': 0.1
        }
        
        return results
        
    def compute_efficiency(self, mw_sequence: List[Dict]) -> float:
        """Compute DNP efficiency for MW sequence"""
        # Simplified model
        return 0.1  # 10% efficiency