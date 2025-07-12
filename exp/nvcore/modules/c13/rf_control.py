"""
RF Control Engine

Direct RF control of ¹³C nuclear spins with selective addressing.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Union
import sys
import os

# Add path for system constants
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'helper'))
from noise_sources import SYSTEM


class RFControlEngine:
    """
    RF control for ¹³C nuclear spins
    
    Provides:
    - Selective RF addressing of individual nuclei
    - Composite pulse sequences for robust control
    - Time-dependent Hamiltonian generation
    """
    
    def __init__(self, n_c13: int):
        """
        Initialize RF control engine
        
        Args:
            n_c13: Number of C13 nuclei
        """
        self.n_c13 = n_c13
        self.gamma_n = SYSTEM.get_constant('nv_center', 'gamma_n_13c')
        
        # Active RF pulses: [(target, start_time, end_time, frequency, amplitude, phase), ...]
        self.active_pulses = []
        
    def add_rf_pulse(self, target_nuclei: Union[int, List[int]], 
                    pulse_params: Dict[str, Any]):
        """
        Add RF pulse to queue
        
        Args:
            target_nuclei: Index or list of target nuclei
            pulse_params: Pulse parameters
        """
        if isinstance(target_nuclei, int):
            target_nuclei = [target_nuclei]
            
        pulse_entry = {
            'targets': target_nuclei,
            'start_time': pulse_params['start_time'],
            'end_time': pulse_params['end_time'], 
            'frequency': pulse_params['frequency'],
            'amplitude': pulse_params['amplitude'],
            'phase': pulse_params.get('phase', 0.0)
        }
        
        self.active_pulses.append(pulse_entry)
        
    def get_rf_hamiltonian(self, t: float, c13_operators: Dict[int, Dict[str, np.ndarray]]) -> np.ndarray:
        """
        Get time-dependent RF Hamiltonian
        
        Args:
            t: Current time [s]
            c13_operators: C13 operators
            
        Returns:
            RF Hamiltonian [Hz]
        """
        if self.n_c13 == 0:
            return np.array([[0.0]])
            
        dim = 2**self.n_c13
        H_rf = np.zeros((dim, dim), dtype=complex)
        
        # Add contribution from each active pulse
        for pulse in self.active_pulses:
            if pulse['start_time'] <= t <= pulse['end_time']:
                # Rotating RF field
                omega_rf = pulse['frequency']
                amplitude = pulse['amplitude']
                phase = pulse['phase']
                
                # Time-dependent phase factor
                time_factor = np.exp(-1j * (omega_rf * t + phase))
                
                for target in pulse['targets']:
                    if target < self.n_c13:
                        # Add RF Hamiltonian: Ω/2 * (I+ * e^(-iωt) + I- * e^(iωt))
                        rabi_freq = self.gamma_n * amplitude
                        H_rf += 0.5 * rabi_freq * (
                            c13_operators[target]['I+'] * time_factor +
                            c13_operators[target]['I-'] * time_factor.conjugate()
                        )
                        
        return H_rf * 2 * np.pi
        
    def clear_pulses(self):
        """Clear all RF pulses"""
        self.active_pulses.clear()
        
    def apply_rf_pulse(self, target_nuclei: Union[int, List[int]], 
                      pulse_params: Dict[str, Any]) -> np.ndarray:
        """
        Apply instantaneous RF pulse (for testing)
        
        Args:
            target_nuclei: Target nuclei
            pulse_params: Pulse parameters
            
        Returns:
            RF pulse propagator
        """
        # This is a simplified version for testing
        # In practice, pulses would be added to the queue and evolved with time
        
        if isinstance(target_nuclei, int):
            target_nuclei = [target_nuclei]
            
        dim = 2**self.n_c13
        U_rf = np.eye(dim, dtype=complex)
        
        # For each target, apply rotation
        angle = pulse_params.get('angle', np.pi/2)  # Default π/2 pulse
        
        # This is a placeholder - real implementation would use proper time evolution
        
        return U_rf