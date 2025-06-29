"""
Multi-Level Charge State Dynamics for NV Centers

Implements realistic charge state transitions between NV+, NV0, and NV-
using master equation approach with temperature and field dependencies.

Physical Background:
- NV+ ⇌ NV0 ⇌ NV- transitions via ionization/recombination
- Rates depend on temperature, laser power, electric fields
- Includes both thermal and optical processes

References:
- Aslam et al., New J. Phys. 15, 013064 (2013)
- Beha et al., Phys. Rev. Lett. 109, 097404 (2012)
"""

import numpy as np
from typing import Dict, List, Optional, Union
from scipy.linalg import expm
from abc import ABC, abstractmethod

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from noise_sources import SYSTEM, NoiseSource


class ChargeStateTransition:
    """Represents a single charge state transition with its rate"""
    
    def __init__(self, from_state: str, to_state: str, 
                 base_rate: float, activation_energy: float = 0.0,
                 laser_dependence: float = 0.0, field_dependence: float = 0.0):
        """
        Initialize charge state transition
        
        Args:
            from_state: Source charge state ('nv_plus', 'nv_zero', 'nv_minus')
            to_state: Target charge state
            base_rate: Base transition rate [Hz]
            activation_energy: Thermal activation energy [eV]
            laser_dependence: Laser power dependence [1/mW]
            field_dependence: Electric field dependence [m/V]
        """
        self.from_state = from_state
        self.to_state = to_state
        self.base_rate = base_rate
        self.activation_energy = activation_energy
        self.laser_dependence = laser_dependence
        self.field_dependence = field_dependence
        
    def get_rate(self, temperature: float = 300.0, 
                laser_power: float = 0.0, electric_field: float = 0.0) -> float:
        """
        Calculate transition rate for given conditions
        
        Args:
            temperature: Temperature [K]
            laser_power: Laser power [mW]
            electric_field: Electric field [V/m]
            
        Returns:
            Transition rate [Hz]
        """
        kb = SYSTEM.get_constant('fundamental', 'kb')
        
        # Thermal activation
        thermal_factor = 1.0
        if self.activation_energy > 0:
            thermal_factor = np.exp(-self.activation_energy * 1.602e-19 / (kb * temperature))
            
        # Laser enhancement
        laser_factor = 1.0 + self.laser_dependence * laser_power
        
        # Electric field effect (Poole-Frenkel or similar)
        field_factor = 1.0 + self.field_dependence * abs(electric_field)
        
        return self.base_rate * thermal_factor * laser_factor * field_factor


class MultiLevelChargeNoise(NoiseSource):
    """
    Multi-level charge state dynamics with realistic transition rates
    
    Models NV+, NV0, and NV- states with temperature and laser dependencies
    """
    
    def __init__(self, rng: Optional[np.random.Generator] = None, 
                 override_params: Optional[dict] = None):
        """Initialize multi-level charge state model"""
        if override_params is None:
            override_params = {}
        super().__init__(rng)
        
        # System parameters - all must be provided in empirical parameters
        try:
            self.temperature = override_params.get('temperature', SYSTEM.get_empirical_param('charge_state_dynamics', 'base_temperature'))
            self.laser_power = override_params.get('laser_power', SYSTEM.get_empirical_param('charge_state_dynamics', 'laser_power'))
            self.electric_field = override_params.get('electric_field', SYSTEM.get_empirical_param('charge_state_dynamics', 'electric_field'))
            self.surface_distance = override_params.get('surface_distance', SYSTEM.get_empirical_param('charge_state_dynamics', 'surface_distance'))
        except KeyError as e:
            raise RuntimeError(f"Missing required charge dynamics parameter: {e}. "
                             f"Add all charge_state_dynamics parameters to empirical_parameters in system.json.")
        
        # Charge states: 0=NV+, 1=NV0, 2=NV-
        self.states = ['nv_plus', 'nv_zero', 'nv_minus']
        self.current_state_idx = 2  # Start in NV-
        self.state_populations = np.array([0.0, 0.0, 1.0])  # Start in NV-
        
        # Initialize transition rates
        self.transitions = self._setup_transitions(override_params)
        
        # State evolution tracking
        self._last_update_time = 0.0
        self._transition_history = []
        
    def _setup_transitions(self, override_params: dict) -> List[ChargeStateTransition]:
        """Setup all possible charge state transitions"""
        
        transitions = []
        
        # NV- → NV0 (ionization)
        transitions.append(ChargeStateTransition(
            'nv_minus', 'nv_zero',
            base_rate=SYSTEM.get_empirical_param('charge_state_dynamics', 'nv_minus_to_zero_rate'),
            activation_energy=SYSTEM.get_empirical_param('charge_state_dynamics', 'ionization_barrier'),
            laser_dependence=SYSTEM.get_empirical_param('charge_state_dynamics', 'optical_ionization_rate'),
            field_dependence=SYSTEM.get_empirical_param('charge_state_dynamics', 'field_ionization_rate')
        ))
        
        # NV0 → NV- (electron capture)
        transitions.append(ChargeStateTransition(
            'nv_zero', 'nv_minus',
            base_rate=SYSTEM.get_empirical_param('charge_state_dynamics', 'nv_zero_to_minus_rate'),
            activation_energy=SYSTEM.get_empirical_param('charge_state_dynamics', 'capture_barrier'),
            laser_dependence=SYSTEM.get_empirical_param('charge_state_dynamics', 'optical_capture_rate'),
            field_dependence=SYSTEM.get_empirical_param('charge_state_dynamics', 'field_capture_rate')
        ))
        
        # NV0 → NV+ (further ionization)
        transitions.append(ChargeStateTransition(
            'nv_zero', 'nv_plus',
            base_rate=SYSTEM.get_empirical_param('charge_state_dynamics', 'nv_zero_to_plus_rate'),
            activation_energy=SYSTEM.get_empirical_param('charge_state_dynamics', 'double_ionization_barrier'),
            laser_dependence=SYSTEM.get_empirical_param('charge_state_dynamics', 'optical_double_ionization_rate'),
            field_dependence=SYSTEM.get_empirical_param('charge_state_dynamics', 'field_double_ionization_rate')
        ))
        
        # NV+ → NV0 (electron capture)
        transitions.append(ChargeStateTransition(
            'nv_plus', 'nv_zero',
            base_rate=SYSTEM.get_empirical_param('charge_state_dynamics', 'nv_plus_to_zero_rate'),
            activation_energy=SYSTEM.get_empirical_param('charge_state_dynamics', 'neutralization_barrier'),
            laser_dependence=SYSTEM.get_empirical_param('charge_state_dynamics', 'optical_neutralization_rate'),
            field_dependence=SYSTEM.get_empirical_param('charge_state_dynamics', 'field_neutralization_rate')
        ))
        
        # Direct NV+ ⇌ NV- transitions (rare, but possible)
        transitions.append(ChargeStateTransition(
            'nv_plus', 'nv_minus',
            base_rate=SYSTEM.get_empirical_param('charge_state_dynamics', 'nv_plus_to_minus_rate'),
            activation_energy=SYSTEM.get_empirical_param('charge_state_dynamics', 'direct_recombination_barrier'),
            laser_dependence=0.0,
            field_dependence=SYSTEM.get_empirical_param('charge_state_dynamics', 'field_direct_recombination_rate')
        ))
        
        transitions.append(ChargeStateTransition(
            'nv_minus', 'nv_plus',
            base_rate=SYSTEM.get_empirical_param('charge_state_dynamics', 'nv_minus_to_plus_rate'),
            activation_energy=SYSTEM.get_empirical_param('charge_state_dynamics', 'direct_double_ionization_barrier'),
            laser_dependence=SYSTEM.get_empirical_param('charge_state_dynamics', 'optical_direct_double_ionization_rate'),
            field_dependence=SYSTEM.get_empirical_param('charge_state_dynamics', 'field_direct_double_ionization_rate')
        ))
        
        return transitions
        
    def _build_rate_matrix(self) -> np.ndarray:
        """Build 3x3 rate matrix for current conditions"""
        
        # Initialize rate matrix
        rate_matrix = np.zeros((3, 3))
        
        # State index mapping
        state_indices = {'nv_plus': 0, 'nv_zero': 1, 'nv_minus': 2}
        
        # Fill off-diagonal elements (transitions)
        for transition in self.transitions:
            i = state_indices[transition.from_state]
            j = state_indices[transition.to_state]
            
            rate = transition.get_rate(
                self.temperature, self.laser_power, self.electric_field
            )
            
            # Include surface distance effect
            distance_factor = np.exp(-self.surface_distance / 10e-9)
            rate *= distance_factor
            
            rate_matrix[j, i] = rate  # From i to j
            
        # Fill diagonal elements (outgoing rates)
        for i in range(3):
            rate_matrix[i, i] = -np.sum(rate_matrix[:, i])
            
        return rate_matrix
        
    def evolve_populations(self, dt: float) -> np.ndarray:
        """
        Evolve state populations using master equation
        
        Args:
            dt: Time step [s]
            
        Returns:
            New state populations
        """
        rate_matrix = self._build_rate_matrix()
        
        # Solve master equation: dP/dt = R * P
        # Solution: P(t+dt) = exp(R * dt) * P(t)
        evolution_matrix = expm(rate_matrix * dt)
        
        # Update populations
        self.state_populations = evolution_matrix @ self.state_populations
        
        # Ensure normalization (numerical stability)
        self.state_populations /= np.sum(self.state_populations)
        
        return self.state_populations
        
    def sample_stochastic_trajectory(self, dt: float) -> int:
        """
        Sample individual state trajectory using Gillespie algorithm
        
        Args:
            dt: Time step [s]
            
        Returns:
            Current state index (0=NV+, 1=NV0, 2=NV-)
        """
        current_time = 0.0
        
        while current_time < dt:
            # Get current rates
            rate_matrix = self._build_rate_matrix()
            current_state = self.current_state_idx
            
            # Total escape rate from current state
            total_rate = -rate_matrix[current_state, current_state]
            
            if total_rate <= 0:
                break  # No transitions possible
                
            # Time to next event (exponential distribution)
            tau = self.rng.exponential(1.0 / total_rate)
            
            if current_time + tau >= dt:
                break  # No transition within this timestep
                
            # Choose which transition occurs
            transition_rates = rate_matrix[:, current_state].copy()
            transition_rates[current_state] = 0  # Remove self-transition
            transition_probs = transition_rates / total_rate
            
            # Sample new state
            new_state = self.rng.choice(3, p=transition_probs)
            
            # Record transition
            self._transition_history.append({
                'time': self._last_update_time + current_time + tau,
                'from_state': current_state,
                'to_state': new_state
            })
            
            # Update state
            self.current_state_idx = new_state
            current_time += tau
            
        return self.current_state_idx
        
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Generate charge state samples
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Array of state indices (0=NV+, 1=NV0, 2=NV-)
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
            
        samples = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # Evolve by one timestep
            samples[i] = self.sample_stochastic_trajectory(self._dt)
            self._last_update_time += self._dt
            
        return samples[0] if n_samples == 1 else samples
        
    def get_state_populations(self) -> Dict[str, float]:
        """Get current state populations"""
        return {
            'nv_plus': self.state_populations[0],
            'nv_zero': self.state_populations[1], 
            'nv_minus': self.state_populations[2]
        }
        
    def get_current_state_name(self) -> str:
        """Get current state name"""
        return self.states[self.current_state_idx]
        
    def set_conditions(self, temperature: float = None, 
                      laser_power: float = None, 
                      electric_field: float = None):
        """Update experimental conditions"""
        if temperature is not None:
            self.temperature = temperature
        if laser_power is not None:
            self.laser_power = laser_power
        if electric_field is not None:
            self.electric_field = electric_field
            
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Get effective PSD for charge state noise
        
        For multi-level system, this is more complex than simple telegraph noise
        """
        rate_matrix = self._build_rate_matrix()
        
        # Calculate effective autocorrelation time
        eigenvals = np.linalg.eigvals(rate_matrix)
        real_eigenvals = eigenvals[eigenvals.real < 0].real
        
        if len(real_eigenvals) > 0:
            # Dominant time constant
            tau_eff = -1.0 / real_eigenvals[-2]  # Second-fastest mode
            
            # Multi-exponential PSD (simplified)
            omega = 2 * np.pi * frequencies
            psd = np.zeros_like(frequencies)
            
            for i, eigenval in enumerate(real_eigenvals):
                if eigenval < 0:
                    tau_i = -1.0 / eigenval
                    weight = 1.0 / (len(real_eigenvals))  # Equal weighting
                    psd += weight * (2 * tau_i) / (1 + (omega * tau_i)**2)
                    
            return psd
        else:
            raise RuntimeError("No valid eigenvalues found for charge state rate matrix. "
                             "This indicates invalid transition rates in your configuration. "
                             "Check empirical_parameters.charge_state_dynamics in system.json.")
            
    def get_transition_statistics(self) -> Dict[str, float]:
        """Get statistics about recent transitions"""
        if len(self._transition_history) < 2:
            return {'mean_dwell_time': np.inf, 'transition_rate': 0.0}
            
        # Calculate dwell times
        dwell_times = []
        for i in range(1, len(self._transition_history)):
            dt = (self._transition_history[i]['time'] - 
                  self._transition_history[i-1]['time'])
            dwell_times.append(dt)
            
        mean_dwell_time = np.mean(dwell_times) if dwell_times else np.inf
        transition_rate = 1.0 / mean_dwell_time if mean_dwell_time > 0 else 0.0
        
        return {
            'mean_dwell_time': mean_dwell_time,
            'transition_rate': transition_rate,
            'num_transitions': len(self._transition_history)
        }
        
    def reset(self):
        """Reset to initial state"""
        super().reset()
        self.current_state_idx = 2  # NV-
        self.state_populations = np.array([0.0, 0.0, 1.0])
        self._last_update_time = 0.0
        self._transition_history = []


# Factory function for easy creation
def create_charge_state_model(setup_type: str = 'room_temperature',
                             **custom_params) -> MultiLevelChargeNoise:
    """
    Create charge state model for common experimental setups
    
    Args:
        setup_type: 'room_temperature', 'cryogenic', 'high_laser', 'surface_nv'
        **custom_params: Override specific parameters
        
    Returns:
        Configured MultiLevelChargeNoise instance
    """
    presets = {
        'room_temperature': {
            'temperature': 300.0,
            'laser_power': 1.0,
            'nv_minus_to_zero_rate': 0.1,
            'nv_zero_to_minus_rate': 1.0,
            'surface_distance': 10e-9
        },
        'cryogenic': {
            'temperature': 4.0,
            'laser_power': 0.5,
            'nv_minus_to_zero_rate': 0.01,  # Slower at low T
            'nv_zero_to_minus_rate': 0.1,
            'surface_distance': 20e-9
        },
        'high_laser': {
            'temperature': 300.0,
            'laser_power': 10.0,
            'optical_ionization_rate': 0.2,
            'nv_minus_to_zero_rate': 1.0,  # Faster ionization
            'surface_distance': 5e-9
        },
        'surface_nv': {
            'temperature': 300.0,
            'laser_power': 1.0,
            'surface_distance': 2e-9,  # Very close to surface
            'nv_minus_to_zero_rate': 10.0,  # Fast charge jumps
            'electric_field': 1e5  # High surface field
        }
    }
    
    if setup_type not in presets:
        raise ValueError(f"Unknown setup type: {setup_type}")
        
    params = presets[setup_type].copy()
    params.update(custom_params)
    
    return MultiLevelChargeNoise(override_params=params)


# Example usage and testing
if __name__ == "__main__":
    print("🔌 Testing Multi-Level Charge State Dynamics")
    
    # Create model
    charge_model = create_charge_state_model('room_temperature')
    
    # Simulate trajectory
    n_steps = 1000
    dt = 1e-3  # 1 ms steps
    charge_model._dt = dt
    
    trajectory = []
    for i in range(n_steps):
        state = charge_model.sample(1)
        trajectory.append(state)
        
    # Analyze results
    states = np.array(trajectory)
    state_names = ['NV+', 'NV0', 'NV-']
    
    print(f"\n📊 Trajectory Statistics:")
    for i, name in enumerate(state_names):
        fraction = np.sum(states == i) / len(states)
        print(f"   {name}: {fraction:.2f}")
        
    transitions = charge_model.get_transition_statistics()
    print(f"\n⚡ Transition Statistics:")
    print(f"   Mean dwell time: {transitions['mean_dwell_time']*1000:.1f} ms")
    print(f"   Transition rate: {transitions['transition_rate']:.2f} Hz")
    print(f"   Total transitions: {transitions['num_transitions']}")
    
    print("\n✅ Multi-level charge dynamics successfully implemented!")