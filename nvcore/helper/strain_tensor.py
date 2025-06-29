"""
Tensor Strain Coupling for NV Centers

Implements realistic strain tensor coupling following C3v symmetry
instead of simplified scalar coupling. Includes both D and E parameter shifts.

Physical Background:
- NV center has C3v point group symmetry
- Strain tensor couples to both axial (D) and transverse (E) zero-field parameters
- d_parallel and d_perp coupling constants from experimental measurements

References:
- Doherty et al., Phys. Rep. 528, 1 (2013)
- Maze et al., New J. Phys. 13, 025025 (2011)
"""

import numpy as np
from typing import Dict, List, Optional, Union, Tuple
from abc import ABC, abstractmethod

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))
from noise_sources import SYSTEM, NoiseSource


class StrainTensor:
    """
    Represents a 3x3 strain tensor with proper symmetry properties
    """
    
    def __init__(self, strain_matrix: np.ndarray = None):
        """
        Initialize strain tensor
        
        Args:
            strain_matrix: 3x3 strain tensor (symmetric)
        """
        if strain_matrix is None:
            strain_matrix = np.zeros((3, 3))
        
        # Ensure symmetry
        self.tensor = 0.5 * (strain_matrix + strain_matrix.T)
        
    @classmethod
    def from_components(cls, exx: float, eyy: float, ezz: float,
                       exy: float = 0, exz: float = 0, eyz: float = 0) -> 'StrainTensor':
        """Create strain tensor from components"""
        matrix = np.array([
            [exx, exy, exz],
            [exy, eyy, eyz], 
            [exz, eyz, ezz]
        ])
        return cls(matrix)
    
    @classmethod
    def from_voigt(cls, voigt_vector: np.ndarray) -> 'StrainTensor':
        """
        Create strain tensor from Voigt notation [exx, eyy, ezz, eyz, exz, exy]
        """
        if len(voigt_vector) != 6:
            raise ValueError("Voigt vector must have 6 components")
            
        exx, eyy, ezz, eyz, exz, exy = voigt_vector
        return cls.from_components(exx, eyy, ezz, exy, exz, eyz)
    
    def to_voigt(self) -> np.ndarray:
        """Convert to Voigt notation"""
        return np.array([
            self.tensor[0, 0],  # exx
            self.tensor[1, 1],  # eyy
            self.tensor[2, 2],  # ezz
            self.tensor[1, 2],  # eyz
            self.tensor[0, 2],  # exz
            self.tensor[0, 1]   # exy
        ])
    
    def get_nv_relevant_combinations(self) -> Dict[str, float]:
        """
        Get strain combinations relevant for NV center (C3v symmetry)
        
        Returns:
            Dictionary with relevant strain combinations
        """
        exx = self.tensor[0, 0]
        eyy = self.tensor[1, 1] 
        ezz = self.tensor[2, 2]
        exy = self.tensor[0, 1]
        exz = self.tensor[0, 2]
        eyz = self.tensor[1, 2]
        
        return {
            # Axial strain (affects D parameter)
            'e_parallel': ezz,
            'e_perp_sum': exx + eyy,
            'e_trace': exx + eyy + ezz,
            
            # Transverse strain (affects E parameter)
            'e_perp_diff': exx - eyy,
            'e_xy': exy,
            'e_xz': exz,
            'e_yz': eyz,
            
            # Hydrostatic strain
            'e_hydrostatic': (exx + eyy + ezz) / 3,
            
            # Deviatoric strain
            'e_dev_zz': ezz - (exx + eyy + ezz) / 3
        }
    
    def __add__(self, other: 'StrainTensor') -> 'StrainTensor':
        """Add two strain tensors"""
        return StrainTensor(self.tensor + other.tensor)
    
    def __mul__(self, scalar: float) -> 'StrainTensor':
        """Multiply strain tensor by scalar"""
        return StrainTensor(self.tensor * scalar)
    
    def __rmul__(self, scalar: float) -> 'StrainTensor':
        """Right multiplication by scalar"""
        return self.__mul__(scalar)


class StrainTensorNoise(NoiseSource):
    """
    Realistic strain noise using full tensor coupling
    
    Implements proper C3v symmetry for NV centers with separate
    couplings to D and E parameters
    """
    
    def __init__(self, rng: Optional[np.random.Generator] = None,
                 override_params: Optional[dict] = None):
        """Initialize tensor strain noise model"""
        if override_params is None:
            override_params = {}
        super().__init__(rng)
        
        # Get strain coupling parameters
        self.d_parallel = self.get_empirical_param('strain_tensor', 'd_parallel_coupling')
        self.d_perp = self.get_empirical_param('strain_tensor', 'd_perp_coupling') 
        self.e_coupling = self.get_empirical_param('strain_tensor', 'e_parameter_coupling')
        
        # Override with any provided parameters
        self.d_parallel = override_params.get('d_parallel_coupling', self.d_parallel)
        self.d_perp = override_params.get('d_perp_coupling', self.d_perp)
        self.e_coupling = override_params.get('e_parameter_coupling', self.e_coupling)
        
        # Noise characteristics
        self.strain_amplitude = override_params.get('strain_amplitude', 1e-6)
        self.correlation_time = override_params.get('correlation_time', 1e-3)  # 1 ms
        self.resonance_frequency = override_params.get('resonance_frequency', 100.0)  # Hz
        self.resonance_amplitude = override_params.get('resonance_amplitude', 1e-7)
        
        # Temperature effects
        self.temperature = override_params.get('temperature', 300.0)
        self.thermal_expansion = override_params.get('thermal_expansion', 1e-6)  # /K
        
        # Current strain state
        self.current_strain = StrainTensor()
        self.strain_velocity = StrainTensor()  # For correlated dynamics
        
        # Static strain offset
        static_strain = override_params.get('static_strain_tensor', None)
        if static_strain is not None:
            self.static_strain = StrainTensor(static_strain)
        else:
            # Default: small biaxial strain
            self.static_strain = StrainTensor.from_components(
                exx=self.strain_amplitude * 0.1,
                eyy=self.strain_amplitude * 0.1, 
                ezz=-self.strain_amplitude * 0.2,  # Poisson effect
            )
        
    def get_empirical_param(self, category: str, name: str) -> float:
        """Get empirical parameter from system config - no fallbacks allowed"""
        try:
            return SYSTEM.get_noise_param('strain', category, name)
        except KeyError as e:
            raise RuntimeError(f"Missing required strain parameter '{category}.{name}' in system.json. "
                             f"This parameter must be measured empirically for your specific setup. "
                             f"Add it to system.json under noise_parameters.mechanical.strain or "
                             f"empirical_parameters.strain_tensor. Error: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load strain parameter '{category}.{name}': {e}")
    
    def evolve_strain_dynamics(self, dt: float):
        """
        Evolve strain tensor using Ornstein-Uhlenbeck process
        
        Args:
            dt: Time step in seconds
        """
        # Damping factor
        gamma = 1.0 / self.correlation_time
        decay = np.exp(-gamma * dt)
        
        # Thermal noise strength
        sigma = self.strain_amplitude * np.sqrt(2 * gamma)
        noise_factor = np.sqrt(1 - decay**2)
        
        # Generate noise for all tensor components
        noise_tensor = StrainTensor.from_voigt(
            self.rng.normal(0, sigma * noise_factor, 6)
        )
        
        # Update strain with OU dynamics
        self.current_strain = decay * self.current_strain + noise_tensor
        
        # Add resonant component (mechanical resonances)
        time = getattr(self, '_current_time', 0.0)
        resonant_amplitude = self.resonance_amplitude * np.sin(
            2 * np.pi * self.resonance_frequency * time
        )
        
        # Add to diagonal components (typical for resonances)
        resonant_strain = StrainTensor.from_components(
            exx=resonant_amplitude,
            eyy=resonant_amplitude,
            ezz=-2 * resonant_amplitude  # Volume conservation
        )
        
        self.current_strain = self.current_strain + resonant_strain
        
        # Update time
        if not hasattr(self, '_current_time'):
            self._current_time = 0.0
        self._current_time += dt
    
    def get_current_strain_tensor(self) -> StrainTensor:
        """Get current total strain tensor"""
        return self.static_strain + self.current_strain
    
    def get_zfs_shifts(self, strain_tensor: StrainTensor = None) -> Dict[str, float]:
        """
        Calculate zero-field splitting shifts from strain tensor
        
        Args:
            strain_tensor: Strain tensor (uses current if None)
            
        Returns:
            Dictionary with D and E parameter shifts
        """
        if strain_tensor is None:
            strain_tensor = self.get_current_strain_tensor()
        
        strain_components = strain_tensor.get_nv_relevant_combinations()
        
        # D parameter shift (axial zero-field splitting)
        # ΔD = d_∥(ε_zz - ε_perp) + d_⊥(hydrostatic terms)
        delta_d = (
            self.d_parallel * (strain_components['e_parallel'] - 
                              strain_components['e_perp_sum'] / 2) +
            self.d_perp * strain_components['e_hydrostatic']
        )
        
        # E parameter (transverse zero-field splitting)
        # ΔE = e_coupling * √[(ε_xx - ε_yy)² + 4ε_xy²]
        e_magnitude = np.sqrt(
            strain_components['e_perp_diff']**2 + 
            4 * strain_components['e_xy']**2
        )
        delta_e = self.e_coupling * e_magnitude
        
        # E parameter orientation
        if strain_components['e_perp_diff'] != 0 or strain_components['e_xy'] != 0:
            e_angle = 0.5 * np.arctan2(
                2 * strain_components['e_xy'],
                strain_components['e_perp_diff']
            )
        else:
            e_angle = 0.0
        
        return {
            'delta_d': delta_d,
            'delta_e': delta_e,
            'e_angle': e_angle,
            'strain_components': strain_components
        }
    
    def get_hamiltonian_perturbation(self, spin_operators: Dict[str, np.ndarray],
                                   strain_tensor: StrainTensor = None) -> np.ndarray:
        """
        Calculate strain Hamiltonian perturbation
        
        Args:
            spin_operators: Dictionary with spin operators
            strain_tensor: Strain tensor (uses current if None)
            
        Returns:
            Strain Hamiltonian matrix
        """
        if strain_tensor is None:
            strain_tensor = self.get_current_strain_tensor()
        
        # Get ZFS shifts
        shifts = self.get_zfs_shifts(strain_tensor)
        
        # Build Hamiltonian perturbation
        H_strain = np.zeros_like(spin_operators['Sz'])
        
        # D parameter shift: ΔD (Sz² - 2/3 I)
        if 'Sz' in spin_operators:
            S_z2 = spin_operators['Sz'] @ spin_operators['Sz']
            dim = S_z2.shape[0]
            H_strain += shifts['delta_d'] * (S_z2 - 2/3 * np.eye(dim))
        
        # E parameter: ΔE (Sx² - Sy²) cos(2φ) + ΔE (SxSy + SySx) sin(2φ)
        if shifts['delta_e'] > 0 and 'Sx' in spin_operators and 'Sy' in spin_operators:
            Sx = spin_operators['Sx']
            Sy = spin_operators['Sy']
            angle = shifts['e_angle']
            
            # E-field terms
            H_strain += shifts['delta_e'] * (
                np.cos(2 * angle) * (Sx @ Sx - Sy @ Sy) +
                np.sin(2 * angle) * (Sx @ Sy + Sy @ Sx)
            )
        
        return H_strain
    
    def sample(self, n_samples: int = 1) -> Union[StrainTensor, List[StrainTensor]]:
        """
        Generate strain tensor samples
        
        Args:
            n_samples: Number of samples
            
        Returns:
            StrainTensor or list of StrainTensors
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        
        samples = []
        for _ in range(n_samples):
            # Evolve dynamics
            self.evolve_strain_dynamics(self._dt)
            samples.append(self.get_current_strain_tensor())
        
        return samples[0] if n_samples == 1 else samples
    
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """
        Get strain noise power spectral density
        
        This is for the effective scalar strain affecting the D parameter
        """
        omega = 2 * np.pi * frequencies
        tau = self.correlation_time
        
        # Base Lorentzian from OU process
        base_psd = 2 * self.strain_amplitude**2 * tau / (1 + (omega * tau)**2)
        
        # Add resonant peak
        resonance_psd = np.zeros_like(frequencies)
        omega_res = 2 * np.pi * self.resonance_frequency
        Q_factor = 10  # Typical mechanical Q
        
        # Lorentzian resonance
        gamma_res = omega_res / Q_factor
        resonance_psd = (
            (self.resonance_amplitude**2 * gamma_res**2) /
            ((omega**2 - omega_res**2)**2 + (omega * gamma_res)**2)
        )
        
        return base_psd + resonance_psd
    
    def get_strain_statistics(self) -> Dict[str, float]:
        """Get statistics about current strain state"""
        current = self.get_current_strain_tensor()
        shifts = self.get_zfs_shifts(current)
        components = current.get_nv_relevant_combinations()
        
        return {
            'rms_strain': np.sqrt(np.mean(current.tensor**2)),
            'trace_strain': components['e_trace'],
            'hydrostatic_strain': components['e_hydrostatic'], 
            'delta_d_hz': shifts['delta_d'],
            'delta_e_hz': shifts['delta_e'],
            'e_parameter_angle': shifts['e_angle']
        }
    
    def set_temperature(self, temperature: float):
        """Update temperature and thermal strain"""
        delta_t = temperature - self.temperature
        
        # Add thermal expansion strain
        thermal_strain = StrainTensor.from_components(
            exx=self.thermal_expansion * delta_t,
            eyy=self.thermal_expansion * delta_t,
            ezz=self.thermal_expansion * delta_t
        )
        
        self.static_strain = self.static_strain + thermal_strain
        self.temperature = temperature
    
    def reset(self):
        """Reset to initial state"""
        super().reset()
        self.current_strain = StrainTensor()
        self.strain_velocity = StrainTensor()
        self._current_time = 0.0


# Factory functions for common setups
def create_bulk_diamond_strain(temperature: float = 300.0) -> StrainTensorNoise:
    """Create strain model for bulk diamond NV"""
    return StrainTensorNoise(override_params={
        'strain_amplitude': 1e-7,  # Low strain in bulk
        'correlation_time': 10e-3,  # Slower dynamics
        'resonance_frequency': 50.0,  # Low frequency phonons
        'temperature': temperature
    })

def create_nanodiamond_strain(size_nm: float = 100.0) -> StrainTensorNoise:
    """Create strain model for nanodiamond"""
    # Smaller particles have higher strain and faster dynamics
    strain_scale = (100.0 / size_nm) ** 0.5
    
    return StrainTensorNoise(override_params={
        'strain_amplitude': 1e-6 * strain_scale,
        'correlation_time': 1e-4 / strain_scale,  # Faster for smaller particles
        'resonance_frequency': 1000.0 * strain_scale,  # Higher frequency modes
        'temperature': 300.0
    })

def create_surface_nv_strain(depth_nm: float = 10.0) -> StrainTensorNoise:
    """Create strain model for surface NV"""
    # Surface strain increases as 1/depth
    strain_scale = 10.0 / depth_nm
    
    return StrainTensorNoise(override_params={
        'strain_amplitude': 1e-6 * strain_scale,
        'correlation_time': 1e-3,
        'resonance_frequency': 200.0,  # Surface resonances
        'temperature': 300.0,
        'static_strain_tensor': np.array([  # Biaxial surface strain
            [1e-5 * strain_scale, 0, 0],
            [0, 1e-5 * strain_scale, 0],
            [0, 0, -2e-5 * strain_scale]
        ])
    })


# Example usage and testing
if __name__ == "__main__":
    print("🔧 Testing Tensor Strain Coupling")
    
    # Create strain model
    strain_model = create_bulk_diamond_strain()
    strain_model._dt = 1e-4  # 0.1 ms timestep
    
    # Test tensor operations
    test_tensor = StrainTensor.from_components(1e-6, 2e-6, -3e-6, 0.5e-6, 0, 0)
    print(f"\n📊 Test Strain Tensor:")
    print(f"   Voigt form: {test_tensor.to_voigt()}")
    
    components = test_tensor.get_nv_relevant_combinations()
    print(f"   NV-relevant combinations:")
    for key, value in components.items():
        print(f"     {key}: {value:.2e}")
    
    # Test ZFS shifts
    shifts = strain_model.get_zfs_shifts(test_tensor)
    print(f"\n⚡ ZFS Shifts:")
    print(f"   ΔD: {shifts['delta_d']/1e9:.2f} GHz")
    print(f"   ΔE: {shifts['delta_e']/1e6:.2f} MHz")
    print(f"   E angle: {np.degrees(shifts['e_angle']):.1f}°")
    
    # Test dynamics
    print(f"\n🔄 Strain Dynamics:")
    stats_history = []
    for i in range(100):
        strain_model.sample(1)
        if i % 20 == 0:
            stats = strain_model.get_strain_statistics()
            stats_history.append(stats)
            print(f"   t={i*0.1:.1f}ms: RMS={stats['rms_strain']:.2e}, ΔD={stats['delta_d_hz']/1e6:.1f} MHz")
    
    print("\n✅ Tensor strain coupling successfully implemented!")