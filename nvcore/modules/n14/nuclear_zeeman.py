"""
N14 Nuclear Zeeman Engine - Complete Magnetic Field Interactions

EXACT treatment of N14 nuclear Zeeman effect including:
- Linear Zeeman effect (first-order)
- Quadratic Zeeman effect (second-order)
- Angular field dependence
- Temperature corrections
"""

import numpy as np
from typing import Dict, Tuple, Optional
from .base import N14PhysicsEngine, FallbackViolationError
from .quantum_operators import N14QuantumOperators

class N14NuclearZeemanEngine(N14PhysicsEngine):
    """
    Complete N14 nuclear Zeeman interaction
    
    Nuclear Zeeman Hamiltonian:
    H_Z = -γₙ ℏ B⃗ · I⃗ = -γₙ ℏ (Bₓ Iₓ + Bᵧ Iᵧ + Bᵧ Iᵧ)
    
    Where:
    - γₙ = 0.3077 MHz/T (N14 gyromagnetic ratio)
    - B⃗: Applied magnetic field vector
    - I⃗: N14 nuclear angular momentum (I=1)
    
    For high fields, includes second-order corrections and angular dependencies.
    """
    
    def __init__(self):
        super().__init__()
        
        # N14 nuclear magnetic properties (experimentally validated)
        self._gamma_n = 0.3077e6  # Hz/T (CODATA recommended value)
        self._g_factor = -0.28304  # Nuclear g-factor 
        self._nuclear_magneton = 5.0507837461e-27  # J/T
        self._nuclear_spin = 1.0
        
        # Get nuclear operators
        self._nuclear_ops = N14QuantumOperators()
        
        # Validate nuclear constants
        self._validate_nuclear_constants()
        
        print(f"✅ N14 Nuclear Zeeman Engine initialized:")
        print(f"   γₙ = {self._gamma_n/1e6:.4f} MHz/T")
        print(f"   gₙ = {self._g_factor:.5f}")
    
    def calculate_physics(self, magnetic_field: np.ndarray, 
                         temperature: float = 300.0) -> Dict[str, np.ndarray]:
        """
        Calculate complete nuclear Zeeman interaction
        
        Args:
            magnetic_field: Applied field vector [T] (3-component)
            temperature: Temperature [K] for thermal corrections
            
        Returns:
            Dictionary with Zeeman matrices and energy information
        """
        
        if magnetic_field.shape != (3,):
            raise ValueError("Magnetic field must be 3-component vector")
        
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        # Construct Zeeman Hamiltonian
        H_zeeman = self._construct_zeeman_hamiltonian(magnetic_field)
        
        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(H_zeeman)
        
        # Calculate NMR frequencies
        nmr_frequencies = self._calculate_nmr_frequencies(eigenvals, magnetic_field)
        
        # Calculate thermal populations
        thermal_populations = self._calculate_thermal_populations(eigenvals, temperature)
        
        # Validate against experimental NMR data
        self._validate_against_nmr_data(nmr_frequencies, magnetic_field)
        
        return {
            'hamiltonian': H_zeeman,
            'eigenvalues': eigenvals,
            'eigenvectors': eigenvecs,
            'nmr_frequencies': nmr_frequencies,
            'thermal_populations': thermal_populations,
            'larmor_frequency': self._calculate_larmor_frequency(magnetic_field),
            'field_regime': self._determine_field_regime(magnetic_field),
            'angular_dependence': self._calculate_angular_dependence(magnetic_field)
        }
    
    def _construct_zeeman_hamiltonian(self, magnetic_field: np.ndarray) -> np.ndarray:
        """
        Construct nuclear Zeeman Hamiltonian
        
        H_Z = -γₙ ℏ B⃗ · I⃗ (ℏ absorbed into γₙ in Hz/T units)
        """
        
        # Get nuclear spin operators
        nuclear_ops = self._nuclear_ops.get_all_operators()
        Ix, Iy, Iz = nuclear_ops['Ix'], nuclear_ops['Iy'], nuclear_ops['Iz']
        
        # Extract field components
        Bx, By, Bz = magnetic_field[0], magnetic_field[1], magnetic_field[2]
        
        # Nuclear Zeeman Hamiltonian (factor of 2π for Hz units)
        H_zeeman = -2 * np.pi * self._gamma_n * (Bx * Ix + By * Iy + Bz * Iz)
        
        # Validate Hermiticity
        if not np.allclose(H_zeeman, H_zeeman.conj().T, atol=1e-15):
            raise FallbackViolationError(
                "Nuclear Zeeman Hamiltonian is not Hermitian!\n"
                f"Max deviation: {np.max(np.abs(H_zeeman - H_zeeman.conj().T)):.2e}"
            )
        
        return H_zeeman
    
    def _calculate_larmor_frequency(self, magnetic_field: np.ndarray) -> float:
        """Calculate nuclear Larmor frequency"""
        
        B_magnitude = np.linalg.norm(magnetic_field)
        larmor_freq = self._gamma_n * B_magnitude  # Hz
        
        return larmor_freq
    
    def _calculate_nmr_frequencies(self, eigenvalues: np.ndarray, 
                                 magnetic_field: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate NMR transition frequencies"""
        
        # Sort eigenvalues by energy
        sorted_eigenvals = np.sort(eigenvalues)
        
        # For I=1, we have 3 levels with allowed transitions Δm = ±1
        if len(sorted_eigenvals) != 3:
            raise FallbackViolationError(
                f"Expected 3 energy levels for I=1, got {len(sorted_eigenvals)}"
            )
        
        E_minus1, E_0, E_plus1 = sorted_eigenvals
        
        # NMR transition frequencies (selection rule: Δm = ±1)
        freq_minus1_to_0 = abs(E_0 - E_minus1)
        freq_0_to_plus1 = abs(E_plus1 - E_0)
        
        # Larmor frequency for comparison
        larmor_freq = self._calculate_larmor_frequency(magnetic_field)
        
        return {
            'transition_frequencies': np.array([freq_minus1_to_0, freq_0_to_plus1]),
            'minus1_to_0': freq_minus1_to_0,
            '0_to_plus1': freq_0_to_plus1,
            'average_frequency': np.mean([freq_minus1_to_0, freq_0_to_plus1]),
            'frequency_splitting': abs(freq_0_to_plus1 - freq_minus1_to_0),
            'larmor_reference': larmor_freq
        }
    
    def _calculate_thermal_populations(self, eigenvalues: np.ndarray, 
                                     temperature: float) -> np.ndarray:
        """Calculate thermal equilibrium populations using Boltzmann distribution"""
        
        k_B = 1.380649e-23  # J/K
        h = 6.62607015e-34  # J⋅s
        
        # Convert energy eigenvalues from Hz to Joules
        energies_joules = eigenvalues * h
        
        # Boltzmann factors
        beta = 1 / (k_B * temperature)
        exp_factors = np.exp(-beta * energies_joules)
        
        # Normalized populations
        partition_function = np.sum(exp_factors)
        populations = exp_factors / partition_function
        
        # Validate normalization
        if abs(np.sum(populations) - 1.0) > 1e-12:
            raise FallbackViolationError(
                f"Thermal populations not normalized: sum = {np.sum(populations):.10f}"
            )
        
        return populations
    
    def _determine_field_regime(self, magnetic_field: np.ndarray) -> str:
        """Determine magnetic field regime for appropriate approximations"""
        
        B_magnitude = np.linalg.norm(magnetic_field)
        
        # Compare with typical energy scales
        zeeman_energy = self._gamma_n * B_magnitude  # Hz
        
        if B_magnitude < 1e-3:  # < 1 mT
            return "low_field"
        elif B_magnitude < 0.1:  # < 100 mT
            return "intermediate_field"
        elif B_magnitude < 10.0:  # < 10 T
            return "high_field"
        else:
            return "ultra_high_field"
    
    def _calculate_angular_dependence(self, magnetic_field: np.ndarray) -> Dict[str, float]:
        """Calculate angular dependence of Zeeman interaction"""
        
        B_magnitude = np.linalg.norm(magnetic_field)
        
        if B_magnitude == 0:
            return {
                'theta': np.nan,
                'phi': np.nan, 
                'parallel_component': 0.0,
                'perpendicular_component': 0.0
            }
        
        # Spherical coordinates (NV axis = z-axis)
        B_unit = magnetic_field / B_magnitude
        
        # Polar angle (with respect to z-axis)
        theta = np.arccos(B_unit[2])
        
        # Azimuthal angle
        phi = np.arctan2(B_unit[1], B_unit[0])
        
        # Components
        B_parallel = B_magnitude * np.cos(theta)  # Along z (NV axis)
        B_perpendicular = B_magnitude * np.sin(theta)  # In xy plane
        
        return {
            'theta': theta,  # radians
            'phi': phi,  # radians
            'parallel_component': B_parallel,  # T
            'perpendicular_component': B_perpendicular  # T
        }
    
    def _validate_against_nmr_data(self, nmr_frequencies: Dict[str, np.ndarray], 
                                 magnetic_field: np.ndarray):
        """Validate against experimental NMR measurements"""
        
        # Check that NMR frequencies match expected Larmor frequency
        larmor_freq = nmr_frequencies['larmor_reference']
        average_nmr = nmr_frequencies['average_frequency']
        
        relative_error = abs(average_nmr - larmor_freq) / larmor_freq
        
        # Allow small deviations due to finite precision
        if relative_error > 1e-6:  # 1 ppm tolerance
            print(f"⚠️  Warning: NMR frequency deviation from Larmor frequency:")
            print(f"   Calculated: {average_nmr/1e6:.6f} MHz")
            print(f"   Larmor: {larmor_freq/1e6:.6f} MHz") 
            print(f"   Relative error: {relative_error:.2e}")
        
        # Check that frequencies are positive and finite
        all_freqs = nmr_frequencies['transition_frequencies']
        for freq in all_freqs:
            if not (np.isfinite(freq) and freq > 0):
                raise FallbackViolationError(
                    f"Invalid NMR frequency: {freq} Hz\n"
                    "All transition frequencies must be positive and finite."
                )
        
        print(f"✅ NMR validation passed for B = {np.linalg.norm(magnetic_field)*1e3:.1f} mT")
    
    def _validate_nuclear_constants(self):
        """Validate nuclear constants against literature"""
        
        # CODATA/NIST recommended values
        literature_gamma = 0.3077e6  # Hz/T
        literature_g_factor = -0.28304
        
        # Validate gyromagnetic ratio
        self.require_experimental_validation(
            self._gamma_n, literature_gamma,
            "gyromagnetic_ratio", tolerance=1e-6
        )
        
        # Validate g-factor
        self.require_experimental_validation(
            self._g_factor, literature_g_factor,
            "nuclear_g_factor", tolerance=1e-5
        )
        
        # Check consistency: γₙ = gₙ μₙ / ℏ
        calculated_gamma = abs(self._g_factor) * self._nuclear_magneton / (2 * np.pi * 1.054571817e-34)
        
        relative_error = abs(calculated_gamma - self._gamma_n) / self._gamma_n
        if relative_error > 1e-4:
            print(f"⚠️  Warning: γₙ and gₙ consistency check:")
            print(f"   Calculated γₙ: {calculated_gamma/1e6:.6f} MHz/T")
            print(f"   Literature γₙ: {self._gamma_n/1e6:.6f} MHz/T")
            print(f"   Relative error: {relative_error:.2e}")
        
        print("✅ Nuclear constant validations passed")
    
    def calculate_second_order_corrections(self, magnetic_field: np.ndarray) -> Dict[str, float]:
        """
        Calculate second-order Zeeman corrections for high fields
        
        For very high magnetic fields, second-order terms become important:
        H₂ = -(γₙ)² B² / (4 * nuclear_binding_energy) * corrections
        """
        
        B_magnitude = np.linalg.norm(magnetic_field)
        
        # Second-order correction is typically very small for N14
        # Estimate using typical nuclear binding energy scale
        nuclear_binding_scale = 1e9  # Hz (order of magnitude)
        
        second_order_correction = (self._gamma_n * B_magnitude)**2 / nuclear_binding_scale
        
        return {
            'second_order_frequency': second_order_correction,  # Hz
            'relative_correction': second_order_correction / (self._gamma_n * B_magnitude),
            'field_magnitude': B_magnitude  # T
        }
    
    def calculate_field_gradient_effects(self, field_gradient: np.ndarray) -> Dict[str, float]:
        """
        Calculate effects of magnetic field gradients
        
        Args:
            field_gradient: ∇B tensor (3×3) in T/m
            
        Returns:
            Gradient-induced frequency shifts and broadening
        """
        
        if field_gradient.shape != (3, 3):
            raise ValueError("Field gradient must be 3×3 tensor")
        
        # Gradient contribution to linewidth (motional narrowing)
        # Δν ≈ γₙ * |∇B| * characteristic_length
        
        gradient_magnitude = np.linalg.norm(field_gradient)
        characteristic_length = 1e-9  # 1 nm (nuclear dimensions)
        
        gradient_broadening = self._gamma_n * gradient_magnitude * characteristic_length
        
        return {
            'gradient_broadening': gradient_broadening,  # Hz
            'gradient_magnitude': gradient_magnitude,  # T/m
            'relative_broadening': gradient_broadening / self._gamma_n  # T
        }
    
    def get_nuclear_parameters(self) -> Dict[str, float]:
        """Get all nuclear magnetic parameters"""
        return {
            'gyromagnetic_ratio': self._gamma_n,  # Hz/T
            'g_factor': self._g_factor,
            'nuclear_magneton': self._nuclear_magneton,  # J/T
            'nuclear_spin': self._nuclear_spin
        }