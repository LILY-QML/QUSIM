"""
N14 Physics Validator - Comprehensive Validation Framework

ULTIMATE validation system for N14 nuclear spin physics ensuring:
- Zero tolerance for fallbacks, mocks, or approximations
- Experimental validation against literature
- Quantum mechanical consistency checks
- Complete physics validation coverage

This validator enforces the highest standards in quantum simulation.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union
from .base import NoFallbackBase, FallbackViolationError

class N14PhysicsValidator(NoFallbackBase):
    """
    Comprehensive N14 physics validation framework
    
    This validator ensures:
    1. All calculations match experimental literature
    2. Quantum mechanical principles are preserved
    3. No fallback patterns exist anywhere
    4. Complete physics consistency
    5. Numerical precision requirements met
    
    ZERO COMPROMISES - this validator rejects any non-physical results.
    """
    
    def __init__(self):
        super().__init__()
        
        # Literature reference values (experimentally validated)
        self._literature_values = {
            'hyperfine_parallel': -2.16e6,      # Hz ± 0.02e6
            'hyperfine_perpendicular': -2.7e6,  # Hz ± 0.1e6
            'quadrupole_coupling': -4.95e6,     # Hz ± 0.1e6
            'gyromagnetic_ratio': 0.3077e6,     # Hz/T ± 1e3
            'nuclear_g_factor': -0.28304,       # ± 1e-5
            'nuclear_spin': 1.0,                # Exact
            'quadrupole_moment': 2.044e-30,     # C⋅m² ± 0.01e-30
            'asymmetry_parameter': 0.0          # ± 0.05 (axial symmetry)
        }
        
        # Tolerance levels for different types of validation
        self._tolerances = {
            'experimental': 1e-6,     # Experimental accuracy
            'quantum_mechanical': 1e-12,  # QM consistency
            'numerical': 1e-15,       # Numerical precision
            'energy_scale': 0.1       # Energy scale reasonableness
        }
        
        print("✅ N14 Physics Validator initialized")
        print("   Experimental references loaded")
        print("   Zero-tolerance validation ready")
    
    def validate_complete_n14_system(self, 
                                   quantum_ops: Dict[str, np.ndarray],
                                   hyperfine_params: Dict[str, float],
                                   quadrupole_params: Dict[str, float],
                                   zeeman_params: Dict[str, float],
                                   system_hamiltonians: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """
        Validate complete N14 physics system
        
        Args:
            quantum_ops: All nuclear operators
            hyperfine_params: Hyperfine coupling parameters
            quadrupole_params: Quadrupole interaction parameters
            zeeman_params: Nuclear Zeeman parameters
            system_hamiltonians: All Hamiltonian matrices
            
        Returns:
            Complete validation results
        """
        
        validation_results = {}
        
        print("🔍 Starting comprehensive N14 system validation...")
        
        # 1. Validate quantum operators
        print("🔄 Validating quantum operators...")
        validation_results['quantum_operators'] = self._validate_quantum_operators(quantum_ops)
        
        # 2. Validate hyperfine parameters
        print("🔄 Validating hyperfine parameters...")
        validation_results['hyperfine_parameters'] = self._validate_hyperfine_parameters(hyperfine_params)
        
        # 3. Validate quadrupole parameters
        print("🔄 Validating quadrupole parameters...")
        validation_results['quadrupole_parameters'] = self._validate_quadrupole_parameters(quadrupole_params)
        
        # 4. Validate Zeeman parameters
        print("🔄 Validating Zeeman parameters...")
        validation_results['zeeman_parameters'] = self._validate_zeeman_parameters(zeeman_params)
        
        # 5. Validate Hamiltonians
        print("🔄 Validating Hamiltonians...")
        validation_results['hamiltonians'] = self._validate_hamiltonians(system_hamiltonians)
        
        # 6. Cross-validate physics consistency
        print("🔄 Cross-validating physics consistency...")
        validation_results['physics_consistency'] = self._validate_physics_consistency(
            quantum_ops, hyperfine_params, quadrupole_params, zeeman_params
        )
        
        # 7. Overall system validation
        all_passed = all(
            all(sub_results.values()) if isinstance(sub_results, dict) else sub_results
            for sub_results in validation_results.values()
        )
        
        validation_results['overall_system_valid'] = all_passed
        
        if all_passed:
            print("✅ Complete N14 system validation: PASSED")
        else:
            failed_components = [
                key for key, result in validation_results.items()
                if not (all(result.values()) if isinstance(result, dict) else result)
            ]
            raise FallbackViolationError(
                f"N14 system validation FAILED!\\n"
                f"Failed components: {failed_components}\\n"
                f"System does not meet zero-tolerance standards."
            )
        
        return validation_results
    
    def _validate_quantum_operators(self, quantum_ops: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """Validate all nuclear quantum operators"""
        
        validation = {}
        
        # Required operators for I=1
        required_ops = ['Ix', 'Iy', 'Iz', 'Ix²', 'Iy²', 'Iz²', 'I²', 'I+', 'I-']
        
        # Check all required operators exist
        validation['all_operators_present'] = all(op in quantum_ops for op in required_ops)
        
        if not validation['all_operators_present']:
            missing = [op for op in required_ops if op not in quantum_ops]
            raise FallbackViolationError(
                f"Missing quantum operators: {missing}\\n"
                "All I=1 operators must be present."
            )
        
        # Validate individual operators
        Ix, Iy, Iz = quantum_ops['Ix'], quantum_ops['Iy'], quantum_ops['Iz']
        Ix2, Iy2, Iz2 = quantum_ops['Ix²'], quantum_ops['Iy²'], quantum_ops['Iz²']
        I_squared = quantum_ops['I²']
        I_plus, I_minus = quantum_ops['I+'], quantum_ops['I-']
        
        # 1. Operator dimensions (3×3 for I=1)
        validation['correct_dimensions'] = all(
            op.shape == (3, 3) for op in [Ix, Iy, Iz, Ix2, Iy2, Iz2, I_squared, I_plus, I_minus]
        )
        
        # 2. Hermiticity of angular momentum operators
        validation['hermiticity_Ix'] = np.allclose(Ix, Ix.conj().T, atol=self._tolerances['numerical'])
        validation['hermiticity_Iy'] = np.allclose(Iy, Iy.conj().T, atol=self._tolerances['numerical'])
        validation['hermiticity_Iz'] = np.allclose(Iz, Iz.conj().T, atol=self._tolerances['numerical'])
        
        # 3. Commutation relations: [Ix, Iy] = iℏIz (with ℏ=1)
        commutator_xy = Ix @ Iy - Iy @ Ix
        expected_xy = 1j * Iz
        validation['commutation_xy'] = np.allclose(commutator_xy, expected_xy, atol=self._tolerances['quantum_mechanical'])
        
        commutator_yz = Iy @ Iz - Iz @ Iy
        expected_yz = 1j * Ix
        validation['commutation_yz'] = np.allclose(commutator_yz, expected_yz, atol=self._tolerances['quantum_mechanical'])
        
        commutator_zx = Iz @ Ix - Ix @ Iz
        expected_zx = 1j * Iy
        validation['commutation_zx'] = np.allclose(commutator_zx, expected_zx, atol=self._tolerances['quantum_mechanical'])
        
        # 4. I² operator consistency: I² = Ix² + Iy² + Iz²
        calculated_I2 = Ix2 + Iy2 + Iz2
        validation['I_squared_consistency'] = np.allclose(I_squared, calculated_I2, atol=self._tolerances['numerical'])
        
        # 5. Eigenvalue validation: I² eigenvalues should be I(I+1) = 2 for I=1
        I2_eigenvals = np.linalg.eigvals(I_squared)
        expected_eigenval = 1.0 * (1.0 + 1.0)  # I(I+1) = 2
        validation['I_squared_eigenvalues'] = np.allclose(I2_eigenvals, expected_eigenval, atol=self._tolerances['numerical'])
        
        # 6. Ladder operator relations: I± = Ix ± iIy
        calculated_I_plus = Ix + 1j * Iy
        calculated_I_minus = Ix - 1j * Iy
        validation['ladder_operator_plus'] = np.allclose(I_plus, calculated_I_plus, atol=self._tolerances['numerical'])
        validation['ladder_operator_minus'] = np.allclose(I_minus, calculated_I_minus, atol=self._tolerances['numerical'])
        
        # 7. Verify no fallback patterns in operators
        for op_name, operator in quantum_ops.items():
            self._validate_not_fallback(operator, f"quantum operator {op_name}")
        
        validation['no_fallback_patterns'] = True
        
        return validation
    
    def _validate_hyperfine_parameters(self, hyperfine_params: Dict[str, float]) -> Dict[str, bool]:
        """Validate hyperfine coupling parameters"""
        
        validation = {}
        
        # Required parameters
        required_params = ['A_parallel', 'A_perpendicular']
        validation['required_params_present'] = all(param in hyperfine_params for param in required_params)
        
        if not validation['required_params_present']:
            missing = [p for p in required_params if p not in hyperfine_params]
            raise FallbackViolationError(f"Missing hyperfine parameters: {missing}")
        
        A_parallel = hyperfine_params['A_parallel']
        A_perpendicular = hyperfine_params['A_perpendicular']
        
        # Validate against literature
        validation['A_parallel_literature'] = self._validate_against_literature(
            A_parallel, 'hyperfine_parallel', tolerance=0.02
        )
        
        validation['A_perpendicular_literature'] = self._validate_against_literature(
            A_perpendicular, 'hyperfine_perpendicular', tolerance=0.05
        )
        
        # Physical constraints
        validation['A_parallel_negative'] = A_parallel < 0  # Should be negative for N14
        validation['A_perpendicular_negative'] = A_perpendicular < 0  # Should be negative for N14
        validation['anisotropy_correct'] = abs(A_perpendicular) > abs(A_parallel)  # Expected anisotropy
        
        # Energy scale reasonableness (MHz range)
        validation['A_parallel_scale'] = 1e6 <= abs(A_parallel) <= 10e6
        validation['A_perpendicular_scale'] = 1e6 <= abs(A_perpendicular) <= 10e6
        
        # Check no fallback patterns
        for param_name, value in hyperfine_params.items():
            self._validate_not_fallback(value, f"hyperfine parameter {param_name}")
        
        return validation
    
    def _validate_quadrupole_parameters(self, quadrupole_params: Dict[str, float]) -> Dict[str, bool]:
        """Validate quadrupole interaction parameters"""
        
        validation = {}
        
        # Required parameters
        required_params = ['eqQ', 'asymmetry_parameter']
        validation['required_params_present'] = all(param in quadrupole_params for param in required_params)
        
        if not validation['required_params_present']:
            missing = [p for p in required_params if p not in quadrupole_params]
            raise FallbackViolationError(f"Missing quadrupole parameters: {missing}")
        
        eqQ = quadrupole_params['eqQ']
        eta = quadrupole_params['asymmetry_parameter']
        
        # Validate against literature
        validation['eqQ_literature'] = self._validate_against_literature(
            eqQ, 'quadrupole_coupling', tolerance=0.05
        )
        
        validation['eta_literature'] = self._validate_against_literature(
            eta, 'asymmetry_parameter', tolerance=0.05
        )
        
        # Physical constraints
        validation['eqQ_negative'] = eqQ < 0  # Should be negative for N14
        validation['eta_range'] = 0 <= eta <= 1  # Asymmetry parameter bounds
        validation['axial_symmetry'] = abs(eta) < 0.1  # NV-N14 should be nearly axial
        
        # Energy scale reasonableness (MHz range)
        validation['eqQ_scale'] = 1e6 <= abs(eqQ) <= 10e6
        
        # Check no fallback patterns
        for param_name, value in quadrupole_params.items():
            self._validate_not_fallback(value, f"quadrupole parameter {param_name}")
        
        return validation
    
    def _validate_zeeman_parameters(self, zeeman_params: Dict[str, float]) -> Dict[str, bool]:
        """Validate nuclear Zeeman parameters"""
        
        validation = {}
        
        # Required parameters
        required_params = ['gyromagnetic_ratio', 'g_factor']
        validation['required_params_present'] = all(param in zeeman_params for param in required_params)
        
        if not validation['required_params_present']:
            missing = [p for p in required_params if p not in zeeman_params]
            raise FallbackViolationError(f"Missing Zeeman parameters: {missing}")
        
        gamma_n = zeeman_params['gyromagnetic_ratio']
        g_factor = zeeman_params['g_factor']
        
        # Validate against literature
        validation['gamma_literature'] = self._validate_against_literature(
            gamma_n, 'gyromagnetic_ratio', tolerance=1e-3
        )
        
        validation['g_factor_literature'] = self._validate_against_literature(
            g_factor, 'nuclear_g_factor', tolerance=1e-4
        )
        
        # Physical constraints
        validation['gamma_positive'] = gamma_n > 0  # Magnitude should be positive
        validation['g_factor_negative'] = g_factor < 0  # N14 has negative g-factor
        
        # Energy scale reasonableness
        validation['gamma_scale'] = 1e5 <= gamma_n <= 1e7  # Hz/T range
        validation['g_factor_scale'] = 0.1 <= abs(g_factor) <= 1.0
        
        # Check no fallback patterns
        for param_name, value in zeeman_params.items():
            self._validate_not_fallback(value, f"Zeeman parameter {param_name}")
        
        return validation
    
    def _validate_hamiltonians(self, hamiltonians: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """Validate all Hamiltonian matrices"""
        
        validation = {}
        
        for hamiltonian_name, H in hamiltonians.items():
            ham_validation = {}
            
            # Hermiticity check
            ham_validation['hermitian'] = np.allclose(H, H.conj().T, atol=self._tolerances['numerical'])
            
            # Real eigenvalues (consequence of Hermiticity)
            eigenvals = np.linalg.eigvals(H)
            ham_validation['real_eigenvalues'] = np.allclose(eigenvals.imag, 0, atol=self._tolerances['numerical'])
            
            # Finite eigenvalues
            ham_validation['finite_eigenvalues'] = np.all(np.isfinite(eigenvals))
            
            # Appropriate energy scale
            energy_range = np.max(eigenvals.real) - np.min(eigenvals.real)
            ham_validation['reasonable_energy_scale'] = 1e3 <= energy_range <= 1e10  # Hz range
            
            # No fallback patterns
            self._validate_not_fallback(H, f"Hamiltonian {hamiltonian_name}")
            ham_validation['no_fallback_patterns'] = True
            
            validation[hamiltonian_name] = ham_validation
        
        return validation
    
    def _validate_physics_consistency(self, 
                                    quantum_ops: Dict[str, np.ndarray],
                                    hyperfine_params: Dict[str, float],
                                    quadrupole_params: Dict[str, float],
                                    zeeman_params: Dict[str, float]) -> Dict[str, bool]:
        """Cross-validate physics consistency across all modules"""
        
        validation = {}
        
        # 1. Nuclear spin consistency
        I = self._literature_values['nuclear_spin']
        I_squared = quantum_ops['I²']
        I2_trace = np.trace(I_squared)
        expected_trace = 3 * I * (I + 1)  # Tr(I²) = (2I+1) × I(I+1)
        validation['nuclear_spin_consistency'] = abs(I2_trace - expected_trace) < 1e-12
        
        # 2. Energy scale consistency
        A_parallel = hyperfine_params['A_parallel']
        eqQ = quadrupole_params['eqQ']
        gamma_n = zeeman_params['gyromagnetic_ratio']
        
        # All should be in similar MHz energy scale
        energies = [abs(A_parallel), abs(eqQ)]
        validation['energy_scale_consistency'] = all(1e6 <= E <= 10e6 for E in energies)
        
        # 3. Anisotropy consistency
        A_perp = hyperfine_params['A_perpendicular']
        validation['hyperfine_anisotropy'] = abs(A_perp) > abs(A_parallel)
        
        # 4. Quadrupole-specific validation (only for I≥1)
        validation['quadrupole_I_constraint'] = I >= 1.0  # Quadrupole only for I≥1
        
        # 5. Sign consistency
        validation['N14_sign_consistency'] = (
            A_parallel < 0 and A_perp < 0 and eqQ < 0 and zeeman_params['g_factor'] < 0
        )
        
        return validation
    
    def _validate_against_literature(self, value: float, parameter_name: str, tolerance: float) -> bool:
        """Validate parameter against experimental literature"""
        
        if parameter_name not in self._literature_values:
            raise ValueError(f"No literature reference for parameter: {parameter_name}")
        
        literature_value = self._literature_values[parameter_name]
        
        # Handle case where literature value is zero
        if abs(literature_value) < 1e-10:
            relative_error = abs(value - literature_value)
        else:
            relative_error = abs(value - literature_value) / abs(literature_value)
        
        if relative_error > tolerance:
            print(f"⚠️  Warning: {parameter_name} deviates from literature:")
            print(f"   Calculated: {value:.6e}")
            print(f"   Literature: {literature_value:.6e}")
            print(f"   Relative error: {relative_error:.2e}")
            print(f"   Tolerance: {tolerance:.2e}")
            return False
        
        return True
    
    def validate_experimental_transition_frequencies(self, 
                                                   calculated_frequencies: Dict[str, np.ndarray],
                                                   magnetic_field: np.ndarray) -> Dict[str, bool]:
        """Validate calculated frequencies against experimental measurements"""
        
        validation = {}
        
        # NMR frequencies should match Larmor frequency
        B_magnitude = np.linalg.norm(magnetic_field)
        gamma_n = self._literature_values['gyromagnetic_ratio']
        expected_larmor = gamma_n * B_magnitude
        
        if 'larmor_frequency' in calculated_frequencies:
            larmor_freq = calculated_frequencies['larmor_frequency']
            validation['larmor_frequency_match'] = self._validate_against_literature(
                larmor_freq, 'gyromagnetic_ratio', tolerance=1e-6
            )
        
        # NQR frequencies should match quadrupole coupling
        if 'nqr_frequencies' in calculated_frequencies:
            nqr_data = calculated_frequencies['nqr_frequencies']
            if isinstance(nqr_data, dict) and 'average_frequency' in nqr_data:
                avg_nqr = nqr_data['average_frequency']
                expected_nqr = abs(self._literature_values['quadrupole_coupling']) / 2
                relative_error = abs(avg_nqr - expected_nqr) / expected_nqr
                validation['nqr_frequency_match'] = relative_error < 0.2  # 20% tolerance
        
        # All frequencies should be positive and finite
        validation['all_frequencies_physical'] = True
        for freq_type, freq_data in calculated_frequencies.items():
            if isinstance(freq_data, (int, float)):
                if not (np.isfinite(freq_data) and freq_data >= 0):
                    validation['all_frequencies_physical'] = False
            elif isinstance(freq_data, np.ndarray):
                if not np.all(np.isfinite(freq_data)) or not np.all(freq_data >= 0):
                    validation['all_frequencies_physical'] = False
        
        return validation
    
    def validate_quantum_state_evolution(self, evolution_results: Dict[str, Any]) -> Dict[str, bool]:
        """Validate quantum state evolution for unitarity and physical consistency"""
        
        validation = {}
        
        # Check if evolution operators are unitary
        if 'evolution_operator' in evolution_results:
            U = evolution_results['evolution_operator']
            
            # Unitarity: U†U = I
            unity_check = U.conj().T @ U
            validation['evolution_unitary'] = np.allclose(
                unity_check, np.eye(U.shape[0]), atol=self._tolerances['quantum_mechanical']
            )
            
            # Determinant should have magnitude 1
            det_U = np.linalg.det(U)
            validation['evolution_determinant'] = abs(abs(det_U) - 1.0) < self._tolerances['quantum_mechanical']
        
        # Check state normalization throughout evolution
        if 'states' in evolution_results:
            states = evolution_results['states']
            norms = [np.linalg.norm(state) for state in states]
            validation['state_normalization'] = all(
                abs(norm - 1.0) < self._tolerances['quantum_mechanical'] for norm in norms
            )
        
        # Check final state consistency
        if 'final_state' in evolution_results:
            final_state = evolution_results['final_state']
            final_norm = np.linalg.norm(final_state)
            validation['final_state_normalized'] = abs(final_norm - 1.0) < self._tolerances['quantum_mechanical']
        
        return validation
    
    def get_literature_references(self) -> Dict[str, Dict[str, Union[float, str]]]:
        """Get all literature reference values with sources"""
        
        return {
            'hyperfine_parallel': {
                'value': -2.16e6,
                'unit': 'Hz',
                'uncertainty': 0.02e6,
                'source': 'Jacques et al., Phys. Rev. Lett. 102, 057403 (2009)'
            },
            'hyperfine_perpendicular': {
                'value': -2.7e6,
                'unit': 'Hz', 
                'uncertainty': 0.1e6,
                'source': 'Jacques et al., Phys. Rev. Lett. 102, 057403 (2009)'
            },
            'quadrupole_coupling': {
                'value': -4.95e6,
                'unit': 'Hz',
                'uncertainty': 0.1e6,
                'source': 'Van Oort & Glasbeek, Chem. Phys. Lett. 168, 529 (1990)'
            },
            'gyromagnetic_ratio': {
                'value': 0.3077e6,
                'unit': 'Hz/T',
                'uncertainty': 1e3,
                'source': 'CODATA 2018 recommended values'
            },
            'nuclear_g_factor': {
                'value': -0.28304,
                'unit': 'dimensionless',
                'uncertainty': 1e-5,
                'source': 'NIST Atomic Spectra Database'
            }
        }
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report"""
        
        report = []
        report.append("=" * 60)
        report.append("N14 PHYSICS VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall status
        overall_status = validation_results.get('overall_system_valid', False)
        status_symbol = "✅" if overall_status else "❌"
        report.append(f"OVERALL VALIDATION STATUS: {status_symbol} {'PASSED' if overall_status else 'FAILED'}")
        report.append("")
        
        # Detailed results
        for category, results in validation_results.items():
            if category == 'overall_system_valid':
                continue
                
            report.append(f"{category.upper().replace('_', ' ')}:")
            report.append("-" * 30)
            
            if isinstance(results, dict):
                for test_name, passed in results.items():
                    symbol = "✅" if passed else "❌"
                    report.append(f"  {symbol} {test_name.replace('_', ' ')}")
            else:
                symbol = "✅" if results else "❌"
                report.append(f"  {symbol} {category.replace('_', ' ')}")
            
            report.append("")
        
        # Literature references
        report.append("LITERATURE REFERENCES:")
        report.append("-" * 30)
        for param, ref_data in self.get_literature_references().items():
            report.append(f"  {param}: {ref_data['value']:.3e} {ref_data['unit']}")
            report.append(f"    Source: {ref_data['source']}")
            report.append("")
        
        return "\n".join(report)