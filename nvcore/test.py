#!/usr/bin/env python3
"""
Comprehensive Test Suite for QUSIM NV Center Simulation

This module contains extensive tests for all QUSIM components including:
- Core NV system functionality
- Noise modeling and integration
- Pulse sequence operations
- Command-line interface
- Performance benchmarks
- Physics validation

The tests are organized into categories:
1. Unit tests for individual components
2. Integration tests for system behavior
3. Physics validation tests
4. Performance and regression tests

Examples:
    Run all tests:
        $ python test.py
        
    Run specific test category:
        $ python test.py TestNVSystem
        
    Run with coverage:
        $ coverage run test.py
        $ coverage report

Author: QUSIM Development Team
License: MIT
"""

import unittest
import numpy as np
import tempfile
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
import warnings

# Add module paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'helper'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import modules to test
from core import QUSIMCore
from nvcore import NVSystem, NVSpinOperators, NVSystemHamiltonian
from nvcore_fast import FastNVSystem
from noise import NoiseGenerator, NoiseConfiguration
from noise_sources import SYSTEM


class TestNVSpinOperators(unittest.TestCase):
    """
    Test suite for NV center spin operators.
    
    Validates the mathematical properties of spin-1 operators including
    commutation relations, matrix representations, and algebraic properties.
    """
    
    def setUp(self):
        """Initialize spin operators for testing."""
        self.ops = NVSpinOperators()
        
    def test_spin_operator_dimensions(self):
        """Test that all spin operators have correct dimensions."""
        self.assertEqual(self.ops.Sx.shape, (3, 3))
        self.assertEqual(self.ops.Sy.shape, (3, 3))
        self.assertEqual(self.ops.Sz.shape, (3, 3))
        
    def test_spin_operator_hermiticity(self):
        """Test that spin operators are Hermitian."""
        np.testing.assert_allclose(self.ops.Sx, self.ops.Sx.conj().T, atol=1e-10)
        np.testing.assert_allclose(self.ops.Sy, self.ops.Sy.conj().T, atol=1e-10)
        np.testing.assert_allclose(self.ops.Sz, self.ops.Sz.conj().T, atol=1e-10)
        
    def test_commutation_relations(self):
        """
        Test fundamental commutation relations: [Si, Sj] = i*ε_ijk*Sk.
        
        For spin-1 system, we verify the SU(2) algebra.
        """
        # [Sx, Sy] = i*Sz
        commutator_xy = self.ops.Sx @ self.ops.Sy - self.ops.Sy @ self.ops.Sx
        np.testing.assert_allclose(commutator_xy, 1j * self.ops.Sz, atol=1e-10)
        
        # [Sy, Sz] = i*Sx
        commutator_yz = self.ops.Sy @ self.ops.Sz - self.ops.Sz @ self.ops.Sy
        np.testing.assert_allclose(commutator_yz, 1j * self.ops.Sx, atol=1e-10)
        
        # [Sz, Sx] = i*Sy
        commutator_zx = self.ops.Sz @ self.ops.Sx - self.ops.Sx @ self.ops.Sz
        np.testing.assert_allclose(commutator_zx, 1j * self.ops.Sy, atol=1e-10)
        
    def test_eigenvalues(self):
        """Test that Sz has correct eigenvalues: -1, 0, +1."""
        eigenvalues = np.linalg.eigvalsh(self.ops.Sz)
        expected = np.array([-1, 0, 1])
        np.testing.assert_allclose(sorted(eigenvalues), expected, atol=1e-10)
        
    def test_total_spin(self):
        """Test S² = S·S = 2 for spin-1 system."""
        S_squared = (self.ops.Sx @ self.ops.Sx + 
                    self.ops.Sy @ self.ops.Sy + 
                    self.ops.Sz @ self.ops.Sz)
        expected = 2 * self.ops.I  # S(S+1) = 1(1+1) = 2
        np.testing.assert_allclose(S_squared, expected, atol=1e-10)
        
    def test_raising_lowering_operators(self):
        """Test properties of raising and lowering operators."""
        # S+ = Sx + i*Sy
        S_plus_expected = self.ops.Sx + 1j * self.ops.Sy
        np.testing.assert_allclose(self.ops.S_plus, S_plus_expected, atol=1e-10)
        
        # S- = Sx - i*Sy  
        S_minus_expected = self.ops.Sx - 1j * self.ops.Sy
        np.testing.assert_allclose(self.ops.S_minus, S_minus_expected, atol=1e-10)
        
        # S+ and S- should be Hermitian conjugates
        np.testing.assert_allclose(self.ops.S_plus, self.ops.S_minus.conj().T, atol=1e-10)


class TestNVSystemHamiltonian(unittest.TestCase):
    """
    Test suite for NV center Hamiltonian construction.
    
    Validates the physical correctness of the Hamiltonian including
    zero-field splitting, Zeeman effect, and noise integration.
    """
    
    def setUp(self):
        """Initialize test configuration."""
        self.B_field = np.array([0.001, 0.002, 0.010])  # Small test field in Tesla
        self.hamiltonian = NVSystemHamiltonian(B_field=self.B_field)
        
    def test_zero_field_splitting(self):
        """Test that ZFS term is correctly implemented."""
        # Create Hamiltonian with no magnetic field
        H_zfs = NVSystemHamiltonian(B_field=np.zeros(3))
        H_static = H_zfs.get_static_hamiltonian()
        
        # Expected ZFS Hamiltonian: D(Sz² - 2/3*I)
        ops = NVSpinOperators()
        D = SYSTEM.get_constant('nv_center', 'd_gs')
        H_expected = 2 * np.pi * D * (ops.Sz2 - (2/3) * ops.I)
        
        np.testing.assert_allclose(H_static, H_expected, rtol=1e-10)
        
    def test_zeeman_effect(self):
        """Test magnetic field interaction (Zeeman effect)."""
        H_static = self.hamiltonian.get_static_hamiltonian()
        
        # Extract Zeeman contribution
        H_no_field = NVSystemHamiltonian(B_field=np.zeros(3)).get_static_hamiltonian()
        H_zeeman = H_static - H_no_field
        
        # Expected Zeeman term: γ*B·S
        ops = NVSpinOperators()
        gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
        H_zeeman_expected = 2 * np.pi * gamma_e * (
            self.B_field[0] * ops.Sx +
            self.B_field[1] * ops.Sy +
            self.B_field[2] * ops.Sz
        )
        
        np.testing.assert_allclose(H_zeeman, H_zeeman_expected, rtol=1e-10)
        
    def test_hamiltonian_hermiticity(self):
        """Test that Hamiltonian is Hermitian."""
        H = self.hamiltonian.get_static_hamiltonian()
        np.testing.assert_allclose(H, H.conj().T, atol=1e-14)
        
    def test_energy_eigenvalues(self):
        """Test that energy eigenvalues are physically reasonable."""
        H = self.hamiltonian.get_static_hamiltonian()
        eigenvalues = np.linalg.eigvalsh(H)
        
        # Energies should be real
        self.assertTrue(np.all(np.isreal(eigenvalues)))
        
        # Check energy scale (should be on order of D ~ 2.87 GHz)
        D_hz = SYSTEM.get_constant('nv_center', 'd_gs')
        max_energy = np.max(np.abs(eigenvalues)) / (2 * np.pi)
        self.assertLess(max_energy, 10 * D_hz)  # Reasonable upper bound


class TestNVSystem(unittest.TestCase):
    """
    Test suite for complete NV system functionality.
    
    Tests the integration of all components including evolution,
    pulse sequences, and noise effects.
    """
    
    def setUp(self):
        """Initialize NV system for testing."""
        self.B_field = np.array([0, 0, 0.01])  # 10 mT along z
        self.system = NVSystem(B_field=self.B_field)
        
    def test_unitary_evolution(self):
        """Test unitary evolution preserves trace and hermiticity."""
        # Initial state: |0⟩⟨0|
        rho0 = np.zeros((3, 3), dtype=complex)
        rho0[1, 1] = 1.0
        
        # Evolve for short time
        t_span = (0, 1e-9)
        times, rho_history = self.system.evolve(rho0, t_span)
        
        for rho in rho_history:
            # Check trace preservation
            self.assertAlmostEqual(np.trace(rho), 1.0, places=10)
            
            # Check hermiticity
            np.testing.assert_allclose(rho, rho.conj().T, atol=1e-10)
            
            # Check positive semi-definiteness
            eigenvalues = np.linalg.eigvalsh(rho)
            self.assertTrue(np.all(eigenvalues >= -1e-10))
            
    def test_rabi_oscillations(self):
        """Test Rabi oscillations under resonant driving."""
        # Initial state: |0⟩
        rho0 = np.zeros((3, 3), dtype=complex)
        rho0[1, 1] = 1.0
        
        # Rabi frequency
        rabi_freq = 2 * np.pi * 10e6  # 10 MHz
        
        # Create pulse for one Rabi period
        pulse = {
            'duration': 1 / (2 * 10e6),  # π pulse
            'rabi_frequency': rabi_freq,
            'phase': 0,
            'detuning': 0
        }
        
        # Evolve with pulse
        times, rho_history = self.system.evolve_with_pulses(rho0, [pulse])
        
        # Final state should have population transfer
        rho_final = rho_history[-1]
        
        # For π pulse, expect population transfer from |0⟩ to |+1⟩
        self.assertLess(rho_final[1, 1].real, 0.1)  # |0⟩ depleted
        self.assertGreater(rho_final[2, 2].real, 0.8)  # |+1⟩ populated
        
    def test_coherence_decay_with_noise(self):
        """Test that coherences decay with noise enabled."""
        # Create system with noise
        noise_config = NoiseConfiguration()
        noise_config.enable_c13_bath = True
        noise_gen = NoiseGenerator(noise_config)
        noisy_system = NVSystem(B_field=self.B_field, noise_gen=noise_gen)
        
        # Initial superposition state
        psi = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        rho0 = np.outer(psi, psi.conj())
        
        # Evolve for longer time
        t_span = (0, 1e-6)
        times, rho_history = noisy_system.evolve(rho0, t_span, dt=1e-9)
        
        # Check coherence decay
        coherence_01 = np.array([np.abs(rho[0, 1]) for rho in rho_history])
        coherence_12 = np.array([np.abs(rho[1, 2]) for rho in rho_history])
        
        # Coherences should decay
        self.assertLess(coherence_12[-1], coherence_12[0] * 0.9)


class TestFastNVSystem(unittest.TestCase):
    """
    Test suite for optimized fast NV system.
    
    Ensures that the fast implementation maintains physical accuracy
    while providing performance improvements.
    """
    
    def setUp(self):
        """Initialize fast NV system."""
        self.B_field = np.array([0, 0, 0.01])
        self.fast_system = FastNVSystem(B_field=self.B_field, enable_noise=False)
        
    def test_fast_vs_full_consistency(self):
        """Test that fast and full systems give consistent results."""
        # Create both systems
        full_system = NVSystem(B_field=self.B_field, noise_gen=None)
        
        # Initial state
        rho0 = np.zeros((3, 3), dtype=complex)
        rho0[1, 1] = 1.0
        
        # Evolve both
        t_span = (0, 1e-8)
        times_fast, rho_fast = self.fast_system.evolve_unitary(rho0, t_span, n_steps=100)
        times_full, rho_full = full_system.evolve(rho0, t_span)
        
        # Compare at similar time points
        for i in range(min(len(times_fast), len(times_full))):
            if i < len(rho_fast) and i < len(rho_full):
                np.testing.assert_allclose(rho_fast[i], rho_full[i], rtol=1e-3)
                
    def test_noise_caching(self):
        """Test that noise caching in fast system works correctly."""
        fast_noisy = FastNVSystem(B_field=self.B_field, enable_noise=True)
        
        # Get multiple noise samples
        noise_samples = []
        for _ in range(10):
            H = fast_noisy.get_hamiltonian()
            noise_samples.append(H.copy())
            
        # Check that samples are different (noise is working)
        all_same = all(np.allclose(noise_samples[0], sample) for sample in noise_samples[1:])
        self.assertFalse(all_same, "Noise samples should be different")


class TestNoiseGeneration(unittest.TestCase):
    """
    Test suite for noise generation and modeling.
    
    Validates statistical properties and physical correctness of
    all noise sources.
    """
    
    def setUp(self):
        """Initialize noise configuration."""
        self.config = NoiseConfiguration()
        self.config.n_samples = 1000
        self.noise_gen = NoiseGenerator(self.config)
        
    def test_magnetic_noise_statistics(self):
        """Test statistical properties of magnetic noise."""
        # Generate many samples
        noise_samples = []
        for _ in range(100):
            noise = self.noise_gen.get_total_magnetic_noise(1)[0]
            noise_samples.append(noise)
            
        noise_array = np.array(noise_samples)
        
        # Check mean is near zero
        mean = np.mean(noise_array, axis=0)
        np.testing.assert_allclose(mean, np.zeros(3), atol=1e-10)
        
        # Check variance is reasonable (depends on enabled sources)
        std = np.std(noise_array, axis=0)
        self.assertTrue(np.all(std > 0))  # Should have some noise
        self.assertTrue(np.all(std < 1e-3))  # But not too much (in Tesla)
        
    def test_noise_source_independence(self):
        """Test that different noise sources can be independently controlled."""
        # Test with only C13 bath
        config1 = NoiseConfiguration()
        config1.enable_c13_bath = True
        config1.enable_charge_noise = False
        config1.enable_temperature = False
        config1.enable_johnson = False
        gen1 = NoiseGenerator(config1)
        
        # Test with only charge noise
        config2 = NoiseConfiguration()
        config2.enable_c13_bath = False
        config2.enable_charge_noise = True
        config2.enable_temperature = False
        config2.enable_johnson = False
        gen2 = NoiseGenerator(config2)
        
        # Generate samples
        noise1 = gen1.get_total_magnetic_noise(100)
        noise2 = gen2.get_total_magnetic_noise(100)
        
        # They should be different
        self.assertFalse(np.allclose(noise1, noise2))


class TestQUSIMCore(unittest.TestCase):
    """
    Test suite for QUSIM command-line interface and core functionality.
    
    Tests argument parsing, system initialization, and result processing.
    """
    
    def setUp(self):
        """Initialize QUSIM core."""
        self.core = QUSIMCore()
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_argument_parsing(self):
        """Test command-line argument parsing."""
        # Test fast mode
        args = self.core.parse_arguments(['--fast', '--time', '1e-6'])
        self.assertTrue(args.fast)
        self.assertEqual(args.time, 1e-6)
        
        # Test noise configuration
        args = self.core.parse_arguments(['--noise', 'c13_bath,charge_noise'])
        self.assertEqual(args.noise, 'c13_bath,charge_noise')
        
        # Test magnetic field
        args = self.core.parse_arguments(['-B', '0.001', '0.002', '0.003'])
        np.testing.assert_allclose(args.magnetic_field, [0.001, 0.002, 0.003])
        
    def test_noise_configuration(self):
        """Test noise source configuration from arguments."""
        # Test disabling all noise
        self.core.args = self.core.parse_arguments(['--no-noise'])
        config = self.core._configure_noise()
        self.assertFalse(config.enable_c13_bath)
        self.assertFalse(config.enable_charge_noise)
        
        # Test specific noise sources
        self.core.args = self.core.parse_arguments(['--noise', 'c13_bath,temperature'])
        config = self.core._configure_noise()
        self.assertTrue(config.enable_c13_bath)
        self.assertTrue(config.enable_temperature)
        self.assertFalse(config.enable_charge_noise)
        
    def test_initial_state_creation(self):
        """Test creation of different initial quantum states."""
        # Ground state
        self.core.args = self.core.parse_arguments(['--initial-state', 'ground'])
        rho = self.core._create_initial_state()
        self.assertAlmostEqual(rho[1, 1], 1.0)
        
        # Superposition
        self.core.args = self.core.parse_arguments(['--initial-state', 'superposition'])
        rho = self.core._create_initial_state()
        self.assertAlmostEqual(rho[1, 1], 0.5)
        self.assertAlmostEqual(rho[2, 2], 0.5)
        
    def test_pulse_sequence_generation(self):
        """Test generation of predefined pulse sequences."""
        # Rabi sequence
        self.core.args = self.core.parse_arguments(
            ['--pulse-sequence', 'rabi', '--time', '1e-6']
        )
        pulses = self.core._create_pulse_sequence()
        self.assertEqual(len(pulses), 1)
        self.assertEqual(pulses[0]['duration'], 1e-6)
        
        # Ramsey sequence
        self.core.args = self.core.parse_arguments(
            ['--pulse-sequence', 'ramsey', '--time', '1e-6']
        )
        pulses = self.core._create_pulse_sequence()
        self.assertEqual(len(pulses), 3)  # π/2 - wait - π/2
        
    def test_result_saving(self):
        """Test that results are properly saved."""
        self.core.args = self.core.parse_arguments([
            '--fast', 
            '--time', '1e-8',
            '--output', self.temp_dir
        ])
        
        # Run simulation
        results = self.core.run()
        
        # Check that file was created
        files = os.listdir(self.temp_dir)
        npz_files = [f for f in files if f.endswith('.npz')]
        self.assertGreater(len(npz_files), 0)
        
        # Load and verify saved data
        saved_file = os.path.join(self.temp_dir, npz_files[0])
        data = np.load(saved_file, allow_pickle=True)
        self.assertIn('times', data)
        self.assertIn('populations', data)
        self.assertIn('metadata', data)


class TestPhysicsValidation(unittest.TestCase):
    """
    Physics validation tests.
    
    These tests ensure that the simulation produces physically correct results
    for known scenarios with analytical solutions.
    """
    
    def test_free_precession(self):
        """Test free precession in magnetic field."""
        # System with B field along x
        B_field = np.array([0.001, 0, 0])  # 1 mT
        system = NVSystem(B_field=B_field)
        
        # Initial state: |0⟩
        rho0 = np.zeros((3, 3), dtype=complex)
        rho0[1, 1] = 1.0
        
        # Expected precession frequency
        gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
        omega = 2 * np.pi * gamma_e * B_field[0]
        
        # Evolve for one period
        T = 2 * np.pi / omega
        times, rho_history = system.evolve(rho0, (0, T), dt=T/100)
        
        # Should return close to initial state after one period
        rho_final = rho_history[-1]
        np.testing.assert_allclose(rho_final[1, 1], rho0[1, 1], atol=0.1)
        
    def test_thermal_equilibrium(self):
        """Test approach to thermal equilibrium with dissipation."""
        # This would require implementing thermal dissipation
        # For now, we skip this advanced test
        pass
        
    def test_quantum_zeno_effect(self):
        """Test quantum Zeno effect with frequent measurements."""
        # This would require implementing measurement operators
        # For now, we skip this advanced test
        pass


class TestPerformance(unittest.TestCase):
    """
    Performance benchmarking tests.
    
    Ensures that optimizations maintain acceptable performance levels.
    """
    
    def test_fast_mode_speedup(self):
        """Test that fast mode provides significant speedup."""
        import time
        
        # Test parameters
        B_field = np.array([0, 0, 0.01])
        rho0 = np.diag([0, 1, 0]).astype(complex)
        t_span = (0, 1e-7)
        
        # Time fast mode
        fast_system = FastNVSystem(B_field=B_field, enable_noise=False)
        start = time.time()
        fast_system.evolve_unitary(rho0, t_span, n_steps=100)
        fast_time = time.time() - start
        
        # Time full mode
        full_system = NVSystem(B_field=B_field, noise_gen=None)
        start = time.time()
        full_system.evolve(rho0, t_span)
        full_time = time.time() - start
        
        # Fast mode should be significantly faster
        speedup = full_time / fast_time
        self.assertGreater(speedup, 2.0, "Fast mode should be at least 2x faster")
        
    def test_memory_efficiency(self):
        """Test memory usage remains reasonable for long simulations."""
        # This would require memory profiling
        # For now, we just ensure simulation completes
        fast_system = FastNVSystem(enable_noise=True)
        rho0 = np.diag([0, 1, 0]).astype(complex)
        
        # Run moderately long simulation
        times, rho_history = fast_system.evolve_unitary(rho0, (0, 1e-6), n_steps=1000)
        
        # Should complete without memory errors
        self.assertEqual(len(times), len(rho_history))


def run_all_tests():
    """
    Run all test suites and generate coverage report.
    
    Returns:
        unittest.TestResult: Test results
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    test_cases = [
        TestNVSpinOperators,
        TestNVSystemHamiltonian,
        TestNVSystem,
        TestFastNVSystem,
        TestNoiseGeneration,
        TestQUSIMCore,
        TestPhysicsValidation,
        TestPerformance
    ]
    
    for test_case in test_cases:
        suite.addTests(loader.loadTestsFromTestCase(test_case))
        
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print("="*70)
    
    return result


if __name__ == "__main__":
    # Run all tests
    result = run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if result.wasSuccessful() else 1)