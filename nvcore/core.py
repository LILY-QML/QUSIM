#!/usr/bin/env python3
"""
QUSIM NV Center Quantum Simulation Core Module

This module provides a unified command-line interface for running NV center quantum simulations
with various configurations and modes. It integrates all simulation capabilities including
noise modeling, pulse sequences, and different computational modes.

Examples:
    Run a fast simulation for 1 microsecond:
        $ python core.py --fast --time 1e-6
        
    Run demonstration mode with specific noise sources:
        $ python core.py --demo --noise c13_bath,charge_noise
        
    Run full simulation with all features:
        $ python core.py --time 1e-6 --pulse-sequence rabi --noise all
        
    Generate documentation:
        $ python core.py --generate-docs

Author: QUSIM Development Team
License: MIT
"""

import argparse
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

# Add module paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'helper'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

# Import core modules
from nvcore import NVSystem, NVSpinOperators, NVSystemHamiltonian
from nvcore_fast import FastNVSystem
from noise import NoiseGenerator, NoiseConfiguration
from noise_sources import SYSTEM


class QUSIMCore:
    """
    Main QUSIM simulation controller.
    
    This class manages the configuration and execution of NV center quantum simulations
    based on command-line arguments. It provides a unified interface for different
    simulation modes and configurations.
    
    Attributes:
        args: Parsed command-line arguments
        system: The NV system instance (regular or fast)
        noise_config: Noise configuration object
        logger: Logger instance for output
        
    Examples:
        >>> core = QUSIMCore()
        >>> core.parse_arguments(['--fast', '--time', '1e-6'])
        >>> results = core.run()
    """
    
    def __init__(self):
        """Initialize QUSIM core with default settings."""
        self.args = None
        self.system = None
        self.noise_config = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """
        Set up logging configuration.
        
        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger('QUSIM')
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        
        logger.addHandler(ch)
        return logger
        
    def parse_arguments(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        """
        Parse command-line arguments.
        
        Args:
            args: Optional list of arguments (for testing). If None, uses sys.argv
            
        Returns:
            argparse.Namespace: Parsed arguments
            
        Examples:
            >>> core = QUSIMCore()
            >>> args = core.parse_arguments(['--fast', '--time', '1e-6'])
            >>> print(args.fast)
            True
        """
        parser = argparse.ArgumentParser(
            description='QUSIM - Quantum Simulation of NV Centers',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  Fast simulation:
    %(prog)s --fast --time 1e-6
    
  Demo mode with specific noise:
    %(prog)s --demo --noise c13_bath,charge_noise
    
  Full simulation with pulse sequence:
    %(prog)s --time 1e-6 --pulse-sequence rabi --output results/
            """
        )
        
        # Simulation modes
        mode_group = parser.add_mutually_exclusive_group()
        mode_group.add_argument(
            '-f', '--fast',
            action='store_true',
            help='Use fast simulation mode (reduced physics accuracy for speed)'
        )
        mode_group.add_argument(
            '-d', '--demo',
            action='store_true',
            help='Run demonstration mode with visualization'
        )
        
        # Time parameters
        parser.add_argument(
            '-t', '--time',
            type=float,
            default=1e-6,
            help='Simulation time in seconds (default: 1e-6)'
        )
        parser.add_argument(
            '--dt',
            type=float,
            default=None,
            help='Time step in seconds (auto-determined if not specified)'
        )
        
        # Noise configuration
        parser.add_argument(
            '-n', '--noise',
            type=str,
            default='all',
            help='Comma-separated list of noise sources to enable. Options: '
                 'all, none, c13_bath, charge_noise, temperature, johnson, '
                 'external_field, strain, microwave, optical'
        )
        parser.add_argument(
            '--no-noise',
            action='store_true',
            help='Disable all noise sources'
        )
        
        # Magnetic field
        parser.add_argument(
            '-B', '--magnetic-field',
            type=float,
            nargs=3,
            default=[0.0, 0.0, 0.0],
            metavar=('Bx', 'By', 'Bz'),
            help='Static magnetic field components in Tesla'
        )
        
        # Pulse sequences
        parser.add_argument(
            '-p', '--pulse-sequence',
            type=str,
            choices=['rabi', 'ramsey', 'echo', 'cpmg', 'custom'],
            help='Predefined pulse sequence to apply'
        )
        parser.add_argument(
            '--pulse-file',
            type=str,
            help='JSON file containing custom pulse sequence'
        )
        
        # Initial state
        parser.add_argument(
            '-s', '--initial-state',
            type=str,
            default='ground',
            choices=['ground', 'excited', 'superposition', 'mixed'],
            help='Initial quantum state (default: ground)'
        )
        
        # Output options
        parser.add_argument(
            '-o', '--output',
            type=str,
            default='.',
            help='Output directory for results (default: current directory)'
        )
        parser.add_argument(
            '--plot',
            action='store_true',
            help='Generate plots of results'
        )
        parser.add_argument(
            '--save-trajectory',
            action='store_true',
            help='Save full density matrix trajectory'
        )
        
        # Advanced options
        parser.add_argument(
            '-v', '--verbose',
            action='store_true',
            help='Enable verbose output'
        )
        parser.add_argument(
            '--config',
            type=str,
            help='JSON configuration file for advanced settings'
        )
        parser.add_argument(
            '--benchmark',
            action='store_true',
            help='Run performance benchmarking'
        )
        parser.add_argument(
            '--generate-docs',
            action='store_true',
            help='Generate Sphinx documentation'
        )
        
        self.args = parser.parse_args(args)
        
        # Set logger level based on verbosity
        if self.args.verbose:
            self.logger.setLevel(logging.DEBUG)
            
        return self.args
        
    def _configure_noise(self) -> NoiseConfiguration:
        """
        Configure noise sources based on command-line arguments.
        
        Returns:
            NoiseConfiguration: Configured noise settings
            
        Raises:
            ValueError: If invalid noise source is specified
        """
        config = NoiseConfiguration()
        
        if self.args.no_noise or self.args.noise == 'none':
            # Disable all noise
            config.enable_c13_bath = False
            config.enable_charge_noise = False
            config.enable_temperature = False
            config.enable_johnson = False
            config.enable_external_field = False
            config.enable_strain = False
            config.enable_microwave = False
            config.enable_optical = False
        elif self.args.noise == 'all':
            # Enable all noise (default)
            pass
        else:
            # Parse specific noise sources
            noise_sources = [s.strip() for s in self.args.noise.split(',')]
            
            # First disable all
            config.enable_c13_bath = False
            config.enable_charge_noise = False
            config.enable_temperature = False
            config.enable_johnson = False
            config.enable_external_field = False
            config.enable_strain = False
            config.enable_microwave = False
            config.enable_optical = False
            
            # Then enable specified ones
            for source in noise_sources:
                if source == 'c13_bath':
                    config.enable_c13_bath = True
                elif source == 'charge_noise':
                    config.enable_charge_noise = True
                elif source == 'temperature':
                    config.enable_temperature = True
                elif source == 'johnson':
                    config.enable_johnson = True
                elif source == 'external_field':
                    config.enable_external_field = True
                elif source == 'strain':
                    config.enable_strain = True
                elif source == 'microwave':
                    config.enable_microwave = True
                elif source == 'optical':
                    config.enable_optical = True
                else:
                    raise ValueError(f"Unknown noise source: {source}")
                    
        # Set time step if specified
        if self.args.dt is not None:
            config.dt = self.args.dt
            
        self.noise_config = config
        return config
        
    def _create_initial_state(self) -> np.ndarray:
        """
        Create initial density matrix based on command-line arguments.
        
        Returns:
            np.ndarray: Initial density matrix (3x3 for NV center)
            
        Examples:
            >>> core = QUSIMCore()
            >>> core.args = argparse.Namespace(initial_state='ground')
            >>> rho0 = core._create_initial_state()
            >>> print(rho0[1, 1])  # Population in |0⟩ state
            1.0
        """
        if self.args.initial_state == 'ground':
            # |0⟩⟨0| state
            rho0 = np.zeros((3, 3), dtype=complex)
            rho0[1, 1] = 1.0
        elif self.args.initial_state == 'excited':
            # |+1⟩⟨+1| state
            rho0 = np.zeros((3, 3), dtype=complex)
            rho0[2, 2] = 1.0
        elif self.args.initial_state == 'superposition':
            # (|0⟩ + |+1⟩)/√2
            psi = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
            rho0 = np.outer(psi, psi.conj())
        elif self.args.initial_state == 'mixed':
            # Maximally mixed state
            rho0 = np.eye(3, dtype=complex) / 3
        else:
            raise ValueError(f"Unknown initial state: {self.args.initial_state}")
            
        return rho0
        
    def _create_pulse_sequence(self) -> List[Dict]:
        """
        Create pulse sequence based on command-line arguments.
        
        Returns:
            List[Dict]: List of pulse dictionaries
            
        Raises:
            ValueError: If pulse sequence is invalid
        """
        if self.args.pulse_file:
            # Load from file
            with open(self.args.pulse_file, 'r') as f:
                return json.load(f)
                
        if not self.args.pulse_sequence:
            return []
            
        # Load default Rabi frequency from empirical measurements
        default_rabi_freq = SYSTEM.get_empirical_param('microwave_system', 'default_rabi_frequency')
        default_rabi_omega = 2 * np.pi * default_rabi_freq
        
        # Predefined sequences
        if self.args.pulse_sequence == 'rabi':
            # Rabi oscillation
            return [{
                'duration': self.args.time,
                'rabi_frequency': default_rabi_omega,
                'phase': 0,
                'detuning': 0
            }]
        elif self.args.pulse_sequence == 'ramsey':
            # Ramsey sequence
            pi_2_time = np.pi / (2 * default_rabi_omega)  # π/2 pulse from Rabi frequency
            free_evolution = self.args.time - 2 * pi_2_time
            return [
                {
                    'duration': pi_2_time,
                    'rabi_frequency': default_rabi_omega,
                    'phase': 0,
                    'detuning': 0
                },
                {
                    'duration': free_evolution,
                    'rabi_frequency': 0,
                    'phase': 0,
                    'detuning': 0
                },
                {
                    'duration': pi_2_time,
                    'rabi_frequency': default_rabi_omega,
                    'phase': 0,
                    'detuning': 0
                }
            ]
        elif self.args.pulse_sequence == 'echo':
            # Hahn echo
            pi_2_time = np.pi / (2 * default_rabi_omega)  # π/2 pulse
            pi_time = np.pi / default_rabi_omega          # π pulse
            tau = (self.args.time - 2 * pi_2_time - pi_time) / 2
            return [
                {
                    'duration': pi_2_time,
                    'rabi_frequency': default_rabi_omega,
                    'phase': 0,
                    'detuning': 0
                },
                {
                    'duration': tau,
                    'rabi_frequency': 0,
                    'phase': 0,
                    'detuning': 0
                },
                {
                    'duration': pi_time,
                    'rabi_frequency': default_rabi_omega,
                    'phase': np.pi/2,  # Y pulse
                    'detuning': 0
                },
                {
                    'duration': tau,
                    'rabi_frequency': 0,
                    'phase': 0,
                    'detuning': 0
                },
                {
                    'duration': pi_2_time,
                    'rabi_frequency': default_rabi_omega,
                    'phase': 0,
                    'detuning': 0
                }
            ]
        else:
            raise ValueError(f"Unknown pulse sequence: {self.args.pulse_sequence}")
            
    def _initialize_system(self):
        """
        Initialize the appropriate NV system based on command-line arguments.
        
        Creates either a FastNVSystem or regular NVSystem instance depending
        on the selected mode.
        """
        B_field = np.array(self.args.magnetic_field)
        
        if self.args.fast:
            self.logger.info("Initializing fast NV system")
            self.system = FastNVSystem(
                B_field=B_field,
                enable_noise=not self.args.no_noise
            )
            if not self.args.no_noise:
                # Apply noise configuration to fast system
                self.system.noise_gen.config = self._configure_noise()
        else:
            self.logger.info("Initializing full NV system")
            noise_gen = None
            if not self.args.no_noise:
                noise_gen = NoiseGenerator(self._configure_noise())
            self.system = NVSystem(B_field=B_field, noise_gen=noise_gen)
            
    def run(self) -> Dict:
        """
        Run the simulation based on configured parameters.
        
        Returns:
            Dict: Dictionary containing simulation results with keys:
                - 'times': Array of time points
                - 'density_matrices': List of density matrices (if save_trajectory)
                - 'populations': Array of state populations vs time
                - 'coherences': Array of coherence values vs time
                - 'fidelity': Fidelity vs initial state (if applicable)
                - 'metadata': Simulation parameters and settings
                
        Examples:
            >>> core = QUSIMCore()
            >>> core.parse_arguments(['--fast', '--time', '1e-6'])
            >>> results = core.run()
            >>> print(results['times'][-1])
            1e-06
        """
        if self.args.generate_docs:
            self._generate_documentation()
            return {}
            
        # Initialize system
        self._initialize_system()
        
        # Create initial state
        rho0 = self._create_initial_state()
        
        # Get pulse sequence
        pulse_sequence = self._create_pulse_sequence()
        
        # Run simulation
        self.logger.info(f"Starting simulation for {self.args.time:.2e} seconds")
        
        if self.args.fast:
            # Fast mode
            if pulse_sequence:
                self.logger.warning("Pulse sequences not fully supported in fast mode")
            times, rho_history = self.system.evolve_unitary(
                rho0, 
                (0, self.args.time),
                n_steps=1000 if self.args.save_trajectory else 100
            )
        else:
            # Full mode
            if pulse_sequence:
                times, rho_history = self.system.evolve_with_pulses(
                    rho0,
                    pulse_sequence,
                    include_noise_sources=None  # Use all configured sources
                )
            else:
                times, rho_history = self.system.evolve(
                    rho0,
                    (0, self.args.time),
                    dt=self.args.dt
                )
                
        # Process results
        results = self._process_results(times, rho_history, rho0)
        
        # Save results
        self._save_results(results)
        
        # Plot if requested
        if self.args.plot or self.args.demo:
            self._plot_results(results)
            
        # Run benchmarking if requested
        if self.args.benchmark:
            self._run_benchmark()
            
        return results
        
    def _process_results(self, times: np.ndarray, rho_history: List[np.ndarray], 
                        rho0: np.ndarray) -> Dict:
        """
        Process simulation results into organized format.
        
        Args:
            times: Array of time points
            rho_history: List of density matrices at each time
            rho0: Initial density matrix
            
        Returns:
            Dict: Processed results
        """
        # Extract populations and coherences
        populations = np.array([[rho[i, i].real for i in range(3)] for rho in rho_history])
        coherences = np.array([
            [rho[0, 1], rho[0, 2], rho[1, 2]] for rho in rho_history
        ])
        
        # Calculate fidelity with initial state
        fidelity = np.array([
            np.real(np.trace(rho @ rho0)) for rho in rho_history
        ])
        
        # Prepare results dictionary
        results = {
            'times': times,
            'populations': populations,
            'coherences': coherences,
            'fidelity': fidelity,
            'metadata': {
                'simulation_mode': 'fast' if self.args.fast else 'full',
                'simulation_time': self.args.time,
                'time_step': self.args.dt,
                'magnetic_field': self.args.magnetic_field,
                'noise_sources': self.args.noise,
                'initial_state': self.args.initial_state,
                'pulse_sequence': self.args.pulse_sequence,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        if self.args.save_trajectory:
            results['density_matrices'] = rho_history
            
        return results
        
    def _save_results(self, results: Dict):
        """
        Save simulation results to output directory.
        
        Args:
            results: Results dictionary to save
        """
        # Create output directory if needed
        os.makedirs(self.args.output, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode = 'fast' if self.args.fast else 'full'
        filename = f"qusim_results_{mode}_{timestamp}.npz"
        filepath = os.path.join(self.args.output, filename)
        
        # Save results
        np.savez_compressed(
            filepath,
            **{k: v for k, v in results.items() if k != 'density_matrices'}
        )
        
        # Save density matrices separately if requested
        if self.args.save_trajectory and 'density_matrices' in results:
            traj_filename = f"qusim_trajectory_{mode}_{timestamp}.npz"
            traj_filepath = os.path.join(self.args.output, traj_filename)
            np.savez_compressed(
                traj_filepath,
                times=results['times'],
                density_matrices=results['density_matrices']
            )
            
        self.logger.info(f"Results saved to {filepath}")
        
    def _plot_results(self, results: Dict):
        """
        Generate plots of simulation results.
        
        Args:
            results: Results dictionary to plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Population dynamics
        ax = axes[0, 0]
        ax.plot(results['times'] * 1e6, results['populations'][:, 0], 'b-', label='|−1⟩')
        ax.plot(results['times'] * 1e6, results['populations'][:, 1], 'r-', label='|0⟩')
        ax.plot(results['times'] * 1e6, results['populations'][:, 2], 'g-', label='|+1⟩')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Population')
        ax.set_title('State Populations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Coherence magnitude
        ax = axes[0, 1]
        coherence_mag = np.abs(results['coherences'])
        ax.plot(results['times'] * 1e6, coherence_mag[:, 0], 'b-', label='|ρ₋₁,₀|')
        ax.plot(results['times'] * 1e6, coherence_mag[:, 1], 'r-', label='|ρ₋₁,₊₁|')
        ax.plot(results['times'] * 1e6, coherence_mag[:, 2], 'g-', label='|ρ₀,₊₁|')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Coherence Magnitude')
        ax.set_title('Quantum Coherences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Fidelity
        ax = axes[1, 0]
        ax.plot(results['times'] * 1e6, results['fidelity'], 'k-')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Fidelity')
        ax.set_title('Fidelity with Initial State')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
        
        # Purity
        ax = axes[1, 1]
        if 'density_matrices' in results:
            purity = np.array([
                np.real(np.trace(rho @ rho)) for rho in results['density_matrices']
            ])
            ax.plot(results['times'] * 1e6, purity, 'k-')
            ax.set_xlabel('Time (μs)')
            ax.set_ylabel('Purity')
            ax.set_title('State Purity')
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 1.05])
        else:
            ax.text(0.5, 0.5, 'Purity calculation requires\n--save-trajectory',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            
        plt.suptitle(f'QUSIM Results - {results["metadata"]["simulation_mode"].capitalize()} Mode')
        plt.tight_layout()
        
        if self.args.plot:
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"qusim_plot_{timestamp}.png"
            plot_filepath = os.path.join(self.args.output, plot_filename)
            plt.savefig(plot_filepath, dpi=150, bbox_inches='tight')
            self.logger.info(f"Plot saved to {plot_filepath}")
            
        if self.args.demo:
            plt.show()
        else:
            plt.close()
            
    def _run_benchmark(self):
        """Run performance benchmarking tests."""
        import time
        
        self.logger.info("Running performance benchmark...")
        
        # Benchmark parameters
        time_points = [1e-8, 1e-7, 1e-6]
        modes = ['full', 'fast']
        
        results = []
        
        for mode in modes:
            for sim_time in time_points:
                # Configure for benchmark
                self.args.fast = (mode == 'fast')
                self.args.time = sim_time
                self.args.save_trajectory = False
                self.args.plot = False
                
                # Reinitialize system
                self._initialize_system()
                
                # Time the simulation
                start_time = time.time()
                self.run()
                elapsed = time.time() - start_time
                
                results.append({
                    'mode': mode,
                    'simulation_time': sim_time,
                    'wall_time': elapsed,
                    'speedup': sim_time / elapsed
                })
                
                self.logger.info(
                    f"{mode} mode: {sim_time:.0e}s simulation in {elapsed:.3f}s "
                    f"(speedup: {sim_time/elapsed:.1f}x)"
                )
                
        # Save benchmark results
        benchmark_file = os.path.join(self.args.output, 'benchmark_results.json')
        with open(benchmark_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        self.logger.info(f"Benchmark results saved to {benchmark_file}")
        
    def _generate_documentation(self):
        """Generate Sphinx documentation."""
        self.logger.info("Generating Sphinx documentation...")
        
        # This would typically call sphinx-build
        # For now, we'll just create the necessary structure
        docs_dir = os.path.join(os.path.dirname(__file__), '..', 'docs')
        os.makedirs(docs_dir, exist_ok=True)
        
        self.logger.info(f"Documentation structure created in {docs_dir}")
        self.logger.info("Run 'sphinx-quickstart' in the docs directory to complete setup")


def main():
    """Main entry point for QUSIM core."""
    core = QUSIMCore()
    core.parse_arguments()
    core.run()


if __name__ == "__main__":
    main()