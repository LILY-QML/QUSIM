#!/usr/bin/env python3
"""
NV Center π-Pulse Readout Experiment

This experiment simulates a complete NV center measurement protocol:
1. Initialize in |0⟩ ground state
2. Apply microwave π-pulse (|0⟩ → |±1⟩)  
3. Laser readout with photon detection
4. Measure fluorescence contrast

Author: QUSIM Development Team
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time

# Add QUSIM modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvcore'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvcore', 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvcore', 'helper'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvcore', 'modules'))

from nvcore import NVSystem, NVSpinOperators, NVSystemHamiltonian
from noise import NoiseGenerator, NoiseConfiguration
from noise_sources import SYSTEM


class PiPulseReadoutExperiment:
    """
    Complete NV Center π-Pulse Readout Experiment
    
    This class implements a realistic NV center experiment including:
    - Microwave π-pulse manipulation
    - Optical pumping and readout  
    - Photon detection with realistic noise
    - Fluorescence contrast measurement
    
    The experiment simulates the key physics:
    - |0⟩ state is bright (high fluorescence)
    - |±1⟩ states are dark (low fluorescence)
    - Shot noise in photon detection
    - Realistic readout fidelity
    """
    
    def __init__(self, B_field: np.ndarray = None, enable_noise: bool = True):
        """
        Initialize the experiment setup.
        
        Args:
            B_field: Magnetic field vector [Bx, By, Bz] in Tesla
            enable_noise: Whether to include realistic noise sources
        """
        if B_field is None:
            B_field = np.array([0.0, 0.0, 0.01])  # 10 mT along z-axis
            
        # Create noise configuration for realistic experiment
        self.noise_config = NoiseConfiguration()
        if enable_noise:
            self.noise_config.enable_optical = True
            self.noise_config.enable_microwave = True
            self.noise_config.enable_charge_noise = True
            self.noise_config.enable_temperature = True
        else:
            # Disable all noise for ideal experiment
            self.noise_config.enable_c13_bath = False
            self.noise_config.enable_optical = False
            self.noise_config.enable_microwave = False
            self.noise_config.enable_charge_noise = False
            self.noise_config.enable_temperature = False
            
        # Initialize NV system
        self.noise_gen = NoiseGenerator(self.noise_config)
        self.nv_system = NVSystem(B_field=B_field, noise_gen=self.noise_gen)
        
        # Experimental parameters
        self.laser_power = 1e-3  # 1 mW
        self.readout_time = 1e-6  # 1 μs readout
        self.collection_efficiency = 0.03  # 3% collection efficiency
        self.detector_efficiency = 0.8  # 80% quantum efficiency
        
        # Fluorescence rates (photons/s)
        self.bright_rate = 1e6   # |0⟩ state fluorescence rate
        self.dark_rate = 5e4     # |±1⟩ state fluorescence rate
        self.background_rate = 1e3  # Background counts
        
        print("🔬 NV Center π-Pulse Readout Experiment Initialized")
        print(f"   Magnetic field: {B_field} T")
        print(f"   Noise enabled: {enable_noise}")
        print(f"   Readout time: {self.readout_time*1e6:.1f} μs")
        
    def create_pi_pulse(self, rabi_frequency: float = None) -> Dict:
        """
        Create a microwave π-pulse for |0⟩ → |±1⟩ transition.
        
        Args:
            rabi_frequency: Rabi frequency in Hz (default: 10 MHz)
            
        Returns:
            Pulse dictionary with timing and parameters
        """
        if rabi_frequency is None:
            rabi_frequency = 2 * np.pi * 10e6  # 10 MHz
            
        # π-pulse duration: π / Ω_Rabi
        pulse_duration = np.pi / rabi_frequency
        
        pulse = {
            'duration': pulse_duration,
            'rabi_frequency': rabi_frequency,
            'phase': 0.0,  # X rotation
            'detuning': 0.0,  # On resonance
            'amplitude': 1.0
        }
        
        print(f"🌊 π-Pulse created:")
        print(f"   Duration: {pulse_duration*1e9:.1f} ns")
        print(f"   Rabi frequency: {rabi_frequency/(2*np.pi*1e6):.1f} MHz")
        
        return pulse
        
    def apply_microwave_pulse(self, initial_state: np.ndarray, pulse: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply microwave pulse to the NV center.
        
        Args:
            initial_state: Initial density matrix
            pulse: Pulse parameters
            
        Returns:
            Tuple of (times, density_matrices)
        """
        print("⚡ Applying microwave π-pulse...")
        
        # Evolve with microwave pulse
        times, rho_history = self.nv_system.evolve_with_pulses(
            initial_state, 
            [pulse]
        )
        
        final_state = rho_history[-1]
        populations = [rho[i,i].real for i in range(3)]
        
        print(f"   Final populations: |−1⟩={populations[0]:.3f}, |0⟩={populations[1]:.3f}, |+1⟩={populations[2]:.3f}")
        
        return times, rho_history
        
    def simulate_photon_detection(self, final_state: np.ndarray, num_measurements: int = 1000) -> Tuple[np.ndarray, Dict]:
        """
        Simulate realistic photon detection with shot noise.
        
        Args:
            final_state: Quantum state after π-pulse
            num_measurements: Number of experimental repetitions
            
        Returns:
            Tuple of (photon_counts, statistics)
        """
        print(f"📸 Simulating photon detection ({num_measurements} measurements)...")
        
        # Extract state populations
        pop_minus1 = final_state[0, 0].real
        pop_0 = final_state[1, 1].real  
        pop_plus1 = final_state[2, 2].real
        
        # Calculate expected fluorescence rate
        # |0⟩ is bright, |±1⟩ are dark
        expected_rate = (
            pop_0 * self.bright_rate +
            (pop_minus1 + pop_plus1) * self.dark_rate +
            self.background_rate
        )
        
        # Account for collection and detection efficiency
        detected_rate = expected_rate * self.collection_efficiency * self.detector_efficiency
        
        # Expected photon counts during readout
        expected_counts = detected_rate * self.readout_time
        
        print(f"   Expected rate: {expected_rate/1e6:.2f} Mcps")
        print(f"   Detected rate: {detected_rate/1e6:.2f} Mcps") 
        print(f"   Expected counts: {expected_counts:.1f} per measurement")
        
        # Simulate shot noise with Poisson statistics
        photon_counts = np.random.poisson(expected_counts, num_measurements)
        
        # Calculate statistics
        stats = {
            'mean_counts': np.mean(photon_counts),
            'std_counts': np.std(photon_counts),
            'snr': np.mean(photon_counts) / np.std(photon_counts),
            'expected_counts': expected_counts,
            'collection_efficiency': self.collection_efficiency,
            'detector_efficiency': self.detector_efficiency,
            'readout_time': self.readout_time,
            'populations': {'minus1': pop_minus1, '0': pop_0, 'plus1': pop_plus1}
        }
        
        return photon_counts, stats
        
    def calculate_readout_fidelity(self, bright_counts: np.ndarray, dark_counts: np.ndarray) -> Dict:
        """
        Calculate readout fidelity from bright/dark state measurements.
        
        Args:
            bright_counts: Photon counts when NV is in |0⟩ (bright)
            dark_counts: Photon counts when NV is in |±1⟩ (dark)
            
        Returns:
            Dictionary with fidelity metrics
        """
        # Find optimal threshold
        all_counts = np.concatenate([bright_counts, dark_counts])
        thresholds = np.linspace(np.min(all_counts), np.max(all_counts), 100)
        
        best_fidelity = 0
        best_threshold = 0
        
        for threshold in thresholds:
            # True positives: bright counts above threshold
            tp = np.sum(bright_counts >= threshold)
            # True negatives: dark counts below threshold  
            tn = np.sum(dark_counts < threshold)
            
            fidelity = (tp + tn) / (len(bright_counts) + len(dark_counts))
            
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_threshold = threshold
                
        contrast = (np.mean(bright_counts) - np.mean(dark_counts)) / (np.mean(bright_counts) + np.mean(dark_counts))
        
        return {
            'fidelity': best_fidelity,
            'threshold': best_threshold,
            'contrast': contrast,
            'bright_mean': np.mean(bright_counts),
            'dark_mean': np.mean(dark_counts),
            'bright_std': np.std(bright_counts),
            'dark_std': np.std(dark_counts)
        }
        
    def run_complete_experiment(self, num_measurements: int = 1000, rabi_frequency: float = None) -> Dict:
        """
        Run the complete π-pulse readout experiment.
        
        Args:
            num_measurements: Number of experimental repetitions
            rabi_frequency: Microwave Rabi frequency in Hz
            
        Returns:
            Complete experimental results
        """
        print("🚀 Starting Complete π-Pulse Readout Experiment")
        print("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Initialize in |0⟩ ground state
        print("1️⃣  Initializing NV in |0⟩ ground state...")
        initial_state = self.nv_system.create_initial_state('ms0')
        
        # Reference measurement: |0⟩ state (bright)
        print("2️⃣  Reference measurement (|0⟩ bright state)...")
        bright_counts, bright_stats = self.simulate_photon_detection(initial_state, num_measurements)
        
        # Step 2: Apply π-pulse
        print("3️⃣  Applying π-pulse (|0⟩ → |±1⟩)...")
        pi_pulse = self.create_pi_pulse(rabi_frequency)
        times, rho_history = self.apply_microwave_pulse(initial_state, pi_pulse)
        final_state = rho_history[-1]
        
        # Step 3: Readout after π-pulse (should be dark)
        print("4️⃣  Readout measurement (|±1⟩ dark state)...")
        dark_counts, dark_stats = self.simulate_photon_detection(final_state, num_measurements)
        
        # Step 4: Calculate fidelity
        print("5️⃣  Calculating readout fidelity...")
        fidelity_results = self.calculate_readout_fidelity(bright_counts, dark_counts)
        
        elapsed_time = time.time() - start_time
        
        # Compile results
        results = {
            'experimental_parameters': {
                'num_measurements': num_measurements,
                'pi_pulse': pi_pulse,
                'readout_time': self.readout_time,
                'laser_power': self.laser_power
            },
            'bright_measurement': {
                'counts': bright_counts,
                'statistics': bright_stats
            },
            'dark_measurement': {
                'counts': dark_counts, 
                'statistics': dark_stats
            },
            'readout_fidelity': fidelity_results,
            'quantum_evolution': {
                'times': times,
                'density_matrices': rho_history
            },
            'execution_time': elapsed_time
        }
        
        print("6️⃣  Experiment completed!")
        print(f"   Execution time: {elapsed_time:.2f} s")
        print("=" * 60)
        
        return results
        
    def plot_experimental_results(self, results: Dict, save_path: str = None):
        """
        Create comprehensive plots of experimental results.
        
        Args:
            results: Results from run_complete_experiment
            save_path: Optional path to save plots
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Photon count histograms
        ax = axes[0, 0]
        bright_counts = results['bright_measurement']['counts']
        dark_counts = results['dark_measurement']['counts']
        
        bins = np.linspace(min(np.min(bright_counts), np.min(dark_counts)), 
                          max(np.max(bright_counts), np.max(dark_counts)), 30)
        
        ax.hist(bright_counts, bins=bins, alpha=0.7, label='|0⟩ (bright)', color='orange', density=True)
        ax.hist(dark_counts, bins=bins, alpha=0.7, label='|±1⟩ (dark)', color='blue', density=True)
        ax.axvline(results['readout_fidelity']['threshold'], color='red', linestyle='--', label='Threshold')
        ax.set_xlabel('Photon Counts')
        ax.set_ylabel('Probability Density')
        ax.set_title('Photon Count Distributions')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Population evolution during π-pulse
        ax = axes[0, 1]
        times = results['quantum_evolution']['times']
        rho_history = results['quantum_evolution']['density_matrices']
        
        populations = np.array([[rho[i,i].real for i in range(3)] for rho in rho_history])
        
        ax.plot(times*1e9, populations[:, 0], 'b-', label='|−1⟩', linewidth=2)
        ax.plot(times*1e9, populations[:, 1], 'r-', label='|0⟩', linewidth=2)  
        ax.plot(times*1e9, populations[:, 2], 'g-', label='|+1⟩', linewidth=2)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Population')
        ax.set_title('State Evolution During π-Pulse')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Readout contrast
        ax = axes[0, 2]
        contrast = results['readout_fidelity']['contrast']
        fidelity = results['readout_fidelity']['fidelity']
        
        categories = ['Bright Mean', 'Dark Mean']
        values = [results['readout_fidelity']['bright_mean'], results['readout_fidelity']['dark_mean']]
        errors = [results['readout_fidelity']['bright_std'], results['readout_fidelity']['dark_std']]
        
        bars = ax.bar(categories, values, yerr=errors, capsize=5, color=['orange', 'blue'], alpha=0.7)
        ax.set_ylabel('Photon Counts')
        ax.set_title(f'Readout Contrast\nC = {contrast:.3f}, F = {fidelity:.3f}')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value, error in zip(bars, values, errors):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + error + 0.5,
                   f'{value:.1f}', ha='center', va='bottom')
        
        # 4. Time trace of photon counts
        ax = axes[1, 0]
        measurement_indices = np.arange(len(bright_counts))
        ax.plot(measurement_indices[:100], bright_counts[:100], 'o-', color='orange', alpha=0.7, label='|0⟩ (bright)')
        ax.plot(measurement_indices[:100], dark_counts[:100], 's-', color='blue', alpha=0.7, label='|±1⟩ (dark)')
        ax.axhline(results['readout_fidelity']['threshold'], color='red', linestyle='--', label='Threshold')
        ax.set_xlabel('Measurement #')
        ax.set_ylabel('Photon Counts')
        ax.set_title('Single-Shot Measurements (first 100)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Coherence evolution
        ax = axes[1, 1]
        coherences = np.array([np.abs(rho[0, 1]) for rho in rho_history])
        ax.plot(times*1e9, coherences, 'purple', linewidth=2)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('|ρ₋₁,₀|')
        ax.set_title('Coherence During π-Pulse')
        ax.grid(True, alpha=0.3)
        
        # 6. Experimental summary
        ax = axes[1, 2]
        ax.axis('off')
        
        # Create summary text
        summary_text = f"""
Experimental Summary
━━━━━━━━━━━━━━━━━━━━━
Measurements: {results['experimental_parameters']['num_measurements']}
π-Pulse Duration: {results['experimental_parameters']['pi_pulse']['duration']*1e9:.1f} ns
Readout Time: {results['experimental_parameters']['readout_time']*1e6:.1f} μs

Results
━━━━━━━━━━━━━━━━━━━━━  
Readout Fidelity: {results['readout_fidelity']['fidelity']:.3f}
Contrast: {results['readout_fidelity']['contrast']:.3f}
SNR (bright): {results['bright_measurement']['statistics']['snr']:.1f}
SNR (dark): {results['dark_measurement']['statistics']['snr']:.1f}

Bright State: {results['readout_fidelity']['bright_mean']:.1f} ± {results['readout_fidelity']['bright_std']:.1f}
Dark State: {results['readout_fidelity']['dark_mean']:.1f} ± {results['readout_fidelity']['dark_std']:.1f}

Execution Time: {results['execution_time']:.2f} s
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.suptitle('NV Center π-Pulse Readout Experiment Results', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
        
        plt.show()
        
    def print_experimental_summary(self, results: Dict):
        """Print a detailed summary of experimental results."""
        print("\n" + "="*70)
        print("🧪 NV CENTER π-PULSE READOUT EXPERIMENT RESULTS")
        print("="*70)
        
        # Experimental parameters
        params = results['experimental_parameters']
        print("\n📋 EXPERIMENTAL PARAMETERS:")
        print(f"   Number of measurements: {params['num_measurements']}")
        print(f"   π-Pulse duration: {params['pi_pulse']['duration']*1e9:.1f} ns")
        print(f"   Rabi frequency: {params['pi_pulse']['rabi_frequency']/(2*np.pi*1e6):.1f} MHz")
        print(f"   Readout time: {params['readout_time']*1e6:.1f} μs")
        print(f"   Laser power: {params['laser_power']*1e3:.1f} mW")
        
        # State populations after π-pulse
        dark_pops = results['dark_measurement']['statistics']['populations']
        print(f"\n⚛️  FINAL STATE POPULATIONS (after π-pulse):")
        print(f"   |−1⟩ population: {dark_pops['minus1']:.3f}")
        print(f"   |0⟩ population:  {dark_pops['0']:.3f}")
        print(f"   |+1⟩ population: {dark_pops['plus1']:.3f}")
        
        # Photon detection results
        bright_stats = results['bright_measurement']['statistics']
        dark_stats = results['dark_measurement']['statistics']
        
        print(f"\n📸 PHOTON DETECTION RESULTS:")
        print(f"   Bright state (|0⟩): {bright_stats['mean_counts']:.1f} ± {bright_stats['std_counts']:.1f} counts")
        print(f"   Dark state (|±1⟩):  {dark_stats['mean_counts']:.1f} ± {dark_stats['std_counts']:.1f} counts")
        print(f"   SNR (bright): {bright_stats['snr']:.1f}")
        print(f"   SNR (dark):   {dark_stats['snr']:.1f}")
        
        # Readout fidelity
        fidelity = results['readout_fidelity']
        print(f"\n🎯 READOUT FIDELITY:")
        print(f"   Fidelity: {fidelity['fidelity']:.3f} ({fidelity['fidelity']*100:.1f}%)")
        print(f"   Contrast: {fidelity['contrast']:.3f}")
        print(f"   Threshold: {fidelity['threshold']:.1f} counts")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Execution time: {results['execution_time']:.2f} s")
        print(f"   Measurements/s: {params['num_measurements']/results['execution_time']:.0f}")
        
        print("="*70)


def main():
    """Run the π-pulse readout experiment."""
    print("🚀 NV Center π-Pulse Readout Experiment")
    print("=" * 50)
    
    # Create experiment
    experiment = PiPulseReadoutExperiment(
        B_field=np.array([0.0, 0.0, 0.01]),  # 10 mT
        enable_noise=True
    )
    
    # Run experiment
    results = experiment.run_complete_experiment(
        num_measurements=1000,
        rabi_frequency=2 * np.pi * 10e6  # 10 MHz
    )
    
    # Print summary
    experiment.print_experimental_summary(results)
    
    # Create plots
    save_path = os.path.join(os.path.dirname(__file__), 'pi_pulse_readout_results.png')
    experiment.plot_experimental_results(results, save_path)
    
    return results


if __name__ == "__main__":
    results = main()