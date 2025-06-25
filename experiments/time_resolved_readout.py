#!/usr/bin/env python3
"""
Time-Resolved Photon Counting Experiment

Realistic NV center experiment:
1. Apply MW π-pulse at resonance frequency
2. Start laser readout immediately after pulse
3. Count photons every nanosecond for 600 ns
4. Show real-time photon emission dynamics

This simulates what you see on a real photon counter with ns time resolution.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import time

# Add QUSIM modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvcore'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvcore', 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvcore', 'helper'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nvcore', 'modules'))

from nvcore import NVSystem, NVSpinOperators
from noise import NoiseGenerator, NoiseConfiguration
from noise_sources import SYSTEM


class TimeResolvedPhotonCounter:
    """
    Time-resolved photon counting experiment with nanosecond resolution.
    
    Simulates realistic photon detection dynamics during laser readout
    following a microwave π-pulse manipulation.
    """
    
    def __init__(self):
        """Initialize the time-resolved photon counting setup."""
        
        # Experimental setup
        self.B_field = np.array([0.0, 0.0, 0.01])  # 10 mT field
        
        # Create minimal noise for realistic experiment
        noise_config = NoiseConfiguration()
        noise_config.enable_optical = True     # Laser RIN and shot noise
        noise_config.enable_charge_noise = True  # Charge state jumps
        noise_config.enable_c13_bath = False   # Too slow for ns resolution
        noise_config.enable_temperature = False
        noise_config.enable_johnson = False
        noise_config.enable_external_field = False
        noise_config.enable_strain = False
        noise_config.enable_microwave = False
        
        self.noise_gen = NoiseGenerator(noise_config)
        self.nv_system = NVSystem(B_field=self.B_field, noise_gen=self.noise_gen)
        
        # Get NV transition frequency for resonance
        gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
        self.mw_frequency = gamma_e * np.linalg.norm(self.B_field)  # Zeeman frequency
        
        # Photon emission parameters
        self.bright_rate = 1.2e6   # |0⟩ state emission rate (photons/s)
        self.dark_rate = 0.05e6    # |±1⟩ state emission rate (photons/s)
        self.collection_eff = 0.03  # Collection efficiency
        self.detector_eff = 0.8     # Detector quantum efficiency
        self.background_rate = 1e3  # Background counts
        
        # Readout parameters
        self.readout_duration = 600e-9  # 600 ns total readout
        self.time_bin = 1e-9           # 1 ns time bins
        self.n_time_bins = int(self.readout_duration / self.time_bin)
        
        print("📡 Time-Resolved Photon Counter Initialized")
        print(f"   Magnetic field: {self.B_field[2]*1000:.1f} mT")
        print(f"   MW frequency: {self.mw_frequency/1e9:.3f} GHz")
        print(f"   Readout duration: {self.readout_duration*1e9:.0f} ns")
        print(f"   Time resolution: {self.time_bin*1e9:.0f} ns")
        print(f"   Time bins: {self.n_time_bins}")
        
    def create_resonant_pi_pulse(self) -> Dict:
        """Create a resonant π-pulse at the NV transition frequency."""
        
        # π-pulse parameters
        rabi_frequency = 2 * np.pi * 15e6  # 15 MHz Rabi frequency
        pulse_duration = np.pi / rabi_frequency  # π-pulse duration
        
        pulse = {
            'duration': pulse_duration,
            'rabi_frequency': rabi_frequency,
            'frequency': self.mw_frequency,  # Resonant frequency
            'phase': 0.0,
            'detuning': 0.0,  # Exactly on resonance
            'amplitude': 1.0
        }
        
        print(f"🌊 Resonant π-Pulse:")
        print(f"   Frequency: {self.mw_frequency/1e9:.3f} GHz")
        print(f"   Duration: {pulse_duration*1e9:.1f} ns")
        print(f"   Rabi frequency: {rabi_frequency/(2*np.pi*1e6):.1f} MHz")
        
        return pulse
        
    def apply_pi_pulse(self, initial_state: np.ndarray) -> np.ndarray:
        """Apply the π-pulse and return final state."""
        
        print("⚡ Applying resonant π-pulse...")
        
        # Create and apply π-pulse
        pi_pulse = self.create_resonant_pi_pulse()
        times, rho_history = self.nv_system.evolve_with_pulses(initial_state, [pi_pulse])
        
        final_state = rho_history[-1]
        
        # Show population transfer
        pop_0_initial = initial_state[1, 1].real
        pop_0_final = final_state[1, 1].real
        pop_plus1_final = final_state[2, 2].real
        
        print(f"   |0⟩ population: {pop_0_initial:.3f} → {pop_0_final:.3f}")
        print(f"   |+1⟩ population: 0.000 → {pop_plus1_final:.3f}")
        print(f"   Population transfer: {(1-pop_0_final)*100:.1f}%")
        
        return final_state
        
    def simulate_time_resolved_photons(self, quantum_state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate time-resolved photon detection with realistic dynamics.
        
        Args:
            quantum_state: Quantum state after π-pulse
            
        Returns:
            Tuple of (time_bins, photon_counts_per_ns)
        """
        
        print("📸 Starting time-resolved photon counting...")
        print(f"   Counting photons for {self.readout_duration*1e9:.0f} ns")
        print(f"   Time resolution: {self.time_bin*1e9:.0f} ns per bin")
        
        # Extract state populations
        pop_0 = quantum_state[1, 1].real
        pop_plus1 = quantum_state[2, 2].real
        pop_minus1 = quantum_state[0, 0].real
        
        print(f"   State populations: |−1⟩={pop_minus1:.3f}, |0⟩={pop_0:.3f}, |+1⟩={pop_plus1:.3f}")
        
        # Calculate emission rate based on state populations
        emission_rate = (
            pop_0 * self.bright_rate +
            (pop_plus1 + pop_minus1) * self.dark_rate +
            self.background_rate
        )
        
        # Account for detection efficiency
        detected_rate = emission_rate * self.collection_eff * self.detector_eff
        
        print(f"   Total emission rate: {emission_rate/1e6:.2f} Mcps")
        print(f"   Detected rate: {detected_rate/1e6:.3f} Mcps")
        print(f"   Expected counts per ns: {detected_rate * self.time_bin:.4f}")
        
        # Time bins
        time_bins = np.arange(0, self.readout_duration, self.time_bin)
        
        # Simulate realistic photon dynamics
        photon_counts = np.zeros(len(time_bins))
        
        # Add time-dependent effects for realism
        for i, t in enumerate(time_bins):
            
            # Base rate from quantum state
            base_rate = detected_rate
            
            # Add realistic effects:
            
            # 1. Laser power fluctuations (optical noise)
            if hasattr(self.noise_gen, 'optical_source'):
                laser_noise = 1.0 + 0.02 * np.random.randn()  # 2% RIN
            else:
                laser_noise = 1.0
                
            # 2. Charge state fluctuations (blinking)
            charge_noise = 1.0
            if hasattr(self.noise_gen, 'charge_source'):
                if np.random.random() < 0.001:  # Rare charge jumps
                    charge_noise = 0.1  # Temporary dark state
                    
            # 3. Detector response variations
            detector_noise = 1.0 + 0.05 * np.random.randn()  # 5% detector noise
            
            # Combined rate for this time bin
            effective_rate = base_rate * laser_noise * charge_noise * detector_noise
            effective_rate = max(0, effective_rate)  # No negative rates
            
            # Expected counts in this time bin
            expected_counts = effective_rate * self.time_bin
            
            # Poisson statistics (shot noise)
            if expected_counts > 0:
                photon_counts[i] = np.random.poisson(expected_counts)
            else:
                photon_counts[i] = 0
                
        total_counts = np.sum(photon_counts)
        avg_rate = total_counts / self.readout_duration
        
        print(f"   Total photons detected: {total_counts:.0f}")
        print(f"   Average rate: {avg_rate/1e6:.3f} Mcps")
        print(f"   Peak count rate: {np.max(photon_counts)/self.time_bin/1e6:.3f} Mcps")
        
        return time_bins, photon_counts
        
    def run_time_resolved_experiment(self, num_shots: int = 1) -> Dict:
        """
        Run the complete time-resolved experiment.
        
        Args:
            num_shots: Number of experimental repetitions
            
        Returns:
            Dictionary with experimental results
        """
        
        print("🚀 Starting Time-Resolved Photon Counting Experiment")
        print("=" * 65)
        
        start_time = time.time()
        
        # Step 1: Initialize in |0⟩
        print("1️⃣  Initializing NV in |0⟩ ground state...")
        initial_state = self.nv_system.create_initial_state('ms0')
        
        # Step 2: Apply resonant π-pulse
        print("2️⃣  Applying resonant π-pulse...")
        final_state = self.apply_pi_pulse(initial_state)
        
        # Step 3: Time-resolved photon counting
        print("3️⃣  Time-resolved photon detection...")
        
        all_time_traces = []
        time_bins = None
        
        for shot in range(num_shots):
            if num_shots > 1:
                print(f"   Shot {shot+1}/{num_shots}")
                
            time_bins, photon_counts = self.simulate_time_resolved_photons(final_state)
            all_time_traces.append(photon_counts)
            
        # Average over shots
        if num_shots > 1:
            avg_photon_counts = np.mean(all_time_traces, axis=0)
            std_photon_counts = np.std(all_time_traces, axis=0)
        else:
            avg_photon_counts = all_time_traces[0]
            std_photon_counts = np.sqrt(avg_photon_counts)  # Poisson noise estimate
            
        elapsed_time = time.time() - start_time
        
        results = {
            'time_bins_ns': time_bins * 1e9,  # Convert to ns for plotting
            'photon_counts': avg_photon_counts,
            'photon_std': std_photon_counts,
            'all_traces': all_time_traces,
            'total_counts': np.sum(avg_photon_counts),
            'peak_count_rate': np.max(avg_photon_counts) / self.time_bin,
            'average_count_rate': np.sum(avg_photon_counts) / self.readout_duration,
            'quantum_state': final_state,
            'experimental_params': {
                'num_shots': num_shots,
                'readout_duration_ns': self.readout_duration * 1e9,
                'time_resolution_ns': self.time_bin * 1e9,
                'mw_frequency_ghz': self.mw_frequency / 1e9
            },
            'execution_time': elapsed_time
        }
        
        print("4️⃣  Experiment completed!")
        print(f"   Execution time: {elapsed_time:.2f} s")
        print("=" * 65)
        
        return results
        
    def plot_time_resolved_results(self, results: Dict, save_path: str = None):
        """Create plots of time-resolved photon counting results."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        time_ns = results['time_bins_ns']
        counts = results['photon_counts']
        errors = results['photon_std']
        
        # 1. Main photon time trace
        ax = axes[0, 0]
        ax.errorbar(time_ns, counts, yerr=errors, fmt='o-', markersize=3, 
                   linewidth=1, capsize=2, alpha=0.8, color='blue')
        ax.fill_between(time_ns, counts-errors, counts+errors, alpha=0.3, color='blue')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Photon Counts per ns')
        ax.set_title('Time-Resolved Photon Emission')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        total_counts = results['total_counts']
        avg_rate = results['average_count_rate']
        ax.text(0.02, 0.98, f'Total: {total_counts:.0f} photons\nAvg Rate: {avg_rate/1e6:.3f} Mcps', 
                transform=ax.transAxes, va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # 2. Zoomed view of first 100 ns
        ax = axes[0, 1]
        zoom_mask = time_ns <= 100
        ax.plot(time_ns[zoom_mask], counts[zoom_mask], 'o-', markersize=4, 
               linewidth=2, color='red', alpha=0.8)
        ax.fill_between(time_ns[zoom_mask], 
                       counts[zoom_mask] - errors[zoom_mask],
                       counts[zoom_mask] + errors[zoom_mask], 
                       alpha=0.3, color='red')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Photon Counts per ns')
        ax.set_title('First 100 ns (High Resolution)')
        ax.grid(True, alpha=0.3)
        
        # 3. Cumulative photon counts
        ax = axes[1, 0]
        cumulative = np.cumsum(counts)
        ax.plot(time_ns, cumulative, '-', linewidth=2, color='green')
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Cumulative Photon Counts')
        ax.set_title('Cumulative Photon Detection')
        ax.grid(True, alpha=0.3)
        
        # Add fit line for average rate
        fit_line = avg_rate * time_ns * 1e-9
        ax.plot(time_ns, fit_line, '--', color='orange', linewidth=2, 
               label=f'Linear fit: {avg_rate/1e6:.3f} Mcps')
        ax.legend()
        
        # 4. Count rate histogram
        ax = axes[1, 1]
        count_rates = counts / self.time_bin  # Convert to Hz
        ax.hist(count_rates, bins=30, alpha=0.7, color='purple', edgecolor='black')
        ax.set_xlabel('Count Rate (Hz)')
        ax.set_ylabel('Frequency')
        ax.set_title('Count Rate Distribution')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_rate = np.mean(count_rates)
        std_rate = np.std(count_rates)
        ax.axvline(mean_rate, color='red', linestyle='--', linewidth=2,
                  label=f'Mean: {mean_rate/1e6:.3f} Mcps')
        ax.axvline(mean_rate + std_rate, color='orange', linestyle=':', 
                  label=f'±σ: {std_rate/1e6:.3f} Mcps')
        ax.axvline(mean_rate - std_rate, color='orange', linestyle=':')
        ax.legend()
        
        plt.tight_layout()
        plt.suptitle('Time-Resolved Photon Counting Results', fontsize=16, y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plot saved to: {save_path}")
            
        plt.show()
        
    def print_experiment_summary(self, results: Dict):
        """Print detailed experimental summary."""
        
        print("\n" + "="*70)
        print("📡 TIME-RESOLVED PHOTON COUNTING EXPERIMENT RESULTS")
        print("="*70)
        
        params = results['experimental_params']
        
        print(f"\n📋 EXPERIMENTAL PARAMETERS:")
        print(f"   MW frequency: {params['mw_frequency_ghz']:.3f} GHz (resonant)")
        print(f"   Readout duration: {params['readout_duration_ns']:.0f} ns")
        print(f"   Time resolution: {params['time_resolution_ns']:.0f} ns")
        print(f"   Number of shots: {params['num_shots']}")
        
        print(f"\n📸 PHOTON DETECTION RESULTS:")
        print(f"   Total photons: {results['total_counts']:.0f}")
        print(f"   Average rate: {results['average_count_rate']/1e6:.3f} Mcps")
        print(f"   Peak rate: {results['peak_count_rate']/1e6:.3f} Mcps")
        
        # Calculate some interesting statistics
        counts = results['photon_counts']
        time_bins = results['time_bins_ns']
        
        # Find peak time
        peak_idx = np.argmax(counts)
        peak_time = time_bins[peak_idx]
        
        # Calculate signal statistics
        mean_counts = np.mean(counts)
        std_counts = np.std(counts)
        snr = mean_counts / std_counts if std_counts > 0 else 0
        
        print(f"\n📊 SIGNAL STATISTICS:")
        print(f"   Peak at: {peak_time:.0f} ns ({counts[peak_idx]:.0f} counts)")
        print(f"   Mean counts/ns: {mean_counts:.2f} ± {std_counts:.2f}")
        print(f"   Signal-to-noise: {snr:.1f}")
        
        # State information
        state = results['quantum_state']
        pops = [state[i,i].real for i in range(3)]
        
        print(f"\n⚛️  QUANTUM STATE (after π-pulse):")
        print(f"   |−1⟩ population: {pops[0]:.3f}")
        print(f"   |0⟩ population:  {pops[1]:.3f}")
        print(f"   |+1⟩ population: {pops[2]:.3f}")
        
        print(f"\n⚡ PERFORMANCE:")
        print(f"   Execution time: {results['execution_time']:.2f} s")
        
        print("="*70)


def main():
    """Run the time-resolved photon counting experiment."""
    
    print("📡 Time-Resolved Photon Counting Experiment")
    print("=" * 50)
    
    # Create experiment
    counter = TimeResolvedPhotonCounter()
    
    # Run experiment
    results = counter.run_time_resolved_experiment(num_shots=1)
    
    # Print results
    counter.print_experiment_summary(results)
    
    # Create plots
    save_path = os.path.join(os.path.dirname(__file__), 'time_resolved_photons.png')
    counter.plot_time_resolved_results(results, save_path)
    
    return results


if __name__ == "__main__":
    results = main()