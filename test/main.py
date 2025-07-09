#!/usr/bin/env python3
"""
NV Center Experiment Simulator with Optimal Control capabilities
Executes experiments defined in JSON configuration files
"""

import json
import sys
import os
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from jax.scipy.linalg import expm
from typing import Dict, List, Tuple, Any, Optional

# Physical constants
PI = jnp.pi

class NVSimulator:
    """NV center quantum simulator with optimal control capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simulator with configuration"""
        self.config = config
        self.setup_parameters()
        self.setup_operators()
        self.setup_initial_state()
        
    def setup_parameters(self):
        """Extract and set all simulation parameters from config"""
        # System parameters
        sys_params = self.config['system_parameters']
        self.D_g_Hz = sys_params['D_g_Hz']
        self.D_e_Hz = sys_params['D_e_Hz']
        self.gamma_e_Hz_per_G = sys_params['gamma_e_Hz_per_G']
        self.B_vec_G = jnp.array(sys_params['B_vec_G'])
        
        # N14 hyperfine
        n14_params = sys_params['N14_hyperfine']
        self.A_para_N_Hz = n14_params['A_para_Hz']
        self.A_perp_N_Hz = n14_params['A_perp_Hz']
        
        # C13 cluster
        c13_params = sys_params['C13_cluster']
        self.N_C = c13_params['N_C']
        self.A_para_C_Hz = jnp.array(c13_params['A_para_Hz'])
        self.A_perp_C_Hz = jnp.array(c13_params['A_perp_Hz'])
        
        # Relaxation
        relax_params = sys_params['relaxation']
        self.T1_s = relax_params['T1_s']
        self.Tphi_s = relax_params['Tphi_s']
        
        # Optical parameters
        optical_params = sys_params['optical']
        self.Tau_rad_s = optical_params['Tau_rad_s']
        self.Tau_MS_s = optical_params['Tau_MS_s']
        self.k_ISC_ms0 = optical_params['k_ISC_ms0_factor'] / self.Tau_rad_s
        self.k_ISC_ms1 = optical_params['k_ISC_ms1_factor'] / self.Tau_rad_s
        
        # Readout parameters
        readout_params = sys_params['readout']
        self.Beta_max_Hz = readout_params['Beta_max_Hz']
        self.W_ms0_late = readout_params['W_ms0_late']
        self.W_ms1_late = readout_params['W_ms1_late']
        
        # Simulation parameters
        sim_params = self.config['simulation_parameters']
        self.DT_ns = sim_params['DT_ns']
        self.seed = sim_params.get('seed', 0)
        self.key_master = jax.random.PRNGKey(self.seed)
        
    def setup_operators(self):
        """Setup quantum operators"""
        # Electron spin-1 operators
        self.Sx_e = jnp.array([[0,1,0],[1,0,1],[0,1,0]], dtype=jnp.complex64) / jnp.sqrt(2)
        self.Sy_e = jnp.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=jnp.complex64) / jnp.sqrt(2)
        self.Sz_e = jnp.diag(jnp.array([-1,0,1], dtype=jnp.complex64))
        
        # N14 nuclear spin operators
        self.Ix_N = jnp.array([[0,1,0],[1,0,1],[0,1,0]], dtype=jnp.complex64) / jnp.sqrt(2)
        self.Iz_N = jnp.diag(jnp.array([1,0,-1])) * 0.5
        
        # C13 nuclear spin operators
        self.sigma_x = jnp.array([[0,1],[1,0]], dtype=jnp.complex64) * 0.5
        self.sigma_z = jnp.array([[0.5,0],[0,-0.5]], dtype=jnp.complex64)
        
        # Identity operators
        self.Eye_e = jnp.eye(3, dtype=jnp.complex64)
        self.Eye_N = jnp.eye(3, dtype=jnp.complex64)
        self.I_C_tot = jnp.eye(2**self.N_C, dtype=jnp.complex64)
        
        # Build composite operators
        self.Sz = jnp.kron(jnp.kron(self.Sz_e, self.Eye_N), self.I_C_tot)
        self.Sx = jnp.kron(jnp.kron(self.Sx_e, self.Eye_N), self.I_C_tot)
        self.Sy = jnp.kron(jnp.kron(self.Sy_e, self.Eye_N), self.I_C_tot)
        self.Sz_N = jnp.kron(jnp.kron(self.Eye_e, self.Iz_N), self.I_C_tot)
        self.Sx_N = jnp.kron(jnp.kron(self.Eye_e, self.Ix_N), self.I_C_tot)
        
        # C13 operators
        self.Iz_list = []
        self.Ix_list = []
        for k in range(self.N_C):
            op_z = [self.sigma_z if i==k else jnp.eye(2, dtype=jnp.complex64) for i in range(self.N_C)]
            op_x = [self.sigma_x if i==k else jnp.eye(2, dtype=jnp.complex64) for i in range(self.N_C)]
            self.Iz_list.append(jnp.kron(jnp.kron(self.Eye_e, self.Eye_N), self.kron_list(op_z)))
            self.Ix_list.append(jnp.kron(jnp.kron(self.Eye_e, self.Eye_N), self.kron_list(op_x)))
        
        # Projection operators
        self.Pg = jnp.kron(jnp.kron(self.Eye_e, self.Eye_N), self.I_C_tot)
        self.Sz0_g = jnp.kron(jnp.kron(jnp.diag(jnp.array([0,1,0], dtype=jnp.complex64)), self.Eye_N), self.I_C_tot)
        self.DIM = self.Sz.shape[0]
        
        # Static Hamiltonian
        self.H0 = self.build_static_hamiltonian()
        
    def kron_list(self, lst):
        """Kronecker product of a list of matrices"""
        out = lst[0]
        for m in lst[1:]:
            out = jnp.kron(out, m)
        return out
        
    def build_static_hamiltonian(self):
        """Build the static part of the Hamiltonian"""
        H = 2*PI*self.gamma_e_Hz_per_G * (self.B_vec_G[0]*self.Sx + self.B_vec_G[1]*self.Sy + self.B_vec_G[2]*self.Sz)
        H += 2*PI*(self.A_para_N_Hz*(self.Sz@self.Sz_N) + self.A_perp_N_Hz*(self.Sx@self.Sx_N))
        for j in range(self.N_C):
            H += 2*PI*(self.A_para_C_Hz[j]*(self.Sz@self.Iz_list[j]) + self.A_perp_C_Hz[j]*(self.Sx@self.Ix_list[j]))
        return H
        
    def setup_initial_state(self):
        """Setup initial quantum state"""
        ket_e = jnp.array([0,1,0], dtype=jnp.complex64)  # ms=0
        ket_N = jnp.array([0,1,0], dtype=jnp.complex64)  # mI=0
        ket_C = self.kron_list([jnp.array([1,0], dtype=jnp.complex64) for _ in range(self.N_C)])
        psi0 = jnp.kron(jnp.kron(ket_e, ket_N), ket_C)
        self.rho0 = jnp.outer(psi0, psi0.conj())
        
    def setup_target_state(self, target_config: Dict[str, Any]) -> jnp.ndarray:
        """Setup target quantum state for optimal control"""
        ket_e = jnp.array(target_config['electron'], dtype=jnp.complex64)
        ket_N = jnp.array(target_config['nitrogen'], dtype=jnp.complex64)
        ket_C = self.kron_list([jnp.array(c, dtype=jnp.complex64) for c in target_config['carbons']])
        psi_target = jnp.kron(jnp.kron(ket_e, ket_N), ket_C)
        return psi_target
        
    def dissipator(self, rho):
        """Apply dissipator for relaxation and dephasing"""
        out = 0j
        Ls = [
            jnp.sqrt(1/self.Tphi_s)*(self.Sz - jnp.trace(self.Sz)/self.DIM),
            jnp.sqrt(1/self.Tau_rad_s)*(self.Pg@self.Pg)
        ]
        for L in Ls:
            out += L@rho@L.conj().T - 0.5*(L.conj().T@L@rho + rho@L.conj().T@L)
        return out
        
    def mw_envelope(self, t, start, duration, omega_max, shape='cos'):
        """Microwave pulse envelope function"""
        if shape == 'cos':
            return jnp.where(
                (t >= start) & (t <= start + duration), 
                omega_max * 0.5 * (1 - jnp.cos(2*PI*(t-start)/duration)), 
                0.0
            )
        elif shape == 'square':
            return jnp.where((t >= start) & (t <= start + duration), omega_max, 0.0)
        elif shape == 'gaussian':
            sigma = duration / 6  # 3 sigma on each side
            center = start + duration / 2
            return jnp.where(
                (t >= start) & (t <= start + duration),
                omega_max * jnp.exp(-0.5 * ((t - center) / sigma) ** 2),
                0.0
            )
        else:
            raise ValueError(f"Unknown pulse shape: {shape}")
        
    def evolution_step(self, rho, t, dt, mw_params):
        """Single evolution step with phase control"""
        # Calculate MW field at this time
        omega = 0.0
        phi = 0.0
        delta = 0.0
        
        if mw_params:
            # Check if we have pulse arrays
            if 'pulse_array' in mw_params:
                idx = int(t / dt)
                if idx < len(mw_params['pulse_array']['omega_Hz']):
                    omega = mw_params['pulse_array']['omega_Hz'][idx]
                    phi = mw_params['pulse_array'].get('phase_rad', [0.0]*len(mw_params['pulse_array']['omega_Hz']))[idx]
                    delta = mw_params['pulse_array'].get('delta_Hz', [0.0]*len(mw_params['pulse_array']['omega_Hz']))[idx]
            else:
                # Simple pulse with envelope
                omega = self.mw_envelope(
                    t, 
                    mw_params['start_ns']*1e-9, 
                    mw_params['duration_ns']*1e-9,
                    mw_params.get('omega_rabi_Hz', self.config['system_parameters']['microwave']['Omega_Rabi_Hz']),
                    mw_params.get('shape', 'cos')
                )
                phi = mw_params.get('phase_rad', 0.0)
                delta = mw_params.get('delta_Hz', 0.0)
            
            # MW Hamiltonian with phase control
            Hx = jnp.cos(phi) * self.Sx + jnp.sin(phi) * self.Sy
            H = self.H0 + 2*PI*(omega * Hx + delta * self.Sz)
        else:
            H = self.H0
            
        # Evolution
        U = expm(-1j*H*dt/2)
        rho = U @ rho @ U.conj().T
        rho = rho + self.dissipator(rho) * dt
        rho = U @ rho @ U.conj().T
        return rho
        
    def run_experiment(self, experiment: Dict[str, Any]):
        """Run experiment based on JSON configuration"""
        sequence = experiment['sequence']
        total_time_ns = experiment['total_time_ns']
        
        # Initialize density matrix
        rho = self.rho0
        
        # Time array
        times = jnp.arange(0, total_time_ns*1e-9, self.DT_ns*1e-9)
        
        # Results storage
        results = {
            'times_ns': times * 1e9,
            'population_ms0': [],
            'population_ms1': [],
            'population_ms_minus1': [],
            'photon_counts': None,
            'fidelity': None
        }
        
        # Find all MW pulses and photon counters in sequence
        mw_pulses = [e for e in sequence if e['type'] == 'mw_pulse']
        photon_counters = [e for e in sequence if e['type'] == 'photon_counter']
        laser_readouts = [e for e in sequence if e['type'] == 'laser_readout']
                
        # Setup target state if specified
        psi_target = None
        if 'target_state' in experiment:
            psi_target = self.setup_target_state(experiment['target_state'])
            
        # Evolution
        for i, t in enumerate(times):
            t_ns = t * 1e9
            
            # Find active MW pulse
            active_pulse = None
            for pulse in mw_pulses:
                if pulse['start_ns'] <= t_ns <= pulse['start_ns'] + pulse['duration_ns']:
                    active_pulse = pulse
                    break
            
            # Evolve
            rho = self.evolution_step(rho, t, self.DT_ns*1e-9, active_pulse)
            
            # Store populations
            p_minus1 = float(jnp.real(jnp.trace(jnp.kron(jnp.kron(jnp.diag(jnp.array([1,0,0], dtype=jnp.complex64)), self.Eye_N), self.I_C_tot) @ rho)))
            p0 = float(jnp.real(jnp.trace(self.Sz0_g @ rho)))
            p_plus1 = float(jnp.real(jnp.trace(jnp.kron(jnp.kron(jnp.diag(jnp.array([0,0,1], dtype=jnp.complex64)), self.Eye_N), self.I_C_tot) @ rho)))
            
            results['population_ms_minus1'].append(p_minus1)
            results['population_ms0'].append(p0)
            results['population_ms1'].append(p_plus1)
            
        # Calculate final fidelity if target state is specified
        if psi_target is not None:
            rho_target = jnp.outer(psi_target, psi_target.conj())
            fidelity = float(jnp.real(jnp.trace(rho_target @ rho)))
            results['fidelity'] = fidelity
            
        # Generate photon counts for all counters
        all_photon_counts = []
        
        for i, photon_counter in enumerate(photon_counters):
            if i < len(laser_readouts):  # Make sure we have corresponding laser readout
                counter_start = photon_counter['start_ns']
                counter_duration = photon_counter['duration_ns']
                bin_width = photon_counter['bin_width_ns']
                shots = photon_counter.get('shots', 1000)
                
                # Time bins for photon counting
                count_times = jnp.arange(counter_start, counter_start + counter_duration, bin_width) * 1e-9
                counts = []
                
                for t in count_times:
                    # Find population at this time
                    idx = int(t / self.DT_ns * 1e9)
                    if idx < len(results['population_ms0']):
                        p0 = results['population_ms0'][idx]
                    else:
                        p0 = results['population_ms0'][-1]
                        
                    # Photon count model
                    t_rel = t - counter_start * 1e-9
                    pump = 1 - jnp.exp(-t_rel / 20e-9)
                    W1t = self.W_ms1_late + (1 - self.W_ms1_late) * jnp.exp(-t_rel / 100e-9)
                    rate = self.Beta_max_Hz * pump * (p0 * self.W_ms0_late + (1 - p0) * W1t)
                    
                    # Generate Poisson counts
                    self.key_master, sub = jax.random.split(self.key_master)
                    count = jax.random.poisson(sub, rate * bin_width * 1e-9 * shots)
                    counts.append(float(count))
                    
                photon_data = {
                    'times_ns': count_times * 1e9,
                    'counts': np.array(counts),
                    'bin_width_ns': bin_width,
                    'shots': shots,
                    'measurement_id': i
                }
                all_photon_counts.append(photon_data)
        
        # Store all photon counts
        results['photon_counts'] = all_photon_counts[0] if len(all_photon_counts) == 1 else all_photon_counts
            
        return results
        
    def plot_results(self, results: Dict[str, Any], experiment: Dict[str, Any], save_dir: str = None):
        """Plot experiment results"""
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            
        outputs = experiment.get('outputs', [])
        
        for i, output in enumerate(outputs):
            if output['type'] == 'population_plot':
                self.plot_population(results, experiment, save_path=os.path.join(save_dir, output['filename']) if save_dir else None)
            elif output['type'] == 'photon_trace':
                if results['photon_counts']:
                    # Handle multiple photon counts
                    if isinstance(results['photon_counts'], list):
                        # Use measurement_id to match the correct photon data
                        if i < len(results['photon_counts']):
                            photon_data = results['photon_counts'][i]
                            self.plot_photon_trace_data(photon_data, save_path=os.path.join(save_dir, output['filename']) if save_dir else None)
                    else:
                        # Single photon count
                        self.plot_photon_trace(results, save_path=os.path.join(save_dir, output['filename']) if save_dir else None)
                    
    def plot_population(self, results: Dict[str, Any], experiment: Dict[str, Any], save_path: str = None):
        """Plot population dynamics"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Population plot
        ax1.plot(results['times_ns'], results['population_ms_minus1'], label='P(ms=-1)', color='red')
        ax1.plot(results['times_ns'], results['population_ms0'], label='P(ms=0)', color='blue')
        ax1.plot(results['times_ns'], results['population_ms1'], label='P(ms=+1)', color='green')
        ax1.set_ylabel('Population')
        ax1.set_title('NV Center Population Dynamics')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-0.05, 1.05)
        
        # MW pulse visualization
        sequence = experiment['sequence']
        mw_pulses = [e for e in sequence if e['type'] == 'mw_pulse']
        
        for i, pulse in enumerate(mw_pulses):
            start = pulse['start_ns']
            duration = pulse['duration_ns']
            
            if 'pulse_array' in pulse:
                # Plot pulse array
                times = np.arange(len(pulse['pulse_array']['omega_Hz'])) * self.DT_ns
                ax2.plot(times, np.array(pulse['pulse_array']['omega_Hz'])/1e6, label=f'MW Pulse {i+1}')
            else:
                # Plot envelope
                t_pulse = np.linspace(start, start + duration, 100)
                omega = [self.mw_envelope(t*1e-9, start*1e-9, duration*1e-9, 
                                         pulse.get('omega_rabi_Hz', 5e6), 
                                         pulse.get('shape', 'cos')) for t in t_pulse]
                ax2.plot(t_pulse, np.array(omega)/1e6, label=f'MW Pulse {i+1}')
        
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Rabi Frequency (MHz)')
        ax2.set_title('Microwave Control Pulses')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add fidelity if available
        if results['fidelity'] is not None:
            fig.suptitle(f'Final Fidelity: {results["fidelity"]:.4f}', fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def plot_photon_trace(self, results: Dict[str, Any], save_path: str = None):
        """Plot photon count trace with higher resolution"""
        photon_data = results['photon_counts']
        self.plot_photon_trace_data(photon_data, save_path)
        
    def plot_photon_trace_data(self, photon_data: Dict[str, Any], save_path: str = None):
        """Plot photon count trace data"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Plot with smaller bins for better resolution
        ax.step(photon_data['times_ns'], photon_data['counts'], where='mid', label='Photon counts', linewidth=0.5)
        
        # Add statistics
        mean_counts = np.mean(photon_data['counts'])
        std_counts = np.std(photon_data['counts'])
        ax.axhline(mean_counts, color='red', linestyle='--', alpha=0.5, label=f'Mean: {mean_counts:.1f}')
        ax.fill_between(photon_data['times_ns'], 
                       mean_counts - std_counts, 
                       mean_counts + std_counts, 
                       alpha=0.2, color='red', label=f'±1σ: {std_counts:.1f}')
        
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel('Photon Counts')
        
        # Add measurement ID to title if available
        title = f"Photon Trace (bin width: {photon_data['bin_width_ns']} ns, shots: {photon_data['shots']})"
        if 'measurement_id' in photon_data:
            title += f" - Measurement {photon_data['measurement_id'] + 1}"
        ax.set_title(title)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load experiment configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def load_system_config(system_path: str = None) -> Dict[str, Any]:
    """Load system configuration from system.json"""
    if system_path is None:
        # Look for system.json in the same directory as the experiment file
        system_path = os.path.join(os.path.dirname(config_path), 'system.json')
        if not os.path.exists(system_path):
            # Fall back to test directory
            system_path = 'test/system.json'
    
    with open(system_path, 'r') as f:
        return json.load(f)


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python main.py <experiment.json> [system.json]")
        sys.exit(1)
        
    experiment_path = sys.argv[1]
    system_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(experiment_path):
        print(f"Error: Experiment file '{experiment_path}' not found")
        sys.exit(1)
        
    # Load experiment configuration
    experiment_config = load_config(experiment_path)
    
    # Load system configuration
    if system_path is None:
        # Auto-detect system.json location
        experiment_dir = os.path.dirname(experiment_path)
        system_path = os.path.join(experiment_dir, 'system.json')
        if not os.path.exists(system_path):
            system_path = 'test/system.json'
    
    if not os.path.exists(system_path):
        print(f"Error: System file '{system_path}' not found")
        sys.exit(1)
        
    system_config = load_system_config(system_path)
    
    # Merge configurations
    config = {**system_config, **experiment_config}
    
    # Create simulator
    simulator = NVSimulator(config)
    
    # Run experiment
    experiment = config['experiment']
    results = simulator.run_experiment(experiment)
    
    # Generate outputs
    output_dir = experiment_config.get('output_dir', 'results')
    simulator.plot_results(results, experiment, save_dir=output_dir)
    
    print(f"Experiment completed. Results saved to {output_dir}/")
    if results['fidelity'] is not None:
        print(f"Final fidelity: {results['fidelity']:.4f}")


if __name__ == "__main__":
    main()