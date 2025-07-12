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
from jax.scipy.linalg import expm
from typing import Dict, List, Tuple, Any, Optional

# Import our modular components
from modules.photon_detection import PhotonDetector
from modules.noise_model import NoiseModel
from modules.noise import DecoherenceModel
from modules.c13 import C13Physics
from modules.n14 import N14Physics

# Physical constants
PI = jnp.pi

class NVSimulator:
    """NV center quantum simulator with optimal control capabilities"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize simulator with configuration
        
        Args:
            config: System configuration from JSON
        """
        self.config = config
        
        # Physics modules will be setup based on config
        self.c13_physics = None
        self.n14_physics = None
        self.decoherence_model = None
        
        self.setup_parameters()
        self.setup_operators()
        self.setup_modules()
        
        # Setup physics modules from config (after DIM is set)
        self._setup_physics_modules()
        
        # Build Hamiltonian (after physics modules are setup)
        self.H0 = self.build_static_hamiltonian()
        
        self.setup_initial_state()
    
    def _setup_physics_modules(self):
        """Setup physics modules from configuration"""
        sys_params = self.config['system_parameters']
        
        # Setup C13 physics if C13 cluster is configured
        c13_params = sys_params.get('C13_cluster', {})
        if c13_params.get('N_C', 0) > 0:
            # Get C13 positions from config or use defaults
            c13_positions = c13_params.get('positions_nm', [[0.0, 0.0, 0.154]])  # nm
            c13_positions = np.array(c13_positions) * 1e-9  # Convert to meters
            
            physical_constants = {
                'mu_0': 4*np.pi*1e-7,
                'gamma_e': 2.8e10,  # Hz/T
                'gamma_n_13c': 10.7e6,  # Hz/T
                'hbar': 1.054e-34
            }
            self.c13_physics = C13Physics(c13_positions, physical_constants)
        
        # Setup N14 physics if N14 parameters are present
        n14_params = sys_params.get('N14_hyperfine', {})
        if n14_params:
            physical_constants = {
                'hbar': 1.054e-34,
                'gamma_n_14n': 3.077e6  # Hz/T
            }
            n14_config = {
                'A_para_Hz': n14_params['A_para_Hz'],
                'A_perp_Hz': n14_params['A_perp_Hz'],
                'quadrupole_P_Hz': n14_params.get('quadrupole_P_Hz', 0.0),
                'nuclear_zeeman_factor': n14_params.get('nuclear_zeeman_factor', 1.0)
            }
            self.n14_physics = N14Physics(physical_constants, n14_config)
        
        # Setup advanced decoherence model if configured
        relaxation_params = sys_params.get('relaxation', {})
        if relaxation_params:
            decoherence_config = {
                'T1_s': relaxation_params['T1_s'],
                'Tphi_s': relaxation_params['Tphi_s'],
                'spectral_diffusion_rate_Hz': relaxation_params.get('spectral_diffusion_rate_Hz', 0.0),
                'charge_noise_amplitude_Hz': relaxation_params.get('charge_noise_amplitude_Hz', 0.0),
                'magnetic_noise_amplitude_G': relaxation_params.get('magnetic_noise_amplitude_G', 0.0)
            }
            self.decoherence_model = DecoherenceModel(decoherence_config, self.DIM)
        
    def setup_parameters(self):
        """Extract and set all simulation parameters from config"""
        # System parameters
        sys_params = self.config['system_parameters']
        self.D_g_Hz = sys_params['D_g_Hz']
        self.D_e_Hz = sys_params['D_e_Hz']
        self.gamma_e_Hz_per_G = sys_params['gamma_e_Hz_per_G']
        self.B_vec_G = jnp.array(sys_params['B_vec_G'])
        
        # N14 hyperfine (optional)
        n14_params = sys_params.get('N14_hyperfine', {'A_para_Hz': 0.0, 'A_perp_Hz': 0.0})
        self.A_para_N_Hz = n14_params['A_para_Hz']
        self.A_perp_N_Hz = n14_params['A_perp_Hz']
        
        # C13 cluster
        c13_params = sys_params['C13_cluster']
        self.N_C = c13_params['N_C']
        self.A_para_C_Hz = jnp.array(c13_params['A_para_Hz'])
        self.A_perp_C_Hz = jnp.array(c13_params['A_perp_Hz'])
        
        # Relaxation (optional)
        relax_params = sys_params.get('relaxation', {'T1_s': 1e-3, 'Tphi_s': 1e-6})
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
        
        # Static Hamiltonian will be built after physics modules are setup
        
        
    def kron_list(self, lst):
        """Kronecker product of a list of matrices"""
        out = lst[0]
        for m in lst[1:]:
            out = jnp.kron(out, m)
        return out
        
    def build_static_hamiltonian(self):
        """Build the static part of the Hamiltonian"""
        # Try physics modules first
        if self.c13_physics or self.n14_physics:
            # Build NV operators dict for physics modules
            nv_ops = {
                'Sz': self.Sz_e,
                'Sx': self.Sx_e, 
                'Sy': self.Sy_e,
                'S+': (self.Sx_e + 1j*self.Sy_e),
                'S-': (self.Sx_e - 1j*self.Sy_e)
            }
            
            H_physics = self._build_physics_hamiltonian(nv_ops)
            if H_physics is not None:
                physics_type = "C13" if self.c13_physics else "N14" if self.n14_physics else "physics"
                print(f"Using {physics_type} Hamiltonian (dimension: {H_physics.shape[0]})")
                return H_physics
        
        # Standard Hamiltonian
        H = 2*PI*self.gamma_e_Hz_per_G * (self.B_vec_G[0]*self.Sx + self.B_vec_G[1]*self.Sy + self.B_vec_G[2]*self.Sz)
        H += 2*PI*(self.A_para_N_Hz*(self.Sz@self.Sz_N) + self.A_perp_N_Hz*(self.Sx@self.Sx_N))
        for j in range(self.N_C):
            H += 2*PI*(self.A_para_C_Hz[j]*(self.Sz@self.Iz_list[j]) + self.A_perp_C_Hz[j]*(self.Sx@self.Ix_list[j]))
        
        print(f"Using standard Hamiltonian (dimension: {H.shape[0]})")
        return H
    
    def _build_physics_hamiltonian(self, nv_ops):
        """Build Hamiltonian using physics modules"""
        H_total = None
        
        if self.c13_physics and not self.n14_physics:
            # Only C13 physics
            H_total = self.c13_physics.complete_c13_hamiltonian(nv_ops, self.B_vec_G * 1e-4)  # Convert G to T
            
        elif self.n14_physics and not self.c13_physics:
            # Only N14 physics
            H_total = self.n14_physics.complete_n14_hamiltonian(nv_ops, self.B_vec_G * 1e-4, c13_dimension=1)
            
        elif self.c13_physics and self.n14_physics:
            # Both modules - need proper joint space construction
            # For now, use the one with more nuclei
            c13_nuclei = self.c13_physics.n_c13 if self.c13_physics else 0
            if c13_nuclei > 0:
                H_total = self.c13_physics.complete_c13_hamiltonian(nv_ops, self.B_vec_G * 1e-4)
            else:
                H_total = self.n14_physics.complete_n14_hamiltonian(nv_ops, self.B_vec_G * 1e-4, c13_dimension=1)
        
        return H_total
        
    def setup_initial_state(self):
        """Setup initial quantum state"""
        if self.c13_physics or self.n14_physics:
            # For physics modules, construct state matching H0 dimension
            H0_dim = self.H0.shape[0]
            
            if self.c13_physics and not self.n14_physics:
                # NV + C13 system (no N14)
                ket_e = jnp.array([0,1,0], dtype=jnp.complex64)  # NV ms=0
                ket_C = self.kron_list([jnp.array([1,0], dtype=jnp.complex64) for _ in range(self.N_C)])
                psi0 = jnp.kron(ket_e, ket_C)
                self.rho0 = jnp.outer(psi0, psi0.conj())
                print(f"Using C13 physics initial state (NV+C13, dim={self.rho0.shape[0]})")
                return
            
            # Generic physics module state
            psi0 = jnp.zeros(H0_dim, dtype=jnp.complex64)
            psi0 = psi0.at[0].set(1.0)  # Ground state
            self.rho0 = jnp.outer(psi0, psi0.conj())
            physics_type = "C13" if self.c13_physics else "N14" if self.n14_physics else "physics"
            print(f"Using {physics_type} initial state (dim={self.rho0.shape[0]})")
        else:
            # Standard initial state
            ket_e = jnp.array([0,1,0], dtype=jnp.complex64)  # ms=0
            ket_N = jnp.array([0,1,0], dtype=jnp.complex64)  # mI=0
            if self.N_C > 0:
                ket_C = self.kron_list([jnp.array([1,0], dtype=jnp.complex64) for _ in range(self.N_C)])
                psi0 = jnp.kron(jnp.kron(ket_e, ket_N), ket_C)
            else:
                psi0 = jnp.kron(ket_e, ket_N)
            self.rho0 = jnp.outer(psi0, psi0.conj())
            print("Using standard initial state (ground state)")
        
    def setup_target_state(self, target_config: Dict[str, Any]) -> jnp.ndarray:
        """Setup target quantum state for optimal control"""
        ket_e = jnp.array(target_config['electron'], dtype=jnp.complex64)
        ket_N = jnp.array(target_config['nitrogen'], dtype=jnp.complex64)
        ket_C = self.kron_list([jnp.array(c, dtype=jnp.complex64) for c in target_config['carbons']])
        psi_target = jnp.kron(jnp.kron(ket_e, ket_N), ket_C)
        return psi_target
        
    def setup_modules(self):
        """Setup noise and photon detection modules"""
        # Setup noise model
        relaxation_params = self.config['system_parameters'].get('relaxation', {'T1_s': self.T1_s, 'Tphi_s': self.Tphi_s})
        self.noise_model = NoiseModel(relaxation_params, self.DIM)
        
        # Setup photon detector
        optical_params = self.config['system_parameters']['optical']
        readout_params = self.config['system_parameters']['readout']
        self.photon_detector = PhotonDetector(optical_params, readout_params)
        
    def dissipator(self, rho, t=0.0):
        """Apply dissipator for relaxation and dephasing"""
        # Check dimension compatibility
        if rho.shape[0] != self.Sz.shape[0]:
            # Physics modules case - use advanced decoherence model
            if self.decoherence_model is not None:
                # Create operators matching rho dimension
                dim = rho.shape[0]
                if dim == 6 and self.c13_physics:  # NV + C13 case
                    # NV Sz embedded in 6D space (NV ⊗ C13)
                    Sz_nv = jnp.diag(jnp.array([-1, 0, 1], dtype=jnp.complex64))
                    operators = {
                        'Sz': jnp.kron(Sz_nv, jnp.eye(2)),
                        'Pg': jnp.eye(dim, dtype=jnp.complex64)
                    }
                else:
                    # Generic case
                    operators = {
                        'Sz': jnp.eye(dim, dtype=jnp.complex64),
                        'Pg': jnp.eye(dim, dtype=jnp.complex64)
                    }
                return self.decoherence_model.basic_dissipator(rho, operators)
            else:
                # Simple dephasing proportional to system size
                gamma_eff = 1.0 / self.Tphi_s if hasattr(self, 'Tphi_s') else 1e-6
                return -gamma_eff * (rho - jnp.trace(rho) * jnp.eye(rho.shape[0]) / rho.shape[0])
        else:
                return jnp.zeros_like(rho)
        
        # Standard dissipator
        operators = {
            'Sz': self.Sz,
            'Pg': self.Pg,
            'Sx': self.Sx,
            'Sy': self.Sy
        }
        
        return self.noise_model.basic_dissipator(rho, operators)
    
    def _build_physics_mw_hamiltonian(self, omega, phi, delta):
        """Build MW Hamiltonian in physics module space"""
        H_dim = self.H0.shape[0]
        
        if self.c13_physics and not self.n14_physics:
            # NV + C13 space: MW acts only on NV (first 3 levels)
            nv_dim = 3
            c13_dim = H_dim // nv_dim
            
            # NV operators in physics space
            Sx_nv = jnp.array([[0,1,0],[1,0,1],[0,1,0]], dtype=jnp.complex64) / jnp.sqrt(2)
            Sy_nv = jnp.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=jnp.complex64) / jnp.sqrt(2) 
            Sz_nv = jnp.diag(jnp.array([-1,0,1], dtype=jnp.complex64))
            
            # Embed in joint space
            Sx_joint = jnp.kron(Sx_nv, jnp.eye(c13_dim))
            Sy_joint = jnp.kron(Sy_nv, jnp.eye(c13_dim))
            Sz_joint = jnp.kron(Sz_nv, jnp.eye(c13_dim))
            
            # Phase-controlled MW Hamiltonian
            Hx = jnp.cos(phi) * Sx_joint + jnp.sin(phi) * Sy_joint
            H_mw = 2*PI*(omega * Hx + delta * Sz_joint)
            
            return H_mw
        else:
            # Generic physics case - apply to first few levels
            H_mw = jnp.zeros((H_dim, H_dim), dtype=jnp.complex64)
            
            # Simple MW on first 3 states (approximation)
            if H_dim >= 3:
                Sx_sub = jnp.array([[0,1,0],[1,0,1],[0,1,0]], dtype=jnp.complex64) / jnp.sqrt(2)
                Sy_sub = jnp.array([[0,-1j,0],[1j,0,-1j],[0,1j,0]], dtype=jnp.complex64) / jnp.sqrt(2)
                Sz_sub = jnp.diag(jnp.array([-1,0,1], dtype=jnp.complex64))
                
                Hx_sub = jnp.cos(phi) * Sx_sub + jnp.sin(phi) * Sy_sub
                H_mw_sub = 2*PI*(omega * Hx_sub + delta * Sz_sub)
                
                H_mw = H_mw.at[:3, :3].set(H_mw_sub)
            
            return H_mw
        
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
  # Fallback to standard evolution
        
        # Standard evolution
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
            if (self.c13_physics or self.n14_physics) and self.H0.shape[0] != self.Sz.shape[0]:
                # Physics modules case - build MW operators in physics space
                H_mw = self._build_physics_mw_hamiltonian(omega, phi, delta)
                H = self.H0 + H_mw
                if omega > 0:  # Debug output
                    print(f"🔧 Physics MW: Ω={omega/1e6:.1f}MHz, φ={phi:.2f}rad, H_mw_max={jnp.max(jnp.abs(H_mw))/2/jnp.pi/1e6:.1f}MHz")
            else:
                # Standard case
                Hx = jnp.cos(phi) * self.Sx + jnp.sin(phi) * self.Sy
                H = self.H0 + 2*PI*(omega * Hx + delta * self.Sz)
        else:
            H = self.H0
            
        # Evolution
        U = expm(-1j*H*dt/2)
        rho = U @ rho @ U.conj().T
        rho = rho + self.dissipator(rho, t) * dt
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
            if (self.c13_physics or self.n14_physics) and rho.shape[0] != self.DIM:
                # Physics modules case - simplified population calculation
                # Assume first few states correspond to NV ms levels
                if rho.shape[0] >= 3:
                    p_minus1 = float(jnp.real(rho[0, 0]))  # Approximate
                    p0 = float(jnp.real(rho[1, 1])) if rho.shape[0] > 1 else 0.0
                    p_plus1 = float(jnp.real(rho[2, 2])) if rho.shape[0] > 2 else 0.0
                else:
                    p_minus1 = 0.0
                    p0 = float(jnp.real(jnp.trace(rho)))
                    p_plus1 = 0.0
            else:
                # Standard population calculation
                p_minus1 = float(jnp.real(jnp.trace(jnp.kron(jnp.kron(jnp.diag(jnp.array([1,0,0], dtype=jnp.complex64)), self.Eye_N), self.I_C_tot) @ rho)))
                p0 = float(jnp.real(jnp.trace(self.Sz0_g @ rho)))
                p_plus1 = float(jnp.real(jnp.trace(jnp.kron(jnp.kron(jnp.diag(jnp.array([0,0,1], dtype=jnp.complex64)), self.Eye_N), self.I_C_tot) @ rho)))
            
            results['population_ms_minus1'].append(p_minus1)
            results['population_ms0'].append(p0)
            results['population_ms1'].append(p_plus1)
            
        # Calculate final fidelity if target state is specified
        if psi_target is not None:
            rho_target = jnp.outer(psi_target, psi_target.conj())
            
            # Check dimension compatibility
            if rho_target.shape == rho.shape:
                fidelity = float(jnp.real(jnp.trace(rho_target @ rho)))
            else:
                # Enhanced physics case - simplified fidelity
                fidelity = 0.0  # Cannot calculate fidelity with mismatched dimensions
            
            results['fidelity'] = fidelity
            
        # Generate photon counts for all counters
        all_photon_counts = []
        
        for i, photon_counter in enumerate(photon_counters):
            if i < len(laser_readouts):  # Make sure we have corresponding laser readout
                # Use photon detector module
                populations = {
                    'times_ns': results['times_ns'],
                    'ms0': results['population_ms0']
                }
                
                photon_data = self.photon_detector.generate_counts(
                    populations,
                    photon_counter,
                    self.DT_ns * 1e-9,
                    self.key_master
                )
                
                # Update RNG key
                self.key_master, _ = jax.random.split(self.key_master)
                
                # Add measurement ID
                photon_data['measurement_id'] = i
                all_photon_counts.append(photon_data)
        
        # Store all photon counts
        results['photon_counts'] = all_photon_counts[0] if len(all_photon_counts) == 1 else all_photon_counts
        
        # Export photon count data to files  
        self.export_photon_data(all_photon_counts, experiment, save_dir='results')
            
        return results
            
    def export_photon_data(self, all_photon_counts: List[Dict[str, Any]], experiment: Dict[str, Any], save_dir: str = 'results'):
        """Export photon count data to data files"""
        import datetime
        
        # Create data directory
        data_dir = save_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Generate timestamp for unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = experiment.get('name', 'experiment').replace(' ', '_')
        
        for i, photon_data in enumerate(all_photon_counts):
            # Create filename
            filename = f"{experiment_name}_measurement_{i+1}_{timestamp}.dat"
            filepath = os.path.join(data_dir, filename)
            
            # Prepare header information
            header_lines = []
            header_lines.append(f"# NV-CENTER PHOTON COUNT DATA")
            header_lines.append(f"# Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            header_lines.append(f"# Experiment: {experiment.get('name', 'Unknown')}")
            header_lines.append(f"# Description: {experiment.get('description', 'No description')}")
            header_lines.append(f"# Measurement ID: {i+1}")
            header_lines.append(f"# Total time: {experiment.get('total_time_ns', 'Unknown')} ns")
            header_lines.append(f"#")
            header_lines.append(f"# MEASUREMENT PARAMETERS:")
            header_lines.append(f"# - Bin width: {photon_data['bin_width_ns']} ns")
            header_lines.append(f"# - Total shots: {photon_data['shots']}")
            header_lines.append(f"# - Start time: {photon_data['times_ns'][0]:.0f} ns")
            header_lines.append(f"# - End time: {photon_data['times_ns'][-1]:.0f} ns")
            header_lines.append(f"# - Total bins: {len(photon_data['counts'])}")
            header_lines.append(f"# - Mean counts: {np.mean(photon_data['counts']):.2f}")
            header_lines.append(f"# - Std counts: {np.std(photon_data['counts']):.2f}")
            header_lines.append(f"# - Min counts: {np.min(photon_data['counts']):.0f}")
            header_lines.append(f"# - Max counts: {np.max(photon_data['counts']):.0f}")
            header_lines.append(f"#")
            header_lines.append(f"# SYSTEM PARAMETERS:")
            header_lines.append(f"# - Magnetic field: {self.B_vec_G.tolist()} G")
            header_lines.append(f"# - C13 nuclei: {self.N_C}")
            header_lines.append(f"# - T1: {self.T1_s:.2e} s")
            header_lines.append(f"# - Tphi: {self.Tphi_s:.2e} s")
            header_lines.append(f"# - Beta_max: {self.Beta_max_Hz:.2e} Hz")
            header_lines.append(f"# - Random seed: {self.config['simulation_parameters'].get('seed', 'Unknown')}")
            header_lines.append(f"#")
            header_lines.append(f"# PULSE SEQUENCE:")
            
            # Add pulse sequence information
            mw_pulses = [e for e in experiment['sequence'] if e['type'] == 'mw_pulse']
            for j, pulse in enumerate(mw_pulses):
                header_lines.append(f"# - Pulse {j+1}: {pulse['start_ns']}-{pulse['start_ns'] + pulse['duration_ns']} ns, "
                                  f"Ω={pulse.get('omega_rabi_Hz', 0)/1e6:.1f} MHz, "
                                  f"φ={pulse.get('phase_rad', 0):.3f} rad, "
                                  f"Δ={pulse.get('delta_Hz', 0)/1e6:.1f} MHz")
            
            header_lines.append(f"#")
            header_lines.append(f"# DATA FORMAT:")
            header_lines.append(f"# Each line contains all photon counts from one counter")
            header_lines.append(f"# Time(ns) -> Bin_center = bin_index * {photon_data['bin_width_ns']} + {photon_data['times_ns'][0]:.0f}")
            header_lines.append(f"# Counts -> Poisson-distributed photon counts per bin")
            header_lines.append(f"#")
            header_lines.append(f"# MEASUREMENT_ID  COUNT  COUNT  COUNT  COUNT  COUNT  COUNT  COUNT  COUNT  COUNT  COUNT...")
            
            # Write data file
            with open(filepath, 'w') as f:
                # Write header
                for line in header_lines:
                    f.write(line + '\n')
                
                # Write all counts from this counter in one line
                counts = photon_data['counts']
                
                # Format: measurement_id followed by all counts
                line_data = [f"{photon_data['measurement_id']:8d}"]
                for count in counts:
                    line_data.append(f"{count:6.0f}")
                
                f.write("  ".join(line_data) + '\n')
                
            print(f"📊 Exported photon data: {filepath}")
            print(f"   📈 {len(counts)} bins, {photon_data['shots']} shots per bin")
            print(f"   📊 Mean: {np.mean(counts):.1f} ± {np.std(counts):.1f} counts")
    
    def _validate_physics_modules(self):
        """Validate physics modules setup"""
        modules = []
        if self.c13_physics: modules.append("C13")
        if self.n14_physics: modules.append("N14")
        if self.decoherence_model: modules.append("noise")
        physics_type = "+".join(modules) if modules else "standard"
        print(f"🔍 Validating {physics_type} physics...")
        
        if self.c13_physics and hasattr(self.c13_physics, 'validate_operators'):
            if not self.c13_physics.validate_operators():
                raise ValueError("C13 physics validation failed!")
            print("✅ C13 physics validated")
        
        if self.n14_physics and hasattr(self.n14_physics, 'validate_operators'):
            if not self.n14_physics.validate_operators():
                raise ValueError("N14 physics validation failed!")
            print("✅ N14 physics validated")
        print(f"✅ Enhanced Noise: {physics_info.get('enhanced_noise_active', False)}")
    
    def get_physics_info(self) -> Dict[str, Any]:
        """Get information about current physics setup"""
        modules = []
        if self.c13_physics: modules.append("C13")
        if self.n14_physics: modules.append("N14")
        if self.decoherence_model: modules.append("noise")
        physics_mode = "+".join(modules) if modules else "standard"
        
        base_info = {
            'physics_mode': physics_mode,
            'physics_modules_active': bool(modules),
            'hamiltonian_dimension': self.H0.shape[0],
            'system_dimension': self.DIM
        }
        
        if self.c13_physics:
            base_info['c13_physics_info'] = self.c13_physics.get_info()
        
        if self.n14_physics:
            base_info['n14_physics_info'] = self.n14_physics.get_info()
        
        if self.decoherence_model:
            base_info['decoherence_parameters'] = self.decoherence_model.get_parameters()
        
        return base_info


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
            system_path = 'system.json'
    
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
            system_path = 'system.json'
    
    if not os.path.exists(system_path):
        print(f"Error: System file '{system_path}' not found")
        sys.exit(1)
        
    system_config = load_system_config(system_path)
    
    # Merge configurations
    config = {**system_config, **experiment_config}
    
    # Create simulator (physics configured via system.json)
    print("🔧 Initializing NV Simulator...")
    simulator = NVSimulator(config)
    
    # Print physics info
    physics_info = simulator.get_physics_info()
    print(f"📊 Physics Info:")
    print(f"   Mode: {physics_info['physics_mode']}")
    print(f"   Physics modules: {physics_info['physics_modules_active']}")
    print(f"   Hamiltonian dimension: {physics_info['hamiltonian_dimension']}")
    if 'c13_physics_info' in physics_info:
        c13_info = physics_info['c13_physics_info']
        print(f"   C13 nuclei: {c13_info['n_c13']}")
    if 'n14_physics_info' in physics_info:
        n14_info = physics_info['n14_physics_info']
        print(f"   N14 parameters: A_para={n14_info['A_para_Hz']/1e6:.1f}MHz")
    
    # Run experiment
    experiment = config['experiment']
    results = simulator.run_experiment(experiment)
    
    print(f"✅ Experiment completed. Raw data files saved to results/")
    if results['fidelity'] is not None:
        print(f"📊 Final fidelity: {results['fidelity']:.4f}")


if __name__ == "__main__":
    main()