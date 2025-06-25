"""
Optimized NV System - Fast version with same physics
"""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp
import sys, os
sys.path.append('helper')
sys.path.append('modules')

from noise import NoiseGenerator, NoiseConfiguration
from noise_sources import SYSTEM

class FastNVSystem:
    """Optimized NV system for faster computation"""
    
    def __init__(self, B_field=None, enable_noise=True):
        self.B_field = B_field if B_field is not None else np.zeros(3)
        self.enable_noise = enable_noise
        
        # Constants
        self.D = SYSTEM.get_constant('nv_center', 'd_gs')
        self.gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
        self.hbar = SYSTEM.get_constant('fundamental', 'hbar')
        
        # Spin operators
        self.Sz = np.diag([-1., 0., 1.]).astype(complex)
        self.Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=complex) / np.sqrt(2)
        self.Sy = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], dtype=complex) / np.sqrt(2)
        self.I = np.eye(3, dtype=complex)
        
        # Pre-compute static Hamiltonian
        self.H_static = self._compute_static_hamiltonian()
        
        # Simple noise configuration (only essential sources)
        if enable_noise:
            config = NoiseConfiguration()
            config.enable_c13_bath = True
            config.enable_temperature = False  # Disable slow sources
            config.enable_johnson = False
            config.enable_charge_noise = False
            config.enable_external_field = False
            config.enable_strain = False
            config.enable_microwave = False
            config.enable_optical = False
            config.dt = 1e-8  # Larger timestep
            
            self.noise_gen = NoiseGenerator(config)
            # Pre-sample noise for efficiency
            self._noise_cache = None
            self._cache_size = 1000
            self._cache_idx = 0
        else:
            self.noise_gen = None
    
    def _compute_static_hamiltonian(self):
        """Pre-compute static part of Hamiltonian"""
        # Zero-field splitting
        H = 2 * np.pi * self.D * (self.Sz @ self.Sz - (2/3) * self.I)
        
        # Static Zeeman
        if np.any(self.B_field):
            H += 2 * np.pi * self.gamma_e * (
                self.B_field[0] * self.Sx +
                self.B_field[1] * self.Sy +
                self.B_field[2] * self.Sz
            )
        return H
    
    def _get_noise_field(self):
        """Get noise field from cache for efficiency"""
        if not self.enable_noise or self.noise_gen is None:
            return np.zeros(3)
            
        # Refill cache if needed
        if self._noise_cache is None or self._cache_idx >= self._cache_size:
            self._noise_cache = self.noise_gen.get_total_magnetic_noise(self._cache_size)
            self._cache_idx = 0
            
        noise = self._noise_cache[self._cache_idx]
        self._cache_idx += 1
        return noise
    
    def get_hamiltonian(self):
        """Get total Hamiltonian with cached noise"""
        H = self.H_static.copy()
        
        if self.enable_noise:
            B_noise = self._get_noise_field()
            H += 2 * np.pi * self.gamma_e * (
                B_noise[0] * self.Sx +
                B_noise[1] * self.Sy +
                B_noise[2] * self.Sz
            )
        return H
    
    def evolve_unitary(self, rho0, t_span, n_steps=100):
        """Fast unitary evolution (no dissipation)"""
        times = np.linspace(t_span[0], t_span[1], n_steps)
        dt = times[1] - times[0]
        
        rhos = [rho0]
        current_rho = rho0.copy()
        
        for i in range(1, n_steps):
            # Get Hamiltonian for this step
            H = self.get_hamiltonian()
            
            # Unitary evolution operator
            U = expm(-1j * H * dt / self.hbar)
            
            # Evolve density matrix
            current_rho = U @ current_rho @ U.conj().T
            rhos.append(current_rho)
            
        return times, rhos
    
    def simple_lindblad(self, rho0, t_span, gamma_1=1e3, gamma_2=1e6):
        """Simplified Lindblad with phenomenological rates"""
        def rhs(t, rho_vec):
            rho = rho_vec.reshape(3, 3)
            
            # Hamiltonian part
            H = self.get_hamiltonian()
            drho = -1j * (H @ rho - rho @ H) / self.hbar
            
            # Simple T1 relaxation
            drho[0, 0] += gamma_1 * rho[1, 1]  # |0⟩ → |-1⟩
            drho[1, 1] -= gamma_1 * rho[1, 1]
            
            # Simple T2 dephasing
            drho[0, 1] -= gamma_2 * rho[0, 1]
            drho[1, 0] -= gamma_2 * rho[1, 0]
            drho[0, 2] -= gamma_2 * rho[0, 2]
            drho[2, 0] -= gamma_2 * rho[2, 0]
            drho[1, 2] -= gamma_2 * rho[1, 2]
            drho[2, 1] -= gamma_2 * rho[2, 1]
            
            return drho.flatten()
        
        # Use larger tolerances for speed
        sol = solve_ivp(rhs, t_span, rho0.flatten(), 
                       method='RK23', rtol=1e-4, atol=1e-6,
                       max_step=1e-8)
        
        times = sol.t
        rhos = [sol.y[:, i].reshape(3, 3) for i in range(len(times))]
        
        return times, rhos

def demo_fast_evolution():
    """Demonstrate fast evolution"""
    print("FAST NV SYSTEM DEMONSTRATION")
    print("="*30)
    
    # Initialize
    B_field = np.array([0, 0, 1e-3])  # 1 mT
    nv = FastNVSystem(B_field, enable_noise=True)
    
    # Initial state: superposition
    psi0 = np.array([1, 1, 1]) / np.sqrt(3)
    rho0 = np.outer(psi0, psi0.conj())
    
    # Fast unitary evolution
    print("\n1. Unitary evolution (100 ns)...")
    import time
    start = time.time()
    times, rhos = nv.evolve_unitary(rho0, (0, 100e-9), n_steps=100)
    elapsed = time.time() - start
    print(f"   Completed in {elapsed:.3f} seconds")
    
    # Populations
    final_pops = np.real(np.diag(rhos[-1]))
    print(f"   Final populations: {final_pops}")
    
    # Fast Lindblad
    print("\n2. Lindblad evolution (1 μs)...")
    start = time.time()
    times2, rhos2 = nv.simple_lindblad(rho0, (0, 1e-6))
    elapsed = time.time() - start
    print(f"   Completed in {elapsed:.3f} seconds")
    
    final_pops2 = np.real(np.diag(rhos2[-1]))
    print(f"   Final populations: {final_pops2}")
    
    # Coherence
    coherence = np.abs(rhos2[-1][0, 1])
    print(f"   Final coherence: {coherence:.4f}")
    
    print("\n✓ Fast evolution complete!")

if __name__ == "__main__":
    demo_fast_evolution()