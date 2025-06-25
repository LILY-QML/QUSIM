"""
NV Center System with Full Noise Integration and Lindblad Evolution

This module implements a complete nitrogen-vacancy (NV) center quantum system simulation
including comprehensive noise modeling, open quantum system dynamics, and pulse control.

The module provides:
    - Complete spin-1 operator algebra for NV centers
    - Hamiltonian construction with zero-field splitting and Zeeman effects
    - Integration of multiple noise sources (C13 bath, charge noise, thermal effects, etc.)
    - Lindblad master equation evolution for open system dynamics
    - Microwave pulse sequence control
    - Common experimental protocols (Ramsey, echo sequences)
    - Visualization and analysis tools

Key Features:
    - Physically accurate modeling based on experimental parameters
    - Flexible noise configuration for different experimental conditions
    - Support for arbitrary pulse sequences
    - Efficient numerical integration with adaptive timesteps
    - Built-in measurement protocols for T2*, T1 characterization

Examples:
    Basic usage with default room temperature noise:
        >>> nv = create_room_temperature_nv()
        >>> rho0 = np.outer(nv.states['ms0'], nv.states['ms0'].conj())
        >>> times, rhos = nv.evolve(rho0, (0, 1e-6))
        
    Custom noise configuration:
        >>> config = NoiseConfiguration()
        >>> config.enable_c13_bath = True
        >>> config.enable_charge_noise = False
        >>> nv = NVSystem(B_field=[0, 0, 0.01], noise_config=config)
        
    Ramsey sequence measurement:
        >>> tau_values, coherences = nv.simulate_t2_measurement()
        
Author: QUSIM Development Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from typing import Dict, List, Optional, Tuple, Union, Callable
import sys
import os
import warnings

# Add helper path for noise sources
sys.path.append(os.path.join(os.path.dirname(__file__), 'helper'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

from noise import NoiseGenerator, NoiseConfiguration
from noise_sources import SYSTEM


class NVSpinOperators:
    """
    Spin operators for NV center electronic ground state (S=1 system).
    
    This class implements the complete set of spin-1 operators for the nitrogen-vacancy
    center's electronic ground state triplet. The operators are represented in the 
    standard |ms=-1,0,+1⟩ basis, where ms is the spin projection along the NV axis.
    
    The spin operators satisfy the standard angular momentum commutation relations:
        [Si, Sj] = i·εijk·Sk
        
    where εijk is the Levi-Civita symbol.
    
    Attributes:
        dim (int): Dimension of the Hilbert space (3 for spin-1)
        Sx (np.ndarray): Spin-x operator matrix (3x3 complex)
        Sy (np.ndarray): Spin-y operator matrix (3x3 complex)
        Sz (np.ndarray): Spin-z operator matrix (3x3 complex)
        S_plus (np.ndarray): Raising operator S+ = Sx + i·Sy
        S_minus (np.ndarray): Lowering operator S- = Sx - i·Sy
        Sx2 (np.ndarray): Squared spin-x operator Sx²
        Sy2 (np.ndarray): Squared spin-y operator Sy²
        Sz2 (np.ndarray): Squared spin-z operator Sz²
        I (np.ndarray): Identity operator (3x3)
        operators (dict): Dictionary mapping operator names to matrices
        
    Notes:
        - All operators are Hermitian (except S+ and S-)
        - The operators form a complete basis for observables
        - Matrix elements follow Condon-Shortley phase conventions
        
    Examples:
        >>> ops = NVSpinOperators()
        >>> # Verify commutation relation [Sx, Sy] = i·Sz
        >>> commutator = ops.Sx @ ops.Sy - ops.Sy @ ops.Sx
        >>> np.allclose(commutator, 1j * ops.Sz)
        True
        >>> # Check total spin S² = 2 for spin-1
        >>> S_squared = ops.Sx2 + ops.Sy2 + ops.Sz2
        >>> np.allclose(S_squared, 2 * ops.I)
        True
    """
    
    def __init__(self):
        """
        Initialize spin-1 operators for the NV center ground state.
        
        Constructs all spin operators in the |ms=-1,0,+1⟩ basis following
        standard angular momentum algebra conventions. The basis states are
        ordered as |ms=-1⟩, |ms=0⟩, |ms=+1⟩.
        
        The operator matrix elements are computed using:
            <ms'|S±|ms> = √[S(S+1) - ms(ms±1)]·δ(ms',ms±1)
            <ms'|Sz|ms> = ms·δ(ms',ms)
            
        where S=1 for the NV ground state.
        """
        # Dimension of spin-1 Hilbert space
        self.dim = 3
        
        # Standard spin-1 operators in |ms=-1,0,+1⟩ basis
        # These satisfy [Si, Sj] = i*εijk*Sk with proper normalization
        
        # Construct from S± operators first for consistency
        S_plus = np.array([
            [0, np.sqrt(2), 0],
            [0, 0, np.sqrt(2)],
            [0, 0, 0]
        ], dtype=complex)
        
        S_minus = np.array([
            [0, 0, 0],
            [np.sqrt(2), 0, 0],
            [0, np.sqrt(2), 0]
        ], dtype=complex)
        
        # Sx = (S+ + S-)/2, Sy = (S+ - S-)/(2i)
        self.Sx = (S_plus + S_minus) / 2
        self.Sy = (S_plus - S_minus) / (2j)
        
        self.Sz = np.array([
            [-1, 0, 0],
            [0, 0, 0],
            [0, 0, 1]
        ], dtype=complex)
        
        # Raising and lowering operators
        self.S_plus = self.Sx + 1j * self.Sy
        self.S_minus = self.Sx - 1j * self.Sy
        
        # Squared operators (useful for strain terms)
        self.Sx2 = self.Sx @ self.Sx
        self.Sy2 = self.Sy @ self.Sy
        self.Sz2 = self.Sz @ self.Sz
        
        # Identity operator
        self.I = np.eye(3, dtype=complex)
        
        # Dictionary for convenient access
        self.operators = {
            'Sx': self.Sx, 'Sy': self.Sy, 'Sz': self.Sz,
            'S+': self.S_plus, 'S-': self.S_minus,
            'Sx2': self.Sx2, 'Sy2': self.Sy2, 'Sz2': self.Sz2,
            'I': self.I
        }
        
    def verify_algebra(self) -> bool:
        """
        Verify that operators satisfy correct algebraic relations.
        
        Checks:
            - Hermiticity of spin operators
            - Commutation relations
            - Total spin S² = S(S+1) = 2
            - Trace properties
            
        Returns:
            bool: True if all checks pass
            
        Raises:
            AssertionError: If any algebraic relation is violated
        """
        # Check Hermiticity
        for op_name in ['Sx', 'Sy', 'Sz']:
            op = self.operators[op_name]
            assert np.allclose(op, op.conj().T), f"{op_name} not Hermitian"
            
        # Check commutation relations
        comm_xy = self.Sx @ self.Sy - self.Sy @ self.Sx
        assert np.allclose(comm_xy, 1j * self.Sz), "[Sx,Sy] ≠ i·Sz"
        
        # Check S² = 2
        S_squared = self.Sx2 + self.Sy2 + self.Sz2
        assert np.allclose(S_squared, 2 * self.I), "S² ≠ 2"
        
        return True


class NVSystemHamiltonian:
    """
    NV center Hamiltonian with comprehensive noise integration.
    
    This class constructs and manages the full Hamiltonian for an NV center including
    all relevant physical interactions and noise sources. The Hamiltonian consists of:
    
    1. Zero-field splitting (ZFS) from spin-spin interaction
    2. Zeeman interaction with external magnetic fields  
    3. Strain-induced splitting
    4. Time-dependent noise from various environmental sources
    
    The general form of the Hamiltonian is:
        H = H_ZFS + H_Zeeman + H_strain + H_noise(t)
        
    where:
        H_ZFS = D·(Sz² - 2/3·I) + E·(Sx² - Sy²)
        H_Zeeman = γe·B·S
        H_noise(t) = Σ_i noise_i(t)·O_i
        
    with D ≈ 2.87 GHz the axial ZFS parameter, E the transverse strain parameter,
    γe ≈ 28 GHz/T the electron gyromagnetic ratio, and O_i appropriate operators
    for each noise source.
    
    Attributes:
        spin_ops (NVSpinOperators): Complete set of spin operators
        B_static (np.ndarray): Static magnetic field vector [Bx, By, Bz] in Tesla
        noise_gen (NoiseGenerator): Generator for time-dependent noise fields
        D (float): Zero-field splitting parameter in Hz (typically 2.87 GHz)
        E (float): Strain-induced splitting parameter in Hz (typically < 10 MHz)
        gamma_e (float): Electron gyromagnetic ratio in Hz/T (28.024 GHz/T)
        
    Parameters:
        B_field: Static magnetic field components in Tesla
        noise_gen: Noise generator instance for environmental effects
        
    Examples:
        >>> # NV in 10 mT field along z-axis
        >>> B = np.array([0, 0, 0.01])
        >>> ham = NVSystemHamiltonian(B_field=B)
        >>> H_static = ham.get_static_hamiltonian()
        >>> eigenvalues = np.linalg.eigvalsh(H_static) / (2*np.pi*1e9)  # in GHz
        >>> print(f"Energy levels: {eigenvalues}")
        
        >>> # Add noise effects
        >>> noise_gen = NoiseGenerator(NoiseConfiguration())
        >>> ham_noisy = NVSystemHamiltonian(B_field=B, noise_gen=noise_gen)
        >>> H_total = ham_noisy.get_total_hamiltonian()
    """
    
    def __init__(self, B_field: np.ndarray = None, noise_gen: NoiseGenerator = None):
        """
        Initialize NV system Hamiltonian with field and noise configuration.
        
        Creates the Hamiltonian manager with specified static magnetic field and
        noise generator. All physical parameters are loaded from the system
        configuration file.
        
        Args:
            B_field: Static magnetic field vector [Bx, By, Bz] in Tesla.
                    Must be a 3-element array-like object.
                    If None, zero field is assumed.
            noise_gen: NoiseGenerator instance for time-dependent effects.
                      If None, no noise is included (ideal system).
                      
        Raises:
            ValueError: If B_field has incorrect shape or invalid values.
            TypeError: If noise_gen is not a NoiseGenerator instance.
            
        Notes:
            - Field values should be in Tesla (not Gauss)
            - Typical lab fields are 0.001-1 T
            - The NV z-axis is along the [111] crystal direction
        """
        # Initialize spin operators
        self.spin_ops = NVSpinOperators()
        
        # Set magnetic field
        if B_field is not None:
            B_field = np.asarray(B_field)
            if B_field.shape != (3,):
                raise ValueError(f"B_field must be 3D vector, got shape {B_field.shape}")
            self.B_static = B_field
        else:
            self.B_static = np.zeros(3)
            
        # Set noise generator
        if noise_gen is not None and not isinstance(noise_gen, NoiseGenerator):
            raise TypeError("noise_gen must be a NoiseGenerator instance")
        self.noise_gen = noise_gen
        
        # Load NV center parameters from system configuration
        self.D = SYSTEM.get_constant('nv_center', 'd_gs')  # ZFS parameter [Hz]
        self.E = SYSTEM.get_constant('nv_center', 'e_gs')  # Strain parameter [Hz]
        self.gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')  # Gyromagnetic ratio [Hz/T]
        
    def get_static_hamiltonian(self) -> np.ndarray:
        """
        Construct the static (time-independent) part of the Hamiltonian.
        
        Builds the static Hamiltonian including zero-field splitting,
        strain effects, and static Zeeman interaction. This represents
        the "bare" NV Hamiltonian without environmental noise.
        
        The static Hamiltonian is:
            H_static = 2π·D·(Sz² - 2/3·I) + 2π·E·(Sx² - Sy²) + 2π·γe·B·S
            
        where the factor of 2π converts from Hz to angular frequency units.
        
        Returns:
            np.ndarray: 3x3 complex Hamiltonian matrix in units of rad/s.
                       Eigenvalues give energies E_i such that |ψ(t)⟩ = e^(-iE_i·t/ℏ)|ψ_i⟩
            
        Notes:
            - The Hamiltonian is Hermitian by construction
            - In zero field, eigenstates are |0⟩ and (|+1⟩±|-1⟩)/√2
            - Energy scale is dominated by D ≈ 2.87 GHz
            - Strain term E breaks the |±1⟩ degeneracy
            
        Examples:
            >>> ham = NVSystemHamiltonian()
            >>> H = ham.get_static_hamiltonian()
            >>> # Check Hermiticity
            >>> np.allclose(H, H.conj().T)
            True
            >>> # Get transition frequencies
            >>> E = np.linalg.eigvalsh(H)
            >>> freq_01 = (E[1] - E[0]) / (2*np.pi*1e9)  # in GHz
        """
        H_static = np.zeros((3, 3), dtype=complex)
        
        # Zero-field splitting: D(Sz² - 2/3·I)
        # This is the dominant term, splitting |0⟩ from |±1⟩ by ~2.87 GHz
        H_static += 2 * np.pi * self.D * (self.spin_ops.Sz2 - (2/3) * self.spin_ops.I)
        
        # Strain term: E(Sx² - Sy²)
        # This breaks the degeneracy between |+1⟩ and |-1⟩ states
        if self.E != 0:
            H_static += 2 * np.pi * self.E * (self.spin_ops.Sx2 - self.spin_ops.Sy2)
        
        # Static Zeeman effect: γe·B·S
        # Linear in field strength, typically small compared to D
        if np.any(self.B_static != 0):
            H_static += 2 * np.pi * self.gamma_e * (
                self.B_static[0] * self.spin_ops.Sx +
                self.B_static[1] * self.spin_ops.Sy +
                self.B_static[2] * self.spin_ops.Sz
            )
            
        return H_static
    
    def get_noise_hamiltonian(self, t: float = 0.0, 
                            include_sources: Optional[List[str]] = None) -> np.ndarray:
        """
        Get the noise contribution to the Hamiltonian at time t.
        
        Computes the time-dependent part of the Hamiltonian arising from
        environmental noise sources. Each noise source contributes terms
        of the form noise_field(t)·operator.
        
        Args:
            t: Time point in seconds at which to evaluate noise.
            include_sources: List of noise sources to include.
                           Options: ['c13_bath', 'charge_noise', 'temperature',
                                    'johnson', 'external_field', 'strain',
                                    'microwave', 'optical']
                           If None, all enabled sources are included.
                           
        Returns:
            np.ndarray: 3x3 complex noise Hamiltonian matrix in rad/s.
                       This should be added to the static Hamiltonian.
                       
        Notes:
            - Noise is sampled from the noise generator at time t
            - Magnetic noise couples through Zeeman interaction
            - Electric noise can couple through strain effects
            - The noise Hamiltonian is Hermitian
            
        Examples:
            >>> # Get noise at t=1 μs with only C13 bath
            >>> H_noise = ham.get_noise_hamiltonian(t=1e-6, include_sources=['c13_bath'])
        """
        if self.noise_gen is None:
            return np.zeros((3, 3), dtype=complex)
            
        return self.noise_gen.get_noise_hamiltonian(
            self.spin_ops.operators, 
            include_sources
        )
    
    def get_total_hamiltonian(self, t: float = 0.0,
                            include_noise_sources: Optional[List[str]] = None) -> np.ndarray:
        """
        Get the total Hamiltonian including static and noise terms.
        
        Constructs the complete time-dependent Hamiltonian:
            H_total(t) = H_static + H_noise(t)
            
        This is the Hamiltonian that enters the Schrödinger or master equation.
        
        Args:
            t: Time in seconds at which to evaluate the Hamiltonian.
            include_noise_sources: Specific noise sources to include.
                                 If None, all enabled sources are used.
                                 
        Returns:
            np.ndarray: 3x3 complex total Hamiltonian matrix in rad/s.
            
        See Also:
            get_static_hamiltonian: For the time-independent part
            get_noise_hamiltonian: For the noise contribution only
        """
        H_total = self.get_static_hamiltonian()
        H_total += self.get_noise_hamiltonian(t, include_noise_sources)
        return H_total


class NVLindblad:
    """
    Lindblad master equation evolution for NV center open quantum dynamics.
    
    This class implements the Lindblad (quantum master) equation for modeling
    the evolution of an NV center as an open quantum system. The Lindblad
    equation captures both coherent evolution and incoherent processes like
    relaxation and decoherence:
    
        dρ/dt = -i/ℏ[H,ρ] + Σ_k γ_k·(L_k·ρ·L_k† - {L_k†·L_k,ρ}/2)
        
    where ρ is the density matrix, H is the Hamiltonian, L_k are Lindblad
    operators describing dissipation channels, and γ_k are the corresponding
    rates.
    
    The dissipation terms model various decoherence mechanisms:
        - Energy relaxation (T1 processes)
        - Pure dephasing (T2* processes)  
        - Spectral diffusion from noise baths
        - Measurement backaction
        
    Attributes:
        hamiltonian (NVSystemHamiltonian): System Hamiltonian manager
        spin_ops (NVSpinOperators): Spin operator matrices
        default_T1 (float): Default longitudinal relaxation time
        default_T2 (float): Default transverse coherence time
        
    Examples:
        >>> # Create Lindblad evolver with noise
        >>> ham = NVSystemHamiltonian(B_field=[0,0,0.01])
        >>> lindblad = NVLindblad(ham)
        >>> 
        >>> # Evolve from initial state
        >>> rho0 = np.diag([0, 1, 0])  # |ms=0⟩ state
        >>> times, rhos = lindblad.evolve(rho0, (0, 1e-6))
        >>> 
        >>> # Check decoherence
        >>> purity = [np.real(np.trace(rho @ rho)) for rho in rhos]
    """
    
    def __init__(self, hamiltonian: NVSystemHamiltonian):
        """
        Initialize Lindblad evolution with given Hamiltonian.
        
        Args:
            hamiltonian: NVSystemHamiltonian instance containing the
                        system Hamiltonian and noise configuration.
                        
        Notes:
            The Lindblad operators are determined by the noise sources
            enabled in the Hamiltonian's noise generator.
        """
        self.hamiltonian = hamiltonian
        self.spin_ops = hamiltonian.spin_ops
        
        # Default relaxation parameters if not provided by noise
        self.default_T1 = SYSTEM.get_constant('nv_center', 'typical_t1')
        self.default_T2 = SYSTEM.get_constant('nv_center', 'typical_t2')
        
    def get_lindblad_operators(self, 
                             include_sources: Optional[List[str]] = None) -> List[Tuple[np.ndarray, float]]:
        """
        Get Lindblad operators and rates for dissipation.
        
        Constructs the set of Lindblad operators L_k and rates γ_k that
        describe dissipation channels. These can come from:
        
        1. Noise-induced decoherence (from noise generator)
        2. Phenomenological relaxation (T1 processes)
        3. Phenomenological dephasing (T2* processes)
        
        Args:
            include_sources: List of noise sources to include in dissipation.
                           If None, all enabled sources are used.
                           
        Returns:
            List[Tuple[np.ndarray, float]]: List of (operator, rate) pairs
                where operator is a 3x3 complex matrix and rate is in Hz.
                
        Notes:
            - Rates are given as γ such that the decoherence rate is γ
            - For thermal relaxation: L = S_- with γ = 1/T1
            - For pure dephasing: L = Sz with γ = 1/T2φ
            - T2φ is the pure dephasing time: 1/T2 = 1/(2T1) + 1/T2φ
            
        Examples:
            >>> ops = lindblad.get_lindblad_operators()
            >>> for L, gamma in ops:
            ...     print(f"Rate: {gamma:.2e} Hz")
        """
        lindblad_ops = []
        
        # Get noise-induced Lindblad operators
        if self.hamiltonian.noise_gen is not None:
            noise_ops = self.hamiltonian.noise_gen.get_lindblad_operators(
                self.spin_ops.operators,
                include_sources
            )
            lindblad_ops.extend(noise_ops)
        
        # Check if we need to add phenomenological relaxation
        has_thermal = any('thermal' in str(op) for op, _ in lindblad_ops)
        if not has_thermal and self.default_T1 is not None:
            # Add T1 relaxation: decay from |±1⟩ to |0⟩
            gamma_1 = 1 / self.default_T1
            # Note: simplified model, could be extended for temperature dependence
            lindblad_ops.append((self.spin_ops.S_minus, np.sqrt(gamma_1)))
            
        # Check if we need to add phenomenological dephasing  
        has_dephasing = any('dephasing' in str(op) for op, _ in lindblad_ops)
        if not has_dephasing and self.default_T2 is not None:
            # Add pure dephasing
            # T2 includes both relaxation and pure dephasing: 1/T2 = 1/(2T1) + 1/T2φ
            if self.default_T1 is not None:
                gamma_phi = 1/self.default_T2 - 1/(2*self.default_T1)
            else:
                gamma_phi = 1/self.default_T2
                
            if gamma_phi > 0:
                lindblad_ops.append((self.spin_ops.Sz, np.sqrt(gamma_phi)))
            
        return lindblad_ops
    
    def lindblad_rhs(self, t: float, rho_vec: np.ndarray, 
                     include_noise_sources: Optional[List[str]] = None) -> np.ndarray:
        """
        Compute right-hand side of the Lindblad master equation.
        
        Evaluates dρ/dt for the Lindblad equation at time t. This function
        is designed to be used with scipy's ODE integrators.
        
        The Lindblad equation is:
            dρ/dt = -i/ℏ[H,ρ] + Σ_k γ_k·D[L_k](ρ)
            
        where D[L](ρ) = L·ρ·L† - {L†·L,ρ}/2 is the dissipator.
        
        Args:
            t: Time in seconds at which to evaluate the derivative.
            rho_vec: Density matrix as a flattened complex vector (length 9).
            include_noise_sources: Noise sources to include in evolution.
            
        Returns:
            np.ndarray: Time derivative dρ/dt as flattened vector.
            
        Notes:
            - The density matrix is vectorized for ODE solver compatibility
            - Ensures trace preservation and Hermiticity
            - Computational complexity is O(n³) for n-level system
        """
        # Reshape to matrix form
        rho = rho_vec.reshape((3, 3))
        
        # Get time-dependent Hamiltonian
        H = self.hamiltonian.get_total_hamiltonian(t, include_noise_sources)
        
        # Coherent evolution: -i[H, ρ]/ℏ
        hbar = SYSTEM.get_constant('fundamental', 'hbar')
        commutator = H @ rho - rho @ H
        drho_dt = -1j * commutator / hbar
        
        # Dissipative evolution: Lindblad terms
        lindblad_ops = self.get_lindblad_operators(include_noise_sources)
        
        for L, gamma in lindblad_ops:
            if gamma > 0:
                # D[L]ρ = LρL† - (1/2){L†L, ρ}
                L_dag = L.conj().T
                L_dag_L = L_dag @ L
                
                # Lindblad superoperator action
                drho_dt += gamma * (
                    L @ rho @ L_dag - 
                    0.5 * (L_dag_L @ rho + rho @ L_dag_L)
                )
        
        return drho_dt.flatten().astype(complex)
    
    def evolve(self, rho0: np.ndarray, t_span: Tuple[float, float], 
               dt: Optional[float] = None, 
               include_noise_sources: Optional[List[str]] = None,
               method: str = 'RK45',
               rtol: float = 1e-6,
               atol: float = 1e-8) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Evolve density matrix using the Lindblad master equation.
        
        Integrates the Lindblad equation from initial state rho0 over the
        specified time interval. Uses adaptive timestepping for accuracy.
        
        Args:
            rho0: Initial density matrix (3x3 complex array).
                 Must be Hermitian with trace 1.
            t_span: Time interval (t_start, t_end) in seconds.
            dt: Time step for output points. If None, uses noise generator dt
                or 1 ps default. Note: actual integration uses adaptive steps.
            include_noise_sources: List of noise sources to include.
                                 None means all enabled sources.
            method: ODE integration method. Options: 'RK45', 'RK23', 'DOP853'.
                   'RK45' is usually best for this problem.
            rtol: Relative tolerance for integration (default 1e-6).
            atol: Absolute tolerance for integration (default 1e-8).
            
        Returns:
            Tuple[np.ndarray, List[np.ndarray]]: 
                - times: Array of time points
                - rho_history: List of density matrices at each time
                
        Raises:
            ValueError: If rho0 is not a valid density matrix.
            RuntimeError: If integration fails to converge.
            
        Notes:
            - The integrator uses adaptive timestepping internally
            - Output is sampled at fixed dt intervals
            - Total probability (trace) is conserved to numerical precision
            - For long simulations, consider using larger dt to save memory
            
        Examples:
            >>> # Evolve from superposition state
            >>> psi0 = (nv.states['ms0'] + nv.states['ms+1']) / np.sqrt(2)
            >>> rho0 = np.outer(psi0, psi0.conj())
            >>> times, rhos = lindblad.evolve(rho0, (0, 10e-6), dt=1e-9)
            >>> 
            >>> # Check trace preservation
            >>> traces = [np.trace(rho) for rho in rhos]
            >>> np.allclose(traces, 1.0)
            True
        """
        # Validate initial state
        if not np.allclose(rho0, rho0.conj().T):
            raise ValueError("Initial density matrix must be Hermitian")
        if not np.isclose(np.trace(rho0), 1.0):
            warnings.warn(f"Trace of rho0 is {np.trace(rho0)}, normalizing to 1")
            rho0 = rho0 / np.trace(rho0)
            
        # Determine time step
        if dt is None:
            if self.hamiltonian.noise_gen and hasattr(self.hamiltonian.noise_gen.config, 'dt'):
                dt = self.hamiltonian.noise_gen.config.dt
            else:
                dt = 1e-11  # 10 ps default
                
        # Create evaluation time points
        t_eval = np.arange(t_span[0], t_span[1], dt)
        if t_eval[-1] < t_span[1]:
            t_eval = np.append(t_eval, t_span[1])
        
        # Solve the master equation
        try:
            sol = solve_ivp(
                lambda t, y: self.lindblad_rhs(t, y, include_noise_sources),
                t_span,
                rho0.flatten().astype(complex),
                t_eval=t_eval,
                method=method,
                rtol=rtol,
                atol=atol,
                max_step=dt*10  # Limit max step size
            )
        except Exception as e:
            raise RuntimeError(f"Integration failed: {e}")
            
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        
        # Reshape solutions back to density matrices
        rho_history = []
        for i in range(len(sol.t)):
            rho = sol.y[:, i].reshape((3, 3))
            # Ensure Hermiticity (may have small numerical errors)
            rho = (rho + rho.conj().T) / 2
            # Ensure trace = 1
            rho = rho / np.trace(rho)
            rho_history.append(rho)
        
        return sol.t, rho_history
    
    def evolve_with_pulses(self, rho0: np.ndarray, pulse_sequence: List[Dict],
                          include_noise_sources: Optional[List[str]] = None,
                          **evolve_kwargs) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Evolve system with a sequence of microwave pulses.
        
        Simulates the evolution under a series of microwave pulses, each
        potentially having different Rabi frequency, phase, and duration.
        Between pulses, the system undergoes free evolution.
        
        Pulses are implemented by adding an oscillating magnetic field
        term to the Hamiltonian. In the rotating frame, this becomes
        an effective static field.
        
        Args:
            rho0: Initial density matrix.
            pulse_sequence: List of pulse dictionaries, each containing:
                - 'duration': Pulse duration in seconds (required)
                - 'rabi_frequency': Rabi frequency in Hz (default 0)
                - 'phase': Pulse phase in radians (default 0)
                - 'detuning': Frequency detuning in Hz (default 0)
                - 'amplitude': Optional amplitude modulation function
            include_noise_sources: Noise sources to include during evolution.
            **evolve_kwargs: Additional arguments passed to evolve()
            
        Returns:
            Tuple[np.ndarray, List[np.ndarray]]:
                - times: Concatenated time array for entire sequence
                - rho_history: Density matrices for entire sequence
                
        Notes:
            - Pulses are applied in the rotating wave approximation
            - Microwave noise can modulate pulse parameters if enabled
            - Total evolution time is sum of all pulse durations
            - Phase convention: 0 = X pulse, π/2 = Y pulse
            
        Examples:
            >>> # Rabi oscillation
            >>> rabi_pulse = [{
            ...     'duration': 1e-6,
            ...     'rabi_frequency': 10e6,  # 10 MHz
            ...     'phase': 0
            ... }]
            >>> times, rhos = lindblad.evolve_with_pulses(rho0, rabi_pulse)
            >>> 
            >>> # Ramsey sequence
            >>> ramsey = [
            ...     {'duration': 25e-9, 'rabi_frequency': 10e6},  # π/2
            ...     {'duration': 1e-6, 'rabi_frequency': 0},      # Free evolution
            ...     {'duration': 25e-9, 'rabi_frequency': 10e6}   # π/2
            ... ]
            >>> times, rhos = lindblad.evolve_with_pulses(rho0, ramsey)
        """
        current_rho = rho0.copy()
        all_times = []
        all_rhos = []
        current_time = 0.0
        
        for i, pulse in enumerate(pulse_sequence):
            # Extract pulse parameters
            duration = pulse['duration']
            rabi = pulse.get('rabi_frequency', 0.0)
            phase = pulse.get('phase', 0.0)
            detuning = pulse.get('detuning', 0.0)
            
            # Apply microwave noise if enabled
            if (self.hamiltonian.noise_gen and 
                hasattr(self.hamiltonian.noise_gen, 'sources') and
                'microwave' in self.hamiltonian.noise_gen.sources):
                # Noise can modify pulse parameters
                pulse_params = self.hamiltonian.noise_gen.process_microwave_pulse(
                    rabi, duration, phase
                )
                # Use average values for simplicity (could be extended)
                rabi = np.mean(pulse_params['rabi_frequency'])
                phase = np.mean(pulse_params['phase'])
                detuning += np.mean(pulse_params['frequency_offset'])
            
            # Store original field
            original_B = self.hamiltonian.B_static.copy()
            
            # Add effective microwave field (rotating wave approximation)
            if rabi > 0:
                # In rotating frame, MW appears as static transverse field
                B_mw = np.array([
                    rabi * np.cos(phase) / self.hamiltonian.gamma_e,
                    rabi * np.sin(phase) / self.hamiltonian.gamma_e,
                    detuning / self.hamiltonian.gamma_e
                ])
                self.hamiltonian.B_static = original_B + B_mw
            
            # Evolve during pulse
            t_span = (current_time, current_time + duration)
            times, rhos = self.evolve(
                current_rho, t_span, 
                include_noise_sources=include_noise_sources,
                **evolve_kwargs
            )
            
            # Append results (avoiding duplicate time points)
            if len(all_times) > 0:
                # Skip first point to avoid duplication
                all_times.extend(times[1:])
                all_rhos.extend(rhos[1:])
            else:
                all_times.extend(times)
                all_rhos.extend(rhos)
            
            # Update state and time
            current_rho = rhos[-1]
            current_time += duration
            
            # Restore original field
            self.hamiltonian.B_static = original_B
            
        return np.array(all_times), all_rhos


class NVSystem:
    """
    Complete NV center quantum system with integrated functionality.
    
    This is the main class for simulating NV center dynamics. It combines:
    - Hamiltonian construction (NVSystemHamiltonian)
    - Open system evolution (NVLindblad)
    - Noise modeling (NoiseGenerator)
    - Common pulse sequences and protocols
    - Measurement and analysis tools
    - Visualization capabilities
    
    The class provides a high-level interface for common NV experiments
    including Rabi oscillations, Ramsey interferometry, spin echo sequences,
    and T1/T2 measurements.
    
    Attributes:
        noise_gen (NoiseGenerator): Noise generator instance
        hamiltonian (NVSystemHamiltonian): System Hamiltonian
        lindblad (NVLindblad): Lindblad evolution manager
        states (dict): Common quantum states as vectors
        evolve: Direct access to lindblad.evolve method
        evolve_with_pulses: Direct access to lindblad.evolve_with_pulses
        
    Examples:
        >>> # Create NV system with custom noise
        >>> config = NoiseConfiguration()
        >>> config.enable_c13_bath = True
        >>> nv = NVSystem(B_field=[0, 0, 0.01], noise_config=config)
        >>> 
        >>> # Simulate T2* measurement
        >>> tau_values, coherences = nv.simulate_t2_measurement()
        >>> 
        >>> # Run custom pulse sequence
        >>> rho0 = nv.create_initial_state('ms0')
        >>> times, rhos = nv.evolve_with_pulses(rho0, pulse_sequence)
    """
    
    def __init__(self, B_field: np.ndarray = None, 
                 noise_config: Optional[NoiseConfiguration] = None,
                 noise_gen: Optional[NoiseGenerator] = None):
        """
        Initialize complete NV system.
        
        Creates a full NV center simulation environment with specified
        magnetic field and noise configuration.
        
        Args:
            B_field: Static magnetic field [Bx, By, Bz] in Tesla.
                    Default is zero field.
            noise_config: NoiseConfiguration instance specifying which
                         noise sources to enable. If None, no noise.
            noise_gen: Alternative way to specify noise using a
                      pre-configured NoiseGenerator. Takes precedence
                      over noise_config if both are provided.
                      
        Raises:
            ValueError: If both noise_config and noise_gen are specified.
            
        Notes:
            - Use noise_config for new simulations
            - Use noise_gen for compatibility or special configurations
            - Common configs available via NoiseConfiguration.from_preset()
        """
        # Handle noise configuration
        if noise_gen is not None and noise_config is not None:
            raise ValueError("Specify either noise_config or noise_gen, not both")
            
        if noise_gen is not None:
            self.noise_gen = noise_gen
        elif noise_config is not None:
            self.noise_gen = NoiseGenerator(noise_config)
        else:
            self.noise_gen = None
        
        # Initialize components
        self.hamiltonian = NVSystemHamiltonian(B_field, self.noise_gen)
        self.lindblad = NVLindblad(self.hamiltonian)
        
        # Convenience methods
        self.evolve = self.lindblad.evolve
        self.evolve_with_pulses = self.lindblad.evolve_with_pulses
        
        # Common quantum states (as vectors)
        self.states = {
            'ms0': np.array([0, 1, 0], dtype=complex),      # |ms=0⟩
            'ms+1': np.array([0, 0, 1], dtype=complex),     # |ms=+1⟩  
            'ms-1': np.array([1, 0, 0], dtype=complex),     # |ms=-1⟩
            'superposition': np.array([0, 1, 1], dtype=complex) / np.sqrt(2),  # (|0⟩+|+1⟩)/√2
            'ghz': np.array([1, 0, 1], dtype=complex) / np.sqrt(2),  # (|-1⟩+|+1⟩)/√2
        }
        
    def create_initial_state(self, state: Union[str, np.ndarray], 
                           pure: bool = True) -> np.ndarray:
        """
        Create an initial density matrix.
        
        Convenience method for creating common initial states as density
        matrices. Supports both pure and mixed states.
        
        Args:
            state: Either a string key from self.states ('ms0', 'ms+1', etc.)
                  or a 3-element state vector.
            pure: If True, creates pure state |ψ⟩⟨ψ|. If False, creates
                 maximally mixed state in the subspace.
                 
        Returns:
            np.ndarray: 3x3 density matrix.
            
        Examples:
            >>> # Pure |ms=0⟩ state
            >>> rho0 = nv.create_initial_state('ms0')
            >>> 
            >>> # Superposition state
            >>> rho0 = nv.create_initial_state('superposition')
            >>> 
            >>> # Custom state vector
            >>> psi = np.array([1, 2, 1]) / np.sqrt(6)
            >>> rho0 = nv.create_initial_state(psi)
        """
        if isinstance(state, str):
            if state not in self.states:
                raise ValueError(f"Unknown state '{state}'. Options: {list(self.states.keys())}")
            psi = self.states[state]
        else:
            psi = np.asarray(state, dtype=complex)
            if psi.shape != (3,):
                raise ValueError(f"State vector must be 3D, got shape {psi.shape}")
            psi = psi / np.linalg.norm(psi)  # Normalize
            
        if pure:
            return np.outer(psi, psi.conj())
        else:
            # Mixed state in subspace spanned by non-zero elements
            support = np.nonzero(psi)[0]
            rho = np.zeros((3, 3), dtype=complex)
            for i in support:
                rho[i, i] = 1.0 / len(support)
            return rho
        
    def get_state_populations(self, rho: np.ndarray) -> Dict[str, float]:
        """
        Extract populations of computational basis states.
        
        Args:
            rho: Density matrix (3x3).
            
        Returns:
            Dict[str, float]: Populations with keys 'ms-1', 'ms0', 'ms+1'.
            
        Examples:
            >>> pops = nv.get_state_populations(rho)
            >>> print(f"ms=0 population: {pops['ms0']:.3f}")
        """
        return {
            'ms-1': np.real(rho[0, 0]),
            'ms0': np.real(rho[1, 1]), 
            'ms+1': np.real(rho[2, 2])
        }
    
    def get_coherences(self, rho: np.ndarray) -> Dict[str, complex]:
        """
        Extract quantum coherences (off-diagonal elements).
        
        Args:
            rho: Density matrix (3x3).
            
        Returns:
            Dict[str, complex]: Coherences with keys like 'c_01' for ρ₀₁.
        """
        return {
            'c_-10': rho[0, 1],  # ⟨ms=-1|ρ|ms=0⟩
            'c_-1+1': rho[0, 2], # ⟨ms=-1|ρ|ms=+1⟩
            'c_0+1': rho[1, 2],  # ⟨ms=0|ρ|ms=+1⟩
        }
    
    def measure_observable(self, rho: np.ndarray, observable: np.ndarray) -> float:
        """
        Measure expectation value of an observable.
        
        Computes ⟨O⟩ = Tr(ρ·O) for observable O.
        
        Args:
            rho: Density matrix.
            observable: Observable as 3x3 Hermitian matrix.
            
        Returns:
            float: Expectation value (real).
            
        Examples:
            >>> # Measure Sz
            >>> sz_exp = nv.measure_observable(rho, nv.hamiltonian.spin_ops.Sz)
        """
        return np.real(np.trace(rho @ observable))
    
    def calculate_purity(self, rho: np.ndarray) -> float:
        """
        Calculate purity Tr(ρ²) of quantum state.
        
        Purity ranges from 1/3 (maximally mixed) to 1 (pure state).
        
        Args:
            rho: Density matrix.
            
        Returns:
            float: Purity value.
        """
        return np.real(np.trace(rho @ rho))
    
    def calculate_fidelity(self, rho1: np.ndarray, rho2: np.ndarray) -> float:
        """
        Calculate fidelity between two quantum states.
        
        For pure states, fidelity = |⟨ψ₁|ψ₂⟩|².
        General formula: F = [Tr(√(√ρ₁·ρ₂·√ρ₁))]²
        
        Args:
            rho1: First density matrix.
            rho2: Second density matrix.
            
        Returns:
            float: Fidelity between 0 and 1.
        """
        # Simplified for common case where one state is pure
        if np.allclose(rho1 @ rho1, rho1) or np.allclose(rho2 @ rho2, rho2):
            return np.real(np.trace(rho1 @ rho2))
            
        # General case requires matrix square root
        sqrt_rho1 = scipy.linalg.sqrtm(rho1)
        M = sqrt_rho1 @ rho2 @ sqrt_rho1
        return np.real(np.trace(scipy.linalg.sqrtm(M)))**2
    
    def create_pulse(self, angle: float, axis: str = 'x', 
                    rabi_freq: float = 10e6) -> Dict:
        """
        Create a rotation pulse around specified axis.
        
        Args:
            angle: Rotation angle in radians (π = pi pulse).
            axis: Rotation axis ('x', 'y', or angle for arbitrary).
            rabi_freq: Rabi frequency in Hz.
            
        Returns:
            Dict: Pulse dictionary for use with evolve_with_pulses.
            
        Examples:
            >>> # π/2 pulse around X
            >>> pulse = nv.create_pulse(np.pi/2, axis='x')
            >>> # π pulse around Y  
            >>> pulse = nv.create_pulse(np.pi, axis='y')
        """
        duration = angle / (2 * np.pi * rabi_freq)
        
        if axis == 'x':
            phase = 0
        elif axis == 'y':
            phase = np.pi/2
        else:
            try:
                phase = float(axis)
            except:
                raise ValueError(f"axis must be 'x', 'y', or a phase angle")
                
        return {
            'duration': duration,
            'rabi_frequency': 2 * np.pi * rabi_freq,
            'phase': phase,
            'detuning': 0
        }
    
    def create_ramsey_sequence(self, T_ramsey: float, 
                              rabi_freq: float = 10e6,
                              phase2: float = 0) -> List[Dict]:
        """
        Create Ramsey interferometry sequence.
        
        Sequence: π/2 - free evolution - π/2(phase)
        
        Args:
            T_ramsey: Free evolution time in seconds.
            rabi_freq: Rabi frequency for π/2 pulses.
            phase2: Phase of second π/2 pulse (0 for standard Ramsey).
            
        Returns:
            List[Dict]: Pulse sequence.
        """
        pi_half = self.create_pulse(np.pi/2, axis='x', rabi_freq=rabi_freq)
        wait = {'duration': T_ramsey, 'rabi_frequency': 0}
        pi_half_2 = self.create_pulse(np.pi/2, axis=phase2, rabi_freq=rabi_freq)
        
        return [pi_half, wait, pi_half_2]
    
    def create_echo_sequence(self, tau: float,
                           rabi_freq: float = 10e6,
                           echo_axis: str = 'y') -> List[Dict]:
        """
        Create Hahn echo sequence.
        
        Sequence: π/2(x) - τ - π(echo_axis) - τ - π/2(x)
        
        Args:
            tau: Half of the total evolution time.
            rabi_freq: Rabi frequency for pulses.
            echo_axis: Axis for π pulse ('x' or 'y').
            
        Returns:
            List[Dict]: Pulse sequence.
        """
        pi_half_1 = self.create_pulse(np.pi/2, axis='x', rabi_freq=rabi_freq)
        wait1 = {'duration': tau, 'rabi_frequency': 0}
        pi_pulse = self.create_pulse(np.pi, axis=echo_axis, rabi_freq=rabi_freq)
        wait2 = {'duration': tau, 'rabi_frequency': 0}
        pi_half_2 = self.create_pulse(np.pi/2, axis='x', rabi_freq=rabi_freq)
        
        return [pi_half_1, wait1, pi_pulse, wait2, pi_half_2]
    
    def create_cpmg_sequence(self, tau: float, n_pulses: int,
                           rabi_freq: float = 10e6) -> List[Dict]:
        """
        Create Carr-Purcell-Meiboom-Gill (CPMG) sequence.
        
        Sequence: π/2(x) - [τ - π(y) - τ]ⁿ - π/2(x)
        
        Args:
            tau: Time between π pulses.
            n_pulses: Number of π pulses.
            rabi_freq: Rabi frequency.
            
        Returns:
            List[Dict]: Pulse sequence.
        """
        sequence = [self.create_pulse(np.pi/2, axis='x', rabi_freq=rabi_freq)]
        
        for i in range(n_pulses):
            sequence.append({'duration': tau/2, 'rabi_frequency': 0})
            sequence.append(self.create_pulse(np.pi, axis='y', rabi_freq=rabi_freq))
            sequence.append({'duration': tau/2, 'rabi_frequency': 0})
            
        sequence.append(self.create_pulse(np.pi/2, axis='x', rabi_freq=rabi_freq))
        return sequence
    
    def simulate_rabi_oscillations(self, rabi_freq: float = 10e6,
                                 duration: float = 1e-6,
                                 n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Rabi oscillations under continuous driving.
        
        Args:
            rabi_freq: Driving frequency in Hz.
            duration: Total evolution time.
            n_points: Number of time points.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (times, populations)
            where populations has shape (n_points, 3).
        """
        rho0 = self.create_initial_state('ms0')
        pulse = [{
            'duration': duration,
            'rabi_frequency': 2 * np.pi * rabi_freq,
            'phase': 0,
            'detuning': 0
        }]
        
        times, rhos = self.evolve_with_pulses(rho0, pulse, dt=duration/n_points)
        
        populations = np.array([
            [self.get_state_populations(rho)[s] for s in ['ms-1', 'ms0', 'ms+1']]
            for rho in rhos
        ])
        
        return times, populations
    
    def simulate_t2_measurement(self, tau_max: float = 10e-6, 
                               n_points: int = 50,
                               sequence: str = 'ramsey') -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate coherence decay measurement (T2* or T2).
        
        Args:
            tau_max: Maximum evolution time.
            n_points: Number of tau values to sample.
            sequence: 'ramsey' for T2* or 'echo' for T2.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (tau_values, coherences)
            
        Examples:
            >>> # Measure T2*
            >>> taus, coherences = nv.simulate_t2_measurement(sequence='ramsey')
            >>> # Fit exponential decay
            >>> from scipy.optimize import curve_fit
            >>> popt, _ = curve_fit(lambda t, A, T2: A*np.exp(-t/T2), taus, coherences)
            >>> print(f"T2* = {popt[1]*1e6:.1f} μs")
        """
        tau_values = np.linspace(0, tau_max, n_points)
        coherences = np.zeros(n_points)
        
        # Initial state: |ms=0⟩
        rho0 = self.create_initial_state('ms0')
        
        for i, tau in enumerate(tau_values):
            if sequence == 'ramsey':
                pulse_seq = self.create_ramsey_sequence(tau)
            elif sequence == 'echo':
                pulse_seq = self.create_echo_sequence(tau/2)  # tau is total time
            else:
                raise ValueError(f"Unknown sequence: {sequence}")
                
            times, rhos = self.evolve_with_pulses(rho0, pulse_seq)
            
            # Measure final state
            final_rho = rhos[-1]
            # For Ramsey/echo, measure ⟨Sx⟩ or population difference
            coherences[i] = self.measure_observable(final_rho, self.hamiltonian.spin_ops.Sx)
            
        return tau_values, coherences
    
    def simulate_t1_measurement(self, t_max: float = 100e-6,
                               n_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate longitudinal relaxation (T1) measurement.
        
        Prepares |ms=+1⟩ state and measures decay to |ms=0⟩.
        
        Args:
            t_max: Maximum wait time.
            n_points: Number of time points.
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (times, populations)
            where populations are for |ms=+1⟩ state.
        """
        times = np.linspace(0, t_max, n_points)
        populations = np.zeros(n_points)
        
        # Initial state: |ms=+1⟩
        rho0 = self.create_initial_state('ms+1')
        
        # Evolve and sample
        t_all, rhos_all = self.evolve(rho0, (0, t_max), dt=t_max/n_points)
        
        # Extract populations at sample times
        for i, t in enumerate(times):
            idx = np.argmin(np.abs(t_all - t))
            pops = self.get_state_populations(rhos_all[idx])
            populations[i] = pops['ms+1']
            
        return times, populations
    
    def visualize_evolution(self, rho0: np.ndarray, t_max: float,
                          include_noise_sources: Optional[List[str]] = None,
                          dt: Optional[float] = None,
                          figsize: Tuple[float, float] = (12, 8)):
        """
        Visualize time evolution of state populations and coherences.
        
        Creates a comprehensive plot showing:
        - Population dynamics
        - Coherence evolution
        - Purity decay
        - Optional: Bloch sphere trajectory
        
        Args:
            rho0: Initial density matrix.
            t_max: Evolution time in seconds.
            include_noise_sources: Noise sources to include.
            dt: Time step for evolution.
            figsize: Figure size (width, height).
            
        Returns:
            fig, axes: Matplotlib figure and axes objects.
        """
        # Evolve system
        times, rhos = self.evolve(rho0, (0, t_max), dt=dt, 
                                 include_noise_sources=include_noise_sources)
        
        # Extract quantities
        populations = {state: [] for state in ['ms-1', 'ms0', 'ms+1']}
        coherences = []
        purities = []
        
        for rho in rhos:
            # Populations
            state_pops = self.get_state_populations(rho)
            for state, pop in state_pops.items():
                populations[state].append(pop)
                
            # Coherences (magnitude)
            coh = self.get_coherences(rho)
            coherences.append([np.abs(c) for c in coh.values()])
            
            # Purity
            purities.append(self.calculate_purity(rho))
        
        coherences = np.array(coherences)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Convert time to microseconds
        times_us = times * 1e6
        
        # Plot populations
        ax = axes[0, 0]
        ax.plot(times_us, populations['ms-1'], 'b-', label='$|m_s=-1\\rangle$', linewidth=2)
        ax.plot(times_us, populations['ms0'], 'r-', label='$|m_s=0\\rangle$', linewidth=2)
        ax.plot(times_us, populations['ms+1'], 'g-', label='$|m_s=+1\\rangle$', linewidth=2)
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Population')
        ax.set_title('State Populations')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)
        
        # Plot coherences
        ax = axes[0, 1]
        coh_labels = ['$|\\rho_{-1,0}|$', '$|\\rho_{-1,+1}|$', '$|\\rho_{0,+1}|$']
        for i, label in enumerate(coh_labels):
            ax.plot(times_us, coherences[:, i], label=label, linewidth=2)
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Coherence Magnitude')
        ax.set_title('Quantum Coherences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.75)
        
        # Plot purity
        ax = axes[1, 0]
        ax.plot(times_us, purities, 'k-', linewidth=2)
        ax.axhline(y=1.0, color='gray', linestyle='--', label='Pure state')
        ax.axhline(y=1/3, color='gray', linestyle=':', label='Max. mixed')
        ax.set_xlabel('Time (μs)')
        ax.set_ylabel('Purity $\\mathrm{Tr}(\\rho^2)$')
        ax.set_title('State Purity')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.3, 1.05)
        
        # Noise power spectrum or additional info
        ax = axes[1, 1]
        if self.noise_gen is not None:
            # Show active noise sources
            active_sources = []
            config = self.noise_gen.config
            if config.enable_c13_bath: active_sources.append('C13 bath')
            if config.enable_charge_noise: active_sources.append('Charge noise')
            if config.enable_temperature: active_sources.append('Temperature')
            if config.enable_johnson: active_sources.append('Johnson noise')
            
            info_text = "Active Noise Sources:\n" + "\n".join(f"• {s}" for s in active_sources)
            info_text += f"\n\nMagnetic field: {self.hamiltonian.B_static*1e3} mT"
            info_text += f"\nSimulation time: {t_max*1e6:.1f} μs"
            
            ax.text(0.1, 0.5, info_text, transform=ax.transAxes, 
                   fontsize=11, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, 'No noise sources active\n(Ideal system)', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=14, style='italic')
                   
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Simulation Info')
        
        plt.suptitle('NV Center Quantum Evolution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        return fig, axes
    
    def benchmark_performance(self, t_max: float = 1e-6) -> Dict:
        """
        Benchmark simulation performance.
        
        Runs test simulations to measure:
        - Integration speed
        - Memory usage
        - Numerical accuracy
        
        Args:
            t_max: Simulation time for benchmark.
            
        Returns:
            Dict: Performance metrics.
        """
        import time
        import tracemalloc
        
        metrics = {}
        rho0 = self.create_initial_state('superposition')
        
        # Time simulation
        tracemalloc.start()
        start_time = time.time()
        
        times, rhos = self.evolve(rho0, (0, t_max))
        
        elapsed = time.time() - start_time
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Compute metrics
        metrics['wall_time'] = elapsed
        metrics['simulated_time'] = t_max
        metrics['speedup'] = t_max / elapsed
        metrics['time_points'] = len(times)
        metrics['memory_peak_mb'] = peak / 1e6
        
        # Check accuracy (trace preservation)
        trace_errors = [abs(np.trace(rho) - 1.0) for rho in rhos[::10]]
        metrics['max_trace_error'] = max(trace_errors)
        metrics['mean_trace_error'] = np.mean(trace_errors)
        
        return metrics


# Convenience functions for common scenarios

def create_room_temperature_nv(B_field: np.ndarray = None) -> NVSystem:
    """
    Create NV system with typical room temperature noise.
    
    Enables noise sources relevant at 300K:
    - C13 nuclear spin bath
    - Charge noise from surface states
    - Temperature fluctuations
    - Johnson noise from electrodes
    
    Args:
        B_field: Static magnetic field in Tesla.
        
    Returns:
        NVSystem: Configured for room temperature.
    """
    config = NoiseConfiguration.from_preset('room_temperature')
    return NVSystem(B_field, config)

def create_cryogenic_nv(B_field: np.ndarray = None, 
                       temperature: float = 4.0) -> NVSystem:
    """
    Create NV system for cryogenic conditions.
    
    Optimized for low temperature (typically 4K) with reduced:
    - Thermal phonon noise
    - Charge state fluctuations
    - Temperature variations
    
    Args:
        B_field: Static magnetic field in Tesla.
        temperature: Operating temperature in Kelvin.
        
    Returns:
        NVSystem: Configured for cryogenic operation.
    """
    config = NoiseConfiguration.from_preset('cryogenic')
    config.temperature = temperature
    return NVSystem(B_field, config)

def create_low_noise_nv(B_field: np.ndarray = None) -> NVSystem:
    """
    Create NV system with minimal noise for testing.
    
    Only essential noise sources enabled:
    - Weak C13 bath coupling
    - Minimal charge noise
    
    Useful for:
    - Algorithm development
    - Identifying noise-limited features
    - Fast simulations
    
    Args:
        B_field: Static magnetic field in Tesla.
        
    Returns:
        NVSystem: Low-noise configuration.
    """
    config = NoiseConfiguration()
    config.enable_c13_bath = True
    config.c13_concentration = 0.001  # 0.1% concentration
    config.enable_charge_noise = True
    config.charge_noise_strength = 1e3  # Weak coupling
    # Disable other sources
    config.enable_temperature = False
    config.enable_johnson = False
    config.enable_external_field = False
    config.enable_strain = False
    config.enable_microwave = False
    config.enable_optical = False
    
    return NVSystem(B_field, config)


if __name__ == "__main__":
    # Example usage demonstrating key features
    print("QUSIM NV Center Simulation - Example Usage")
    print("="*50)
    
    # Create system with room temperature noise
    print("\n1. Creating NV system with room temperature noise...")
    B_field = np.array([0.0, 0.0, 1e-3])  # 1 mT along z
    nv_system = create_room_temperature_nv(B_field)
    print(f"   Magnetic field: {B_field*1e3} mT")
    print(f"   Active noise sources: {nv_system.noise_gen.get_active_sources()}")
    
    # Simulate coherent evolution
    print("\n2. Simulating coherent evolution from superposition state...")
    rho0 = nv_system.create_initial_state('superposition')
    times, rhos = nv_system.evolve(rho0, (0, 100e-9), dt=1e-9)
    
    initial_pops = nv_system.get_state_populations(rhos[0])
    final_pops = nv_system.get_state_populations(rhos[-1])
    print(f"   Initial populations: {initial_pops}")
    print(f"   Final populations: {final_pops}")
    print(f"   Purity change: {nv_system.calculate_purity(rhos[0]):.3f} → "
          f"{nv_system.calculate_purity(rhos[-1]):.3f}")
    
    # Rabi oscillations
    print("\n3. Simulating Rabi oscillations...")
    times_rabi, pops_rabi = nv_system.simulate_rabi_oscillations(
        rabi_freq=10e6, duration=200e-9, n_points=50
    )
    max_transfer = np.max(pops_rabi[:, 2])  # Max |+1⟩ population
    print(f"   Maximum population transfer to |+1⟩: {max_transfer:.3f}")
    
    # T2* measurement
    print("\n4. Measuring T2* with Ramsey sequence...")
    tau_vals, coherences = nv_system.simulate_t2_measurement(
        tau_max=5e-6, n_points=20, sequence='ramsey'
    )
    
    # Fit exponential decay
    from scipy.optimize import curve_fit
    def exp_decay(t, A, T2):
        return A * np.exp(-t / T2)
    
    try:
        popt, _ = curve_fit(exp_decay, tau_vals, np.abs(coherences), p0=[0.5, 1e-6])
        print(f"   Fitted T2*: {popt[1]*1e6:.2f} μs")
    except:
        print("   T2* fitting failed - coherence decay too fast or irregular")
    
    # Performance benchmark
    print("\n5. Running performance benchmark...")
    metrics = nv_system.benchmark_performance(t_max=1e-7)
    print(f"   Simulation speedup: {metrics['speedup']:.1f}x realtime")
    print(f"   Memory usage: {metrics['memory_peak_mb']:.1f} MB")
    print(f"   Numerical accuracy: {metrics['max_trace_error']:.2e}")
    
    print("\n" + "="*50)
    print("Example completed. Use nv_system.visualize_evolution() for plots.")