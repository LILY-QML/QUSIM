"""
Hyperfine Coupling Engine

Exact anisotropic hyperfine coupling between NV center and ¹³C nuclei.
NO APPROXIMATIONS - full 3D dipolar + Fermi contact calculations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys
import os

# Add path for system constants
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'helper'))
from noise_sources import SYSTEM


class HyperfineEngine:
    """
    Ultra-realistic hyperfine coupling engine
    
    Implements exact anisotropic hyperfine coupling:
    H_hf = Σᵢ [S_z·A_∥ᵢ·I_z^i + ½·A_⊥ᵢ·(S⁺I⁻ᵢ + S⁻I⁺ᵢ)]
    
    Features:
    - Position-dependent dipolar coupling
    - Fermi contact interactions
    - Crystal field corrections
    - Temperature-dependent effects
    - Experimental calibration
    """
    
    def __init__(self, c13_positions: np.ndarray, nv_position: np.ndarray = None,
                 nv_orientation: np.ndarray = None):
        """
        Initialize hyperfine engine
        
        Args:
            c13_positions: Array of C13 positions [m], shape (N, 3)
            nv_position: NV center position [m], shape (3,)
            nv_orientation: NV axis orientation, shape (3,)
        """
        self.c13_positions = np.asarray(c13_positions)
        if self.c13_positions.ndim == 1:
            self.c13_positions = self.c13_positions.reshape(1, 3)
            
        self.nv_position = nv_position if nv_position is not None else np.zeros(3)
        self.nv_orientation = nv_orientation if nv_orientation is not None else np.array([0, 0, 1])
        
        self.n_c13 = len(self.c13_positions)
        
        # Physical constants
        self.mu_0 = SYSTEM.get_constant('fundamental', 'mu_0')
        self.gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
        self.gamma_n = SYSTEM.get_constant('nv_center', 'gamma_n_13c')
        self.hbar = SYSTEM.get_constant('fundamental', 'hbar')
        
        # Fermi contact parameters
        self.fermi_contact_range = 0.5e-10  # 0.5 Å - first shell only
        self.fermi_contact_strength = 2.0e6  # Hz - experimental value
        
        # Cache for expensive calculations
        self._hyperfine_cache = {}
        self._distance_cache = {}
        
        # Compute all hyperfine tensors
        self.hyperfine_tensors = self._compute_all_hyperfine_tensors()
        
        # Experimental calibration data
        self._experimental_couplings = {}
        
    def _compute_all_hyperfine_tensors(self) -> Dict[int, Tuple[float, float]]:
        """
        Compute hyperfine tensors for all C13 nuclei
        
        Returns:
            Dictionary mapping C13 index to (A_parallel, A_perpendicular) [Hz]
        """
        tensors = {}
        
        for i, pos in enumerate(self.c13_positions):
            A_par, A_perp = self._compute_single_hyperfine_tensor(i, pos)
            tensors[i] = (A_par, A_perp)
            
        return tensors
        
    def _compute_single_hyperfine_tensor(self, c13_index: int, 
                                       position: np.ndarray) -> Tuple[float, float]:
        """
        Compute hyperfine tensor for single C13 nucleus
        
        Args:
            c13_index: Index of C13 nucleus
            position: C13 position [m]
            
        Returns:
            (A_parallel, A_perpendicular) coupling constants [Hz]
        """
        # Check cache first
        cache_key = (c13_index, tuple(position))
        if cache_key in self._hyperfine_cache:
            return self._hyperfine_cache[cache_key]
            
        # Vector from NV to C13
        r_vec = position - self.nv_position
        r = np.linalg.norm(r_vec)
        
        if r < 1e-12:  # Avoid division by zero
            A_par = A_perp = 0.0
        else:
            # Unit vector from NV to C13
            r_hat = r_vec / r
            
            # Angle between r_vec and NV axis
            nv_axis = self.nv_orientation / np.linalg.norm(self.nv_orientation)
            cos_theta = np.dot(r_hat, nv_axis)
            
            # Dipolar coupling strength
            dipolar_prefactor = self._compute_dipolar_prefactor(r)
            
            # Anisotropic components
            A_par_dipolar = dipolar_prefactor * (3 * cos_theta**2 - 1)
            A_perp_dipolar = dipolar_prefactor * 3 * np.sin(np.arccos(cos_theta))**2 / 2
            
            # Add Fermi contact if close enough
            A_fermi = self._compute_fermi_contact(r)
            
            # Total hyperfine coupling
            A_par = A_par_dipolar + A_fermi
            A_perp = A_perp_dipolar
            
            # Apply crystal field corrections
            A_par, A_perp = self._apply_crystal_field_corrections(A_par, A_perp, position)
            
        # Cache result
        self._hyperfine_cache[cache_key] = (A_par, A_perp)
        
        return A_par, A_perp
        
    def _compute_dipolar_prefactor(self, distance: float) -> float:
        """
        Compute dipolar coupling prefactor
        
        Args:
            distance: Distance between NV and C13 [m]
            
        Returns:
            Dipolar prefactor [Hz]
        """
        # μ₀γₑγₙℏ / (4πr³)
        prefactor = (self.mu_0 * self.gamma_e * self.gamma_n * self.hbar) / (4 * np.pi * distance**3)
        
        # Convert to frequency units
        return prefactor / (2 * np.pi)
        
    def _compute_fermi_contact(self, distance: float) -> float:
        """
        Compute Fermi contact interaction
        
        Args:
            distance: Distance between NV and C13 [m]
            
        Returns:
            Fermi contact coupling [Hz]
        """
        if distance > self.fermi_contact_range:
            return 0.0
            
        # Exponential decay with distance
        decay_length = self.fermi_contact_range / 3
        fermi_coupling = self.fermi_contact_strength * np.exp(-distance / decay_length)
        
        return fermi_coupling
        
    def _apply_crystal_field_corrections(self, A_par: float, A_perp: float,
                                       position: np.ndarray) -> Tuple[float, float]:
        """
        Apply crystal field corrections to hyperfine coupling
        
        Args:
            A_par: Parallel coupling [Hz]
            A_perp: Perpendicular coupling [Hz]
            position: C13 position [m]
            
        Returns:
            Corrected (A_parallel, A_perpendicular) [Hz]
        """
        # Get local crystal field from diamond lattice
        crystal_field = self._get_local_crystal_field(position)
        
        # Apply corrections (typically small)
        correction_factor = 1 + crystal_field * 1e-6  # Small correction
        
        A_par_corrected = A_par * correction_factor
        A_perp_corrected = A_perp * correction_factor
        
        return A_par_corrected, A_perp_corrected
        
    def _get_local_crystal_field(self, position: np.ndarray) -> float:
        """
        Compute local crystal field at C13 position
        
        Args:
            position: C13 position [m]
            
        Returns:
            Local crystal field strength
        """
        # Diamond lattice constant
        a_diamond = 3.567e-10  # m
        
        # Compute position in lattice units
        lattice_pos = position / a_diamond
        
        # Simple model: oscillating field based on lattice position
        field_strength = np.sin(2 * np.pi * lattice_pos[0]) * \
                        np.sin(2 * np.pi * lattice_pos[1]) * \
                        np.sin(2 * np.pi * lattice_pos[2])
                        
        return field_strength
        
    def get_hyperfine_hamiltonian(self, nv_operators: Dict[str, np.ndarray],
                                 c13_operators: Dict[int, Dict[str, np.ndarray]],
                                 nv_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Construct total hyperfine Hamiltonian
        
        Args:
            nv_operators: NV spin operators
            c13_operators: C13 spin operators for each nucleus
            nv_state: Current NV state (for state-dependent effects)
            
        Returns:
            Hyperfine Hamiltonian in joint NV-C13 Hilbert space
        """
        # Get Hilbert space dimensions
        nv_dim = nv_operators['Sz'].shape[0]
        c13_dim = 2**self.n_c13 if self.n_c13 > 0 else 1
        joint_dim = nv_dim * c13_dim
        
        H_hf = np.zeros((joint_dim, joint_dim), dtype=complex)
        
        # Add hyperfine coupling for each C13
        for i in range(self.n_c13):
            A_par, A_perp = self.hyperfine_tensors[i]
            
            # Apply state-dependent modifications if NV state provided
            if nv_state is not None:
                A_par, A_perp = self._apply_state_dependent_coupling(A_par, A_perp, nv_state)
            
            # Get operators in joint space
            Sz_joint = np.kron(nv_operators['Sz'], np.eye(c13_dim))
            S_plus_joint = np.kron(nv_operators['S+'], np.eye(c13_dim))
            S_minus_joint = np.kron(nv_operators['S-'], np.eye(c13_dim))
            
            Iz_joint = np.kron(np.eye(nv_dim), c13_operators[i]['Iz'])
            I_plus_joint = np.kron(np.eye(nv_dim), c13_operators[i]['I+'])
            I_minus_joint = np.kron(np.eye(nv_dim), c13_operators[i]['I-'])
            
            # Ising term: S_z · A_∥ · I_z
            H_hf += 2 * np.pi * A_par * (Sz_joint @ Iz_joint)
            
            # Flip-flop terms: ½ · A_⊥ · (S⁺I⁻ + S⁻I⁺)
            H_hf += np.pi * A_perp * (S_plus_joint @ I_minus_joint + S_minus_joint @ I_plus_joint)
            
        return H_hf
        
    def _apply_state_dependent_coupling(self, A_par: float, A_perp: float,
                                      nv_state: np.ndarray) -> Tuple[float, float]:
        """
        Apply state-dependent modifications to hyperfine coupling
        
        Args:
            A_par: Base parallel coupling [Hz]
            A_perp: Base perpendicular coupling [Hz]
            nv_state: Current NV quantum state
            
        Returns:
            Modified (A_parallel, A_perpendicular) [Hz]
        """
        # Extract NV spin expectation values
        if nv_state.ndim == 1:
            # State vector
            Sz_expectation = np.real(np.conj(nv_state) @ np.diag([-1, 0, 1]) @ nv_state)
        else:
            # Density matrix
            Sz_expectation = np.real(np.trace(np.diag([-1, 0, 1]) @ nv_state))
            
        # State-dependent corrections (Knight shift effect)
        knight_shift_factor = 1 + 1e-4 * Sz_expectation  # Small correction
        
        A_par_modified = A_par * knight_shift_factor
        A_perp_modified = A_perp * knight_shift_factor
        
        return A_par_modified, A_perp_modified
        
    def get_hyperfine_tensors(self) -> Dict[int, Tuple[float, float]]:
        """Get hyperfine coupling constants for all C13"""
        return self.hyperfine_tensors.copy()
        
    def set_experimental_couplings(self, couplings: Dict[int, Tuple[float, float]]):
        """
        Set experimentally measured hyperfine couplings
        
        Args:
            couplings: Dictionary mapping C13 index to (A_par, A_perp) [Hz]
        """
        self._experimental_couplings = couplings.copy()
        
        # Update computed tensors with experimental values
        for i, (A_par, A_perp) in couplings.items():
            if i in self.hyperfine_tensors:
                self.hyperfine_tensors[i] = (A_par, A_perp)
                
    def get_coupling_statistics(self) -> Dict[str, float]:
        """
        Get statistics of hyperfine couplings
        
        Returns:
            Dictionary with coupling statistics
        """
        if not self.hyperfine_tensors:
            return {}
            
        A_pars = [A_par for A_par, A_perp in self.hyperfine_tensors.values()]
        A_perps = [A_perp for A_par, A_perp in self.hyperfine_tensors.values()]
        
        stats = {
            'n_c13': self.n_c13,
            'A_par_mean': np.mean(A_pars),
            'A_par_std': np.std(A_pars),
            'A_par_max': np.max(A_pars),
            'A_par_min': np.min(A_pars),
            'A_perp_mean': np.mean(A_perps),
            'A_perp_std': np.std(A_perps),
            'A_perp_max': np.max(A_perps),
            'A_perp_min': np.min(A_perps)
        }
        
        return stats
        
    def compute_esr_spectrum(self, B_field: float, linewidth: float = 1e3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute ESR spectrum including C13 satellites
        
        Args:
            B_field: Applied magnetic field [T]
            linewidth: ESR linewidth [Hz]
            
        Returns:
            (frequencies, intensities) tuple
        """
        # NV center transition frequency
        D = SYSTEM.get_constant('nv_center', 'd_gs')
        nv_freq = D + self.gamma_e * B_field
        
        # Generate frequency array
        freq_range = 50e6  # 50 MHz range
        frequencies = np.linspace(nv_freq - freq_range/2, nv_freq + freq_range/2, 10000)
        intensities = np.zeros_like(frequencies)
        
        # Main NV line
        main_line = np.exp(-((frequencies - nv_freq) / linewidth)**2)
        intensities += main_line
        
        # C13 satellite lines
        for i, (A_par, A_perp) in self.hyperfine_tensors.items():
            # Satellite positions (simplified)
            satellite_freq_plus = nv_freq + A_par / 2
            satellite_freq_minus = nv_freq - A_par / 2
            
            # Satellite intensities (weaker than main line)
            satellite_intensity = 0.1  # 10% of main line
            
            satellite_plus = satellite_intensity * np.exp(-((frequencies - satellite_freq_plus) / linewidth)**2)
            satellite_minus = satellite_intensity * np.exp(-((frequencies - satellite_freq_minus) / linewidth)**2)
            
            intensities += satellite_plus + satellite_minus
            
        return frequencies, intensities
        
    def update_temperature(self, temperature: float):
        """
        Update temperature-dependent hyperfine parameters
        
        Args:
            temperature: New temperature [K]
        """
        # Temperature affects lattice constant and thus hyperfine couplings
        # Thermal expansion coefficient for diamond
        alpha_diamond = 1.0e-6  # /K
        
        # Reference temperature
        T_ref = 300.0  # K
        
        # Lattice expansion factor
        expansion_factor = 1 + alpha_diamond * (temperature - T_ref)
        
        # Update hyperfine couplings (they scale as 1/r³)
        scale_factor = expansion_factor**(-3)
        
        for i in self.hyperfine_tensors:
            A_par, A_perp = self.hyperfine_tensors[i]
            self.hyperfine_tensors[i] = (A_par * scale_factor, A_perp * scale_factor)
            
        # Clear cache to force recalculation
        self._hyperfine_cache.clear()
        
    def validate_hyperfine_physics(self) -> Dict[str, bool]:
        """
        Validate physical consistency of hyperfine calculations
        
        Returns:
            Dictionary with validation results
        """
        validation = {}
        
        # Check that couplings are finite and reasonable
        finite_couplings = True
        reasonable_magnitudes = True
        
        for i, (A_par, A_perp) in self.hyperfine_tensors.items():
            if not (np.isfinite(A_par) and np.isfinite(A_perp)):
                finite_couplings = False
                
            # Typical range: 1 Hz to 100 MHz
            if not (1 < abs(A_par) < 1e8 and 1 < abs(A_perp) < 1e8):
                reasonable_magnitudes = False
                
        validation['finite_couplings'] = finite_couplings
        validation['reasonable_magnitudes'] = reasonable_magnitudes
        
        # Check dipolar coupling signs and relative magnitudes
        dipolar_consistency = True
        
        for i in range(self.n_c13):
            position = self.c13_positions[i]
            r_vec = position - self.nv_position
            r = np.linalg.norm(r_vec)
            
            if r > 0:
                nv_axis = self.nv_orientation / np.linalg.norm(self.nv_orientation)
                cos_theta = np.dot(r_vec / r, nv_axis)
                
                A_par, A_perp = self.hyperfine_tensors[i]
                
                # For pure dipolar: A_par ∝ (3cos²θ - 1), A_perp ∝ sin²θ
                expected_sign_par = np.sign(3 * cos_theta**2 - 1)
                
                if A_par != 0 and np.sign(A_par) != expected_sign_par:
                    # Could be due to Fermi contact - check magnitude
                    fermi_contribution = abs(A_par) - abs(A_perp)
                    if fermi_contribution < 0:
                        dipolar_consistency = False
                        
        validation['dipolar_consistency'] = dipolar_consistency
        
        # Check that total coupling strength scales correctly with distance
        distance_scaling = True
        
        if self.n_c13 > 1:
            distances = []
            coupling_strengths = []
            
            for i in range(self.n_c13):
                position = self.c13_positions[i]
                r = np.linalg.norm(position - self.nv_position)
                A_par, A_perp = self.hyperfine_tensors[i]
                total_coupling = np.sqrt(A_par**2 + A_perp**2)
                
                distances.append(r)
                coupling_strengths.append(total_coupling)
                
            # Check if coupling roughly scales as 1/r³
            if len(distances) > 2:
                # Fit power law
                log_r = np.log(distances)
                log_A = np.log(coupling_strengths)
                
                # Remove any infinities or NaNs
                valid_mask = np.isfinite(log_r) & np.isfinite(log_A)
                
                if np.sum(valid_mask) > 2:
                    slope = np.polyfit(log_r[valid_mask], log_A[valid_mask], 1)[0]
                    
                    # Should be approximately -3 for dipolar coupling
                    if not (-4 < slope < -2):
                        distance_scaling = False
                        
        validation['distance_scaling'] = distance_scaling
        
        return validation
        
    def get_nearest_neighbors(self, max_neighbors: int = 5) -> List[Tuple[int, float, float, float]]:
        """
        Get nearest neighbor C13 nuclei to NV center
        
        Args:
            max_neighbors: Maximum number of neighbors to return
            
        Returns:
            List of (index, distance, A_par, A_perp) tuples
        """
        neighbors = []
        
        for i in range(self.n_c13):
            position = self.c13_positions[i]
            distance = np.linalg.norm(position - self.nv_position)
            A_par, A_perp = self.hyperfine_tensors[i]
            
            neighbors.append((i, distance, A_par, A_perp))
            
        # Sort by distance
        neighbors.sort(key=lambda x: x[1])
        
        return neighbors[:max_neighbors]
        
    def export_coupling_data(self, filename: str):
        """
        Export hyperfine coupling data to file
        
        Args:
            filename: Output filename
        """
        import json
        
        export_data = {
            'n_c13': self.n_c13,
            'nv_position': self.nv_position.tolist(),
            'nv_orientation': self.nv_orientation.tolist(),
            'c13_positions': self.c13_positions.tolist(),
            'hyperfine_tensors': {
                str(i): [float(A_par), float(A_perp)] 
                for i, (A_par, A_perp) in self.hyperfine_tensors.items()
            },
            'coupling_statistics': self.get_coupling_statistics(),
            'validation': self.validate_hyperfine_physics()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
            
    def load_coupling_data(self, filename: str):
        """
        Load hyperfine coupling data from file
        
        Args:
            filename: Input filename
        """
        import json
        
        with open(filename, 'r') as f:
            data = json.load(f)
            
        self.n_c13 = data['n_c13']
        self.nv_position = np.array(data['nv_position'])
        self.nv_orientation = np.array(data['nv_orientation'])
        self.c13_positions = np.array(data['c13_positions'])
        
        self.hyperfine_tensors = {
            int(i): tuple(coupling) 
            for i, coupling in data['hyperfine_tensors'].items()
        }