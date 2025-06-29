# Detailed Mathematical Equations in QUSIM

## 1. C13 Nuclear Spin Bath Dynamics

### Current Implementation:

**Ornstein-Uhlenbeck Process:**
```
dB_i(t) = -B_i(t)/τ dt + σ√(2/τ) dW_i(t)    (i = x,y,z)
```

**Discrete Evolution (Euler-Maruyama):**
```python
B_i(t+Δt) = exp(-Δt/τ) B_i(t) + √(1-exp(-2Δt/τ)) σ ξ_i

Where:
- τ = 1 μs (correlation time)
- σ = √(ρ_C13) × B_coupling
- ρ_C13 = 0.011 (natural abundance)
- B_coupling = 1 μT
- ξ_i ~ N(0,1)
```

**Power Spectral Density:**
```
S_B(ω) = 2σ²τ/(1 + ω²τ²)
```

### What to Look for in PDFs:

1. **Microscopic Hamiltonian:**
   ```
   H_bath = Σ_k γ_n I_k · B_ext + Σ_{k,l} I_k · D_{kl} · I_l
   ```

2. **Dipolar Coupling Tensor:**
   ```
   D_{kl} = (μ₀/4π) γ_e γ_n ℏ² / r³_{kl} [3(n̂·σ)(n̂·I) - σ·I]
   ```

3. **Cluster Correlation Functions:**
   ```
   C(t) = ⟨B(t)·B(0)⟩ = Σ_k p_k exp(-t/τ_k)
   ```

## 2. Decoherence Models

### T1 Relaxation:

**Current Implementation:**
```python
# Phonon occupation:
n(ω,T) = 1/(exp(ℏω/k_B T) - 1)

# Relaxation rates:
Γ↓ = A_phonon × ω³ × (n(ω,T) + 1)    # ms=0 → ms=±1
Γ↑ = A_phonon × ω³ × n(ω,T)          # ms=±1 → ms=0

# Where:
- ω = D_gs = 2π × 2.87 GHz
- A_phonon = 1e-3 (room temp) or 1e-5 (cryo)
```

**Lindblad Operators:**
```
L₁ = √Γ↓ S⁻
L₂ = √Γ↑ S⁺
```

### T2 Dephasing:

**Pure Dephasing Model:**
```
Γ_φ = γ_e² ∫ S_B(ω) F(ω,t) dω

Where F(ω,t) is the filter function
```

**Current Simplified Implementation:**
```python
# Constant dephasing rate:
Γ_φ = 1 MHz (typical)

# Lindblad operator:
L_φ = √Γ_φ S_z
```

### T2* (Free Induction Decay):

**Estimation from Noise:**
```
1/T2* ≈ γ_e σ_B √τ_c

Where:
- γ_e = 2.8 × 10¹⁰ Hz/T
- σ_B = RMS magnetic noise
- τ_c = correlation time
```

### What to Look for in PDFs:

1. **Microscopic Phonon Coupling:**
   ```
   H_ep = Σ_q g_q (a_q + a_q†) ⊗ O_spin
   ```

2. **Redfield Theory Results:**
   ```
   R_{ij,kl} = ∫₀^∞ dt' e^{iω_{kl}t'} ⟨V_i(t')V_j(0)⟩
   ```

3. **Filter Functions for Pulse Sequences:**
   ```
   F(ω) = |∫₀^T y(t) e^{iωt} dt|²
   ```

## 3. Charge State Dynamics

### Current Implementation:

**Rate Equation Model:**
```
dP_{NV-}/dt = -R₊ P_{NV-} + R₋ P_{NV0}
dP_{NV0}/dt = +R₊ P_{NV-} - R₋ P_{NV0}

Where:
R₊ = R₀(1 + αP_laser) exp(-d/λ)    # NV- → NV0
R₋ = R₀                              # NV0 → NV-
```

**Telegraph Noise PSD:**
```
S_charge(ω) = 4R/(1 + (ω/R)²)

Where R = (R₊ + R₋)/2
```

### What to Look for in PDFs:

1. **Photo-ionization Cross Sections:**
   ```
   σ_{-→0}(λ) = σ₀ f(E_photon - E_threshold)
   ```

2. **Auger Processes:**
   ```
   R_Auger ∝ n_e × n_h × |⟨ψ_i|V_Coulomb|ψ_f⟩|²
   ```

3. **Surface Band Bending:**
   ```
   φ(z) = φ_s exp(-z/λ_D)
   ```

## 4. Strain Coupling

### Current Implementation:

**Strain Model:**
```
ε(t) = ε_static + ε_dynamic sin(2πf_res t) + ε_noise(t)

Where:
- ε_static = 10⁻⁶
- ε_dynamic = 10⁻⁷
- f_res = 100 Hz
- ε_noise ~ N(0, 10⁻⁸)
```

**Hamiltonian Perturbation:**
```
δH = d_∥ ε_z (S_z² - 2/3) + d_⊥ (ε_x S_x² + ε_y S_y²)

Where d_∥ ≈ 1e7 Hz/strain
```

### What to Look for in PDFs:

1. **Full Strain Tensor Coupling:**
   ```
   H_strain = Σ_{ij} M_{ij} ε_{ij}
   ```

2. **Phonon Mode Coupling:**
   ```
   ε_{ij}(r,t) = Σ_k √(ℏ/2ρVω_k) e_k^{ij} (a_k e^{ik·r} + h.c.)
   ```

## 5. Noise Power Spectral Densities

### Current Implementations:

**1. Lorentzian (OU Process):**
```
S(f) = S₀/(1 + (f/f_c)²)
```

**2. 1/f^α Noise:**
```
S(f) = A/f^α    (f_min < f < f_max)
```

**3. White Noise:**
```
S(f) = N₀ = constant
```

**4. Telegraph Noise:**
```
S(f) = 4Aτ/(1 + (2πfτ)²)
```

### What to Look for in PDFs:

1. **Non-Markovian Spectra:**
   ```
   S(ω) = ∫_{-∞}^∞ dt e^{iωt} K(t)
   ```

2. **Multi-Lorentzian Models:**
   ```
   S(ω) = Σ_i A_i τ_i/(1 + ω²τ_i²)
   ```

3. **Quantum Noise Spectra:**
   ```
   S_Q(ω) = ℏω coth(ℏω/2k_B T) S_classical(ω)
   ```

## Key Mathematical Tools to Extract from PDFs:

1. **Master Equation Coefficients**
2. **Floquet Theory for Driven Systems**
3. **Magnus Expansion Terms**
4. **Cluster Expansion Methods**
5. **Path Integral Formulations**
6. **Stochastic Schrödinger Equations**
7. **Quantum Process Tomography Results**

## Validation Metrics:

To compare with experiments, extract:

1. **Coherence Times:**
   - T1 vs temperature
   - T2 vs pulse sequence
   - T2* vs magnetic field

2. **Noise Spectra:**
   - Measured PSDs
   - Correlation functions
   - Allan variance

3. **State Fidelities:**
   - Gate fidelities
   - Readout fidelities
   - Process fidelities

This framework will help identify which equations from the PDFs should be incorporated into the QUSIM implementation.