# Mathematical Models Analysis for QUSIM Noise Sources

## Current Implementation in noise_sources.py

### 1. Spin Bath Dynamics (C13 Nuclear Spin Bath)

#### Model: Ornstein-Uhlenbeck Process
The C13 bath noise is modeled as an Ornstein-Uhlenbeck (OU) process:

```python
# State evolution equation:
dX(t) = -X(t)/τ dt + σ√(2/τ) dW(t)

# Discrete update (line 125-127):
X(t+dt) = exp(-dt/τ) * X(t) + √(1-exp(-2dt/τ)) * σ * N(0,1)
```

**Parameters:**
- `concentration`: 0.011 (natural abundance)
- `correlation_time` (τ): 1e-6 s
- `coupling_strength`: 1e-6 T
- `sigma` (σ): √(concentration) * coupling_strength

**Power Spectral Density (Lorentzian):**
```
S(f) = 2σ²τ / (1 + (2πfτ)²)
```

### 2. Decoherence Models

#### T1 Relaxation (Implemented in noise.py, lines 274-297)
Temperature-dependent phonon-induced relaxation:

```python
# Phonon occupation number (Bose-Einstein):
n(ω) = 1 / (exp(ℏω/kBT) - 1)

# Relaxation rates:
γ↓ = γ_phonon * ω³ * (n(ω) + 1)  # Emission
γ↑ = γ_phonon * ω³ * n(ω)        # Absorption

# Where ω = D_gs = 2.87 GHz (zero-field splitting)
```

**Parameters:**
- `phonon_coupling_strength` (γ_phonon): 1e-3 (room temp), 1e-5 (cryo)
- Temperature-dependent

#### T2* Dephasing (Estimated in noise.py, lines 397-437)
Estimated from magnetic noise autocorrelation:

```python
# T2* estimation:
T2* ≈ 1 / (γ_e * σ_B * √τ_c)

# Where:
# γ_e = electron gyromagnetic ratio (2.8e10 Hz/T)
# σ_B = RMS magnetic field noise
# τ_c = correlation time
```

#### Pure Dephasing (T2)
Simplified model with constant rate:
- `typical_dephasing_rate`: 1e6 Hz (default)

### 3. Charge State Dynamics

#### Telegraph Noise Model (ChargeStateNoise class)
Stochastic transitions between NV⁻ and NV⁰:

```python
# Effective transition rate:
R_eff = R_base * (1 + 0.1 * P_laser) * exp(-d/d₀)

# Where:
# R_base = base jump rate (1 Hz)
# P_laser = laser power (mW)
# d = surface distance
# d₀ = decay length (10 nm)

# Transition probability per timestep:
P_transition = R_eff * dt
```

**Power Spectral Density (Telegraph noise):**
```
S(f) = 4R / (1 + (2πf/R)²)
```

### 4. Strain Coupling

#### Static + Dynamic + Random Model (StrainNoise class)

```python
# Total strain:
ε(t) = ε_static + ε_dynamic * sin(2πf_osc * t) + ε_random * N(0,1)

# Zero-field splitting shift:
ΔD = α_strain * ε(t)

# Where α_strain = strain_coupling = 1e7 Hz/strain
```

**Parameters:**
- `static_strain`: 1e-6 (dimensionless)
- `dynamic_amplitude`: 1e-7
- `oscillation_frequency`: 100 Hz
- `random_amplitude`: 1e-8
- `strain_coupling`: 1e7 Hz/strain

### 5. Noise Spectral Densities

#### Magnetic Field Noise PSDs:

1. **C13 Bath (Lorentzian):**
   ```
   S_C13(f) = 2σ²τ / (1 + (2πfτ)²)
   ```

2. **External Field (1/f^α):**
   ```
   S_ext(f) = A² / f^α  (for f < f_cutoff)
   ```

3. **Johnson Noise (White):**
   ```
   S_Johnson(f) = B_rms² = (μ₀kBTρ)/(π^(3/2) * d³ * √δ)
   ```

4. **Charge State (Telegraph):**
   ```
   S_charge(f) = 4R / (1 + (2πf/R)²)
   ```

5. **Temperature Fluctuations (Lorentzian):**
   ```
   S_temp(f) = 2σ_T²τ_T / (1 + (2πfτ_T)²)
   ```

6. **Microwave Noise:**
   - Phase noise: S_φ(f) = σ_φ² / f² (flicker)
   - Frequency drift: Lorentzian with τ_drift

7. **Optical Noise (RIN):**
   ```
   S_RIN(f) = RIN² * f_c / (f + f_c)
   ```

## Key Approximations Used:

1. **Markovian Dynamics**: All noise processes are Markovian
2. **Weak Coupling**: Noise is treated perturbatively
3. **Rotating Wave Approximation**: Not explicitly implemented
4. **Secular Approximation**: Used in Lindblad operators
5. **Point Dipole**: For C13 bath interactions
6. **Classical Noise**: Quantum correlations neglected

## Missing Models (Need from PDFs):

1. **Detailed C13 bath Hamiltonian**
   - Dipolar coupling tensor
   - Nuclear-nuclear interactions
   - Cluster correlation functions

2. **Non-Markovian effects**
   - Memory kernels
   - Colored noise correlations

3. **Quantum noise correlations**
   - Spin-boson model
   - Entanglement with bath

4. **Advanced decoherence**
   - CPMG/dynamical decoupling response
   - Filter functions
   - Noise spectroscopy protocols

5. **Multi-level dynamics**
   - Excited state physics
   - Intersystem crossing rates
   - Orbital dynamics

## Parameter Validation Needs:

1. **Empirical measurements required:**
   - Actual T1, T2, T2* values
   - Charge state jump rates
   - Strain coupling coefficients
   - Lab-specific noise levels

2. **Cross-validation:**
   - Compare simulated vs measured noise spectra
   - Validate decoherence rates
   - Check temperature dependencies

## Recommendations for PDF Analysis:

When analyzing the PDFs, look for:

1. **Microscopic derivations** of the phenomenological parameters
2. **Non-secular terms** in the Hamiltonian
3. **Many-body effects** in the spin bath
4. **Quantum corrections** to classical noise models
5. **Experimental validation** data and protocols
6. **Advanced pulse sequences** and their filter functions
7. **Temperature and field dependencies** of all parameters

This analysis provides a framework for comparing the current implementation with the theoretical models in the PDFs.