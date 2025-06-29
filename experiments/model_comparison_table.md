# Model Comparison Table: QUSIM Implementation vs Literature

## Summary of Mathematical Models in noise_sources.py

| Noise Source | Model Type | Key Equation | Parameters | PSD Type | Typical Values |
|--------------|------------|--------------|------------|----------|----------------|
| **C13 Bath** | Ornstein-Uhlenbeck | `dB = -B/τ dt + σ√(2/τ) dW` | τ=1μs, σ=√ρ×B_c | Lorentzian | B_rms ~ 10 nT |
| **External Field** | Colored Noise | `S(f) ∝ 1/f^α` | α=1, f_cut=1MHz | 1/f^α | Drift ~ 100 nT |
| **Johnson Noise** | Thermal White | `B_rms = √(μ₀kTρ/πd³√δ)` | T=300K, d=1μm | White | ~1 nT/√Hz |
| **Charge State** | Telegraph | `P_jump = R×dt` | R=1Hz base | Telegraph | 0.1-100 Hz |
| **Temperature** | OU Process | `dT = -(T-T₀)/τ dt + σ dW` | τ=1ms, σ=0.1K | Lorentzian | ΔT ~ 0.1 K |
| **Strain** | Mixed | `ε = ε₀ + A sin(ωt) + noise` | f=100Hz, A=1e-7 | Delta + White | ε ~ 10⁻⁶ |
| **MW Noise** | Multiple | Phase: 1/f², Amp: log-normal | φ=0.01 rad/√Hz | 1/f² + Lorentz | 1% amplitude |
| **Optical** | RIN + Shot | `I(t) = I₀(1 + ξ(t))` | RIN=10⁻⁴ | 1/f at low freq | SNR ~ 100 |

## Decoherence Models

| Process | Current Implementation | Physical Origin | Key Parameters |
|---------|----------------------|-----------------|----------------|
| **T1 (Relaxation)** | `Γ = A×ω³×(n+1)` | Phonon emission | A ~ 10⁻³ (RT) |
| **T2 (Dephasing)** | `Γ_φ = const` | Magnetic noise | 1 MHz typical |
| **T2* (FID)** | `1/T2* ≈ γσ_B√τ` | Inhomogeneous B | τ ~ 1 μs |

## Key Equations to Extract from PDFs

### 1. **Spin Bath Hamiltonian** (Look for in PDFs)
```
H = ω_e S_z + Σ_k ω_k I_k^z + Σ_k A_k S_z I_k^z + higher order terms
```

### 2. **Master Equation** (Look for in PDFs)
```
dρ/dt = -i[H,ρ] + Σ_i γ_i (L_i ρ L_i† - 1/2{L_i†L_i, ρ})
```

### 3. **Noise Correlations** (Look for in PDFs)
```
⟨B_i(t)B_j(t')⟩ = δ_ij C(t-t')
```

### 4. **Filter Functions** (Look for in PDFs)
```
χ(t) = |∫₀^t y(t') dt'|²
```

## Validation Data Needed from PDFs

| Measurement | Typical Range | Temperature Dependence | Field Dependence |
|-------------|--------------|----------------------|------------------|
| T1 | 1 μs - 10 ms | ∝ 1/T⁵ (low T) | Weak |
| T2 | 0.1 - 100 μs | Complex | ∝ 1/B (low field) |
| T2* | 0.1 - 10 μs | Weak | Strong |
| Charge stability | 0.1 - 100 Hz | Weak | Via spin state |

## Implementation Gaps to Address

1. **Non-Markovian Dynamics**
   - Current: All Markovian
   - Need: Memory kernels, colored noise correlations

2. **Quantum Correlations**
   - Current: Classical noise
   - Need: Zero-point fluctuations, entanglement

3. **Pulse Sequence Response**
   - Current: Free evolution only
   - Need: Filter functions, dynamical decoupling

4. **Multi-level Physics**
   - Current: Two-level (mostly)
   - Need: Excited states, orbital dynamics

5. **Spatial Correlations**
   - Current: Point NV
   - Need: Extended defect, strain fields

## Recommended PDF Analysis Strategy

For each PDF, extract:

1. **Fundamental Hamiltonian** with all terms
2. **Derivation method** (microscopic → effective)
3. **Parameter values** with error bars
4. **Experimental validation** plots
5. **Temperature/field dependencies**
6. **Comparison with other models**
7. **Limitations and approximations**

## Code Improvement Priorities

Based on current implementation:

1. **High Priority:**
   - Add filter functions for pulse sequences
   - Implement proper T2 calculation (not just constant)
   - Add quantum corrections to thermal noise

2. **Medium Priority:**
   - Non-Markovian bath dynamics
   - Coupled charge-spin dynamics
   - Anisotropic strain coupling

3. **Future Enhancement:**
   - Full many-body spin bath
   - Ab initio parameter calculation
   - Machine learning noise models

This comparison framework will help systematically extract and compare the mathematical models from the PDF sources.