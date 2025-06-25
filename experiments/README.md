# QUSIM Experiments

This directory contains realistic NV center experiments implemented using the QUSIM simulation framework.

## Available Experiments

### 1. Minimal π-Pulse Demo (`minimal_demo.py`)

**Description:** Ultra-fast demonstration of core NV physics without full QUSIM simulation:
- Shows |0⟩ → |±1⟩ state flip with π-pulse
- Calculates fluorescence contrast (bright vs dark)
- Includes realistic shot noise
- **Runtime: ~2 seconds**

**Usage:**
```bash
cd experiments
python minimal_demo.py
# OR: cd nvcore && make experiment-minimal
```

### 2. π-Pulse Readout Experiment (`pi_pulse_readout.py`)

**Description:** Complete simulation of a fundamental NV center experiment:
1. Initialize NV center in |0⟩ ground state
2. Apply microwave π-pulse to flip to |±1⟩ state  
3. Optical readout with laser excitation
4. Detect fluorescence photons
5. Measure readout fidelity and contrast

**Key Physics:**
- |0⟩ state is optically bright (high fluorescence)
- |±1⟩ states are optically dark (low fluorescence)  
- Realistic photon shot noise
- Collection and detection efficiency
- Readout fidelity calculation

**Usage:**
```bash
cd experiments
python pi_pulse_readout.py
```

**Output:**
- Photon count histograms (bright vs dark)
- Population evolution during π-pulse
- Readout fidelity metrics
- Experimental summary with SNR

### 2. Quick Demo (`quick_demo.py`)

**Description:** Simplified version for quick testing and demonstrations.

**Usage:**
```bash
cd experiments  
python quick_demo.py
```

## Experimental Parameters

### Typical Values:
- **Magnetic field:** 5-10 mT
- **π-pulse duration:** ~25-50 ns (depends on Rabi frequency)
- **Rabi frequency:** 10-20 MHz
- **Readout time:** 1 μs
- **Collection efficiency:** 3%
- **Detector efficiency:** 80%

### Fluorescence Rates:
- **Bright state (|0⟩):** ~1 MHz
- **Dark state (|±1⟩):** ~50 kHz  
- **Background:** ~1 kHz

## Expected Results

### Ideal Case (no noise):
- **Readout fidelity:** >99%
- **Contrast:** ~0.9
- **Clear separation** between bright/dark distributions

### Realistic Case (with noise):
- **Readout fidelity:** 85-95%
- **Contrast:** 0.7-0.9
- **Shot noise** limits single-shot fidelity

## Physics Background

### NV Center States:
```
    |+1⟩  ───────────  Dark (low fluorescence)
           ≈ 2.87 GHz
    |0⟩   ───────────  Bright (high fluorescence)  
           ≈ 2.87 GHz
    |-1⟩  ───────────  Dark (low fluorescence)
```

### Experimental Sequence:
1. **Initialization:** Laser pumping → |0⟩
2. **Manipulation:** MW π-pulse → |±1⟩
3. **Readout:** Laser + photon detection

### Readout Mechanism:
- **|0⟩ → excited state → fluorescence** (bright)
- **|±1⟩ → no optical transition** (dark)
- **Photon counting** distinguishes states

## Customization

### Modify Experimental Parameters:
```python
experiment = PiPulseReadoutExperiment(
    B_field=np.array([0, 0, 0.01]),  # Magnetic field
    enable_noise=True                # Realistic noise
)

# Custom readout parameters
experiment.readout_time = 2e-6      # 2 μs readout
experiment.laser_power = 2e-3       # 2 mW laser
experiment.collection_efficiency = 0.05  # 5% collection
```

### Add New Experiments:
1. Copy `pi_pulse_readout.py` as template
2. Modify pulse sequences and readout
3. Add to this README

## Integration with QUSIM

These experiments use the full QUSIM framework:
- **NV System:** Complete Hamiltonian with noise
- **Pulse Sequences:** Arbitrary MW control
- **Noise Modeling:** 8 realistic sources
- **Optical Physics:** Fluorescence and detection

## References

1. Doherty et al., "The nitrogen-vacancy colour centre in diamond", Physics Reports 528, 1-45 (2013)
2. Childress et al., "Coherent dynamics of coupled electron and nuclear spin qubits in diamond", Science 314, 281-285 (2006)
3. Tetienne et al., "Magnetic-field-dependent photodynamics of single NV defects in diamond", New Journal of Physics 14, 103033 (2012)