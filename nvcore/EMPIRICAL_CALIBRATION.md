# QUSIM Empirical Parameter Calibration Guide

## ⚠️ IMPORTANT: Empirical Parameters Must Be Measured!

The parameters at the top of `system.json` under `empirical_parameters` are **NOT** universal constants. They **MUST** be measured for your specific experimental setup to get accurate simulations.

## 📋 Required Measurements for Accurate Simulations

### 1. **Charge State Dynamics** 🔌
```json
"jump_rate": 1.0  // [Hz] - MEASURE THIS!
```
**How to measure:**
- Record fluorescence time trace under constant illumination
- Look for telegraph noise (blinking)
- Calculate switching rate from statistics
- Tools: Time-correlated single photon counting (TCSPC)

```json
"laser_ionization_factor": 0.1  // [1/mW] - MEASURE THIS!
```
**How to measure:**
- Vary laser power from 0.1 to 10 mW
- Measure charge state switching rate at each power
- Plot rate vs power, extract slope
- Should be linear at low powers

### 2. **Strain Coupling** 💎
```json
"strain_to_zfs_coupling": 1e7  // [Hz/strain] - MEASURE THIS!
```
**How to measure:**
- Apply known stress to diamond (e.g., bending cantilever)
- Measure ODMR frequency shift
- Calculate strain from Young's modulus
- Coupling = Δf_ODMR / strain

### 3. **Laboratory Noise** 🏢
```json
"lab_field_drift": 1e-7  // [T] - MEASURE THIS!
```
**How to measure:**
- Place magnetometer near experiment
- Record B-field for 24 hours
- Calculate RMS drift and spectrum
- Identify AC line noise, elevators, etc.

```json
"temperature_stability": 0.1  // [K] - MEASURE THIS!
```
**How to measure:**
- Place thermocouple on sample holder
- Record temperature during typical experiment
- Calculate RMS fluctuations

### 4. **Microwave System** 📡
```json
"mw_phase_noise_density": 0.01  // [rad/√Hz] - CHECK SPEC SHEET!
```
**How to find:**
- Check your MW generator specification
- Or measure with spectrum analyzer
- Different for each MW source model

```json
"mw_amplitude_stability": 0.01  // [relative] - MEASURE THIS!
```
**How to measure:**
- Monitor MW power with detector
- Record amplitude over time
- Calculate relative fluctuations

### 5. **Optical System** 🔦
```json
"laser_rin": 1e-4  // [relative] - MEASURE THIS!
```
**How to measure:**
- Direct laser onto fast photodiode
- Record intensity vs time
- Calculate power spectral density
- Extract RIN at relevant frequencies

```json
"collection_efficiency": 0.03  // [fraction] - CALIBRATE THIS!
```
**How to calibrate:**
- Use calibrated single photon source
- Or: Use NV with known brightness
- Compare detected vs emitted photons
- Include all optics losses

### 6. **Sample Properties** 💠
```json
"actual_c13_concentration": 0.011  // [fraction] - VERIFY THIS!
```
**How to verify:**
- Buy isotopically enriched diamond with certificate
- Or: Measure via NMR
- Or: Fit spin echo decay curves

```json
"nv_depth": 10e-9  // [m] - MEASURE THIS!
```
**How to measure:**
- Method 1: Etch-back and track NV properties
- Method 2: Measure charge state stability vs laser power
- Method 3: NV-NV coupling for known separation

## 🔧 Quick Measurement Protocol

### Day 1: Basic Characterization
1. **ODMR spectrum** → Extract D, E parameters
2. **Rabi oscillations** → Calibrate MW power
3. **T1 measurement** → Relaxation time
4. **Ramsey fringes** → T2* and field stability

### Day 2: Noise Characterization  
1. **Long B-field trace** (overnight) → Field drift
2. **Charge state statistics** → Jump rates
3. **Power-dependent measurements** → Laser effects
4. **Temperature logging** → Thermal stability

### Day 3: System Calibration
1. **Laser RIN spectrum** → Optical noise
2. **MW source characterization** → Phase/amplitude noise
3. **Collection efficiency** → Optical setup
4. **Strain sensitivity** → Mechanical coupling

## 📊 Example: Good vs Bad Parameters

### ❌ **Default Parameters (WRONG for your setup)**
```python
noise_gen = NoiseGenerator()  # Uses defaults - INACCURATE!
```

### ✅ **Calibrated Parameters (CORRECT)**
```python
# First, measure your setup
measured_params = {
    'empirical_parameters': {
        'charge_state_dynamics': {
            'jump_rate': 0.3,  # We measured 0.3 Hz blinking
            'laser_ionization_factor': 0.05  # Our laser is gentler
        },
        'noise_amplitudes': {
            'lab_field_drift': 5e-8,  # Our lab is quieter
            'temperature_stability': 0.02  # Better temperature control
        }
        # ... etc
    }
}

# Load custom parameters
config = NoiseConfiguration.from_measured_values(measured_params)
noise_gen = NoiseGenerator(config)
```

## 🚨 Common Mistakes

1. **Using default values** → 10x error in T2 predictions
2. **Not measuring RIN** → Wrong photon statistics  
3. **Ignoring lab environment** → Missing dominant noise
4. **Wrong collection efficiency** → Incorrect SNR

## 📝 Measurement Template

Copy this template for your lab notebook:

```
=== QUSIM Empirical Parameter Measurements ===
Date: _________
Sample: _________
Setup: _________

[ ] Charge jump rate: _______ Hz
[ ] Laser factor: _______ /mW  
[ ] B-field drift: _______ nT
[ ] Temperature stability: _______ mK
[ ] MW amplitude noise: _______ %
[ ] MW phase noise: _______ rad/√Hz
[ ] Laser RIN @ 1kHz: _______ dB/Hz
[ ] Collection efficiency: _______ %
[ ] 13C concentration: _______ %
[ ] NV depth: _______ nm

Notes: _________________________________
```

## 🎯 Impact on Simulation Accuracy

| Parameter | Default | Measured | Impact if Wrong |
|-----------|---------|----------|-----------------|
| Jump rate | 1 Hz | 0.1-100 Hz | Wrong charge dynamics |
| Lab B-drift | 100 nT | 10-1000 nT | Wrong T2* |
| Collection η | 3% | 0.1-10% | Wrong SNR |
| Laser RIN | -140 dB | -120 to -160 | Wrong shot noise |

## ✅ Validation Checklist

After measuring all parameters:

1. [ ] Simulate T2* - compare with measurement
2. [ ] Simulate photon statistics - compare histogram
3. [ ] Simulate Rabi decay - check damping rate
4. [ ] Simulate charge state dynamics - check switching

Only when simulations match experiments are your parameters correct!

---

**Remember**: QUSIM is only as accurate as the parameters you provide. Measure first, simulate second! 📏🔬