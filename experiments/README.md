# QUSIM Experiments

This directory contains the main interfaces for running NV center experiments using the QUSIM simulation framework.

## 🎯 Main Interfaces

### 1. **Interactive GUI** (`GUI/qusim_gui.py`) 🖥️

**Description:** Complete graphical interface for all QUSIM experiments:
- **4 Experiment Types**: Minimal, photons, quick, full simulations
- **Real-time plotting**: Live visualization of results  
- **Parameter control**: Adjust B-field, measurements, Rabi frequency
- **Data export**: Save results and plots
- **Progress tracking**: See experiment status

**Features:**
- Easy point-and-click operation
- Professional plots with matplotlib
- Experiment history and logging
- Parameter optimization tools

**Usage:**
```bash
cd experiments/GUI
python qusim_gui.py
```

### 2. **Educational Notebook** (`notebooks/QUSIM_Experiments.ipynb`) 📚

**Description:** Comprehensive Jupyter notebook with detailed explanations:
- **4 Complete Experiments**: From basic π-pulse to full quantum simulation
- **Physics Background**: Detailed theory and explanations
- **Step-by-step Code**: Well-documented implementations
- **Interactive Plots**: Comprehensive visualizations
- **Educational Content**: Perfect for learning NV physics

**Experiments Included:**
1. **Minimal π-Pulse Demo** - Foundation physics
2. **Time-Resolved Photon Counting** - 1 ns resolution readout ⭐
3. **Parameter Study** - Optimization analysis  
4. **Full QUSIM Simulation** - Complete quantum dynamics

**Usage:**
```bash
cd experiments/notebooks
jupyter notebook QUSIM_Experiments.ipynb
```

## 🔧 How to Run Experiments

### Option 1: Interactive GUI (Recommended)
```bash
cd QUSIM/experiments/GUI
python qusim_gui.py
```
- Point-and-click interface
- Real-time plotting
- Parameter control
- Save results

### Option 2: Educational Notebook  
```bash
cd QUSIM/experiments/notebooks
jupyter notebook QUSIM_Experiments.ipynb
```
- Step-by-step explanations
- Interactive code cells
- Physics background
- Complete tutorials

### Option 3: Command Line (via nvcore)
```bash
cd QUSIM/nvcore
python core.py --help                    # See all options
python core.py --demo                    # Quick demo
make experiment-photons                  # Time-resolved photons
make experiment-help                     # See all experiments
```

## 📊 What You Can Do

### **Time-Resolved Photon Counting** ⭐
- MW π-pulse at resonance frequency
- 1 nanosecond time resolution
- 600 ns readout duration  
- Realistic count rates (10+ Mcps)

### **Parameter Optimization**
- Study B-field, laser power, readout time
- Find optimal collection efficiency
- Analyze SNR and fidelity trade-offs

### **Full Quantum Simulation**
- Complete NV Hamiltonian 
- All 8 noise sources active
- Lindblad evolution dynamics
- Arbitrary pulse sequences

## 🎯 Physics Highlights

### NV Center Basics:
- **Spin-1 quantum system** in diamond
- **|0⟩ bright, |±1⟩ dark** optical readout
- **~2.87 GHz zero-field splitting**
- **Room temperature operation**

### Key Experimental Techniques:
- **π-pulse manipulation:** |0⟩ ↔ |±1⟩ coherent control
- **Optical readout:** Single-photon detection
- **Time-resolved counting:** Nanosecond precision
- **Noise modeling:** 8 realistic sources

## 🚀 Getting Started

1. **Start with GUI**: Easy point-and-click experiments
2. **Try Notebook**: Learn the physics step-by-step  
3. **Explore Parameters**: Optimize your experiments
4. **Advanced Simulations**: Full quantum dynamics

Both interfaces give you complete access to QUSIM's NV center simulation capabilities!