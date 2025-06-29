#!/usr/bin/env python3
"""
QUSIM Experiment GUI

Intuitive graphical interface for running all QUSIM NV center experiments.
Provides easy parameter control and real-time visualization of results.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import subprocess
import time
from datetime import datetime

# Add parent directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'nvcore'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'nvcore', 'lib'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'nvcore', 'helper'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'nvcore', 'modules'))

# Import QUSIM modules - Required for operation
try:
    from noise import (
        NoiseGenerator,
        NoiseConfiguration,
        create_realistic_noise_generator,
        create_advanced_realistic_generator, 
        create_precision_experiment_generator
    )
    from nv_system import NVSystem
    from lindblad import LindLadEvo
    from noise_sources import SYSTEM
except ImportError as e:
    print(f"FEHLER: QUSIM Module konnten nicht importiert werden: {e}")
    print("Bitte stellen Sie sicher, dass QUSIM korrekt installiert ist.")
    print("Führen Sie aus: cd nvcore && python -m pip install -e .")
    sys.exit(1)


class QUSIMExperimentGUI:
    """
    Main GUI class for QUSIM experiments.
    
    Provides an intuitive interface for:
    - Selecting experiments
    - Setting parameters
    - Running simulations
    - Visualizing results
    - Saving data
    """
    
    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("QUSIM - NV Center Experiment Suite")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Experiment tracking
        self.current_experiment = None
        self.experiment_thread = None
        self.results = {}
        
        # GUI State
        self.is_running = False
        
        self.setup_gui()
        self.center_window()
        
    def center_window(self):
        """Center the window on the screen."""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
        
    def setup_gui(self):
        """Set up the main GUI layout."""
        
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="QUSIM - NV Center Experiment Suite", 
                              font=('Arial', 20, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(title_frame, text="Quantum Simulation of Nitrogen-Vacancy Centers", 
                                 font=('Arial', 12), fg='#ecf0f1', bg='#2c3e50')
        subtitle_label.pack()
        
        # Main content area
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Experiment selection and parameters
        left_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        left_frame.pack(side='left', fill='y', padx=(0, 5))
        
        self.setup_experiment_panel(left_frame)
        
        # Right panel - Results and visualization
        right_frame = tk.Frame(main_frame, bg='white', relief='raised', bd=2)
        right_frame.pack(side='right', fill='both', expand=True, padx=(5, 0))
        
        self.setup_results_panel(right_frame)
        
        # Status bar
        self.setup_status_bar()
        
    def setup_experiment_panel(self, parent):
        """Set up the experiment selection and parameter panel."""
        
        # Panel title
        exp_title = tk.Label(parent, text="🔬 Experiments", font=('Arial', 16, 'bold'), 
                           bg='white', fg='#2c3e50')
        exp_title.pack(pady=10)
        
        # Experiment selection
        exp_frame = tk.LabelFrame(parent, text="Select Experiment", font=('Arial', 12, 'bold'),
                                 bg='white', fg='#34495e', padx=10, pady=10)
        exp_frame.pack(fill='x', padx=10, pady=5)
        
        self.experiment_var = tk.StringVar(value="minimal")
        
        experiments = [
            ("minimal", "🚀 Minimal π-Pulse Demo", "Ultra-fast demonstration (2s)"),
            ("photons", "⭐ Time-Resolved Photons", "YOUR requested experiment! (10s)"),
            ("quick", "⚡ Quick Demo", "QUSIM simulation (30s)"),
            ("full", "🔬 Full Experiment", "Complete realistic simulation (60s)")
        ]
        
        for value, title, desc in experiments:
            frame = tk.Frame(exp_frame, bg='white')
            frame.pack(fill='x', pady=2)
            
            rb = tk.Radiobutton(frame, text=title, variable=self.experiment_var, value=value,
                               font=('Arial', 11, 'bold'), bg='white', fg='#2c3e50',
                               command=self.on_experiment_select)
            rb.pack(anchor='w')
            
            desc_label = tk.Label(frame, text=desc, font=('Arial', 9), 
                                 bg='white', fg='#7f8c8d')
            desc_label.pack(anchor='w', padx=(20, 0))
        
        # Parameters panel
        self.param_frame = tk.LabelFrame(parent, text="Parameters", font=('Arial', 12, 'bold'),
                                        bg='white', fg='#34495e', padx=10, pady=10)
        self.param_frame.pack(fill='x', padx=10, pady=5)
        
        self.setup_parameters()
        
        # Control buttons
        button_frame = tk.Frame(parent, bg='white')
        button_frame.pack(fill='x', padx=10, pady=10)
        
        self.run_button = tk.Button(button_frame, text="🚀 Run Experiment", 
                                   command=self.run_experiment,
                                   font=('Arial', 12, 'bold'), bg='#27ae60', fg='white',
                                   height=2, relief='raised', bd=3)
        self.run_button.pack(fill='x', pady=2)
        
        self.stop_button = tk.Button(button_frame, text="⏹ Stop", 
                                    command=self.stop_experiment,
                                    font=('Arial', 12, 'bold'), bg='#e74c3c', fg='white',
                                    height=1, relief='raised', bd=3, state='disabled')
        self.stop_button.pack(fill='x', pady=2)
        
        button_row = tk.Frame(button_frame, bg='white')
        button_row.pack(fill='x', pady=(5, 0))
        
        self.save_button = tk.Button(button_row, text="💾 Save Results", 
                                    command=self.save_results,
                                    font=('Arial', 10), bg='#3498db', fg='white',
                                    state='disabled')
        self.save_button.pack(side='left', fill='x', expand=True, padx=(0, 2))
        
        self.clear_button = tk.Button(button_row, text="🗑 Clear", 
                                     command=self.clear_results,
                                     font=('Arial', 10), bg='#95a5a6', fg='white')
        self.clear_button.pack(side='right', fill='x', expand=True, padx=(2, 0))
        
    def setup_parameters(self):
        """Set up parameter input widgets."""
        
        # Clear existing parameters
        for widget in self.param_frame.winfo_children():
            widget.destroy()
            
        # Magnetic field
        tk.Label(self.param_frame, text="Magnetic Field (mT):", 
                font=('Arial', 10), bg='white').grid(row=0, column=0, sticky='w', pady=2)
        self.b_field_var = tk.DoubleVar(value=10.0)
        tk.Entry(self.param_frame, textvariable=self.b_field_var, width=10,
                font=('Arial', 10)).grid(row=0, column=1, sticky='w', padx=5, pady=2)
        
        # Number of measurements
        tk.Label(self.param_frame, text="Measurements:", 
                font=('Arial', 10), bg='white').grid(row=1, column=0, sticky='w', pady=2)
        self.n_measurements_var = tk.IntVar(value=100)
        tk.Entry(self.param_frame, textvariable=self.n_measurements_var, width=10,
                font=('Arial', 10)).grid(row=1, column=1, sticky='w', padx=5, pady=2)
        
        # Rabi frequency
        tk.Label(self.param_frame, text="Rabi Freq (MHz):", 
                font=('Arial', 10), bg='white').grid(row=2, column=0, sticky='w', pady=2)
        self.rabi_freq_var = tk.DoubleVar(value=15.0)
        tk.Entry(self.param_frame, textvariable=self.rabi_freq_var, width=10,
                font=('Arial', 10)).grid(row=2, column=1, sticky='w', padx=5, pady=2)
        
        # Readout time (for photons experiment)
        if self.experiment_var.get() == "photons":
            tk.Label(self.param_frame, text="Readout Time (μs):", 
                    font=('Arial', 10), bg='white').grid(row=3, column=0, sticky='w', pady=2)
            self.readout_time_var = tk.DoubleVar(value=10.0)  # 10 μs default for better statistics
            tk.Entry(self.param_frame, textvariable=self.readout_time_var, width=10,
                    font=('Arial', 10)).grid(row=3, column=1, sticky='w', padx=5, pady=2)
        
        # Enable noise checkbox
        self.noise_var = tk.BooleanVar(value=True)
        tk.Checkbutton(self.param_frame, text="Enable realistic noise", 
                      variable=self.noise_var, font=('Arial', 10), bg='white',
                      ).grid(row=4, column=0, columnspan=2, sticky='w', pady=5)
        
        # Advanced noise model selection
        tk.Label(self.param_frame, text="Noise Model:", 
                font=('Arial', 10), bg='white').grid(row=5, column=0, sticky='w', pady=2)
        self.noise_model_var = tk.StringVar(value="basic")
        noise_combo = ttk.Combobox(self.param_frame, textvariable=self.noise_model_var,
                                  values=["basic", "advanced", "precision"], width=8, state="readonly")
        noise_combo.grid(row=5, column=1, sticky='w', padx=5, pady=2)
        
        # NV type selection
        tk.Label(self.param_frame, text="NV Type:", 
                font=('Arial', 10), bg='white').grid(row=6, column=0, sticky='w', pady=2)
        self.nv_type_var = tk.StringVar(value="bulk")
        nv_combo = ttk.Combobox(self.param_frame, textvariable=self.nv_type_var,
                               values=["bulk", "surface", "nanodiamond"], width=8, state="readonly")
        nv_combo.grid(row=6, column=1, sticky='w', padx=5, pady=2)
        
    def setup_results_panel(self, parent):
        """Set up the results and visualization panel."""
        
        # Panel title
        results_title = tk.Label(parent, text="📊 Results & Visualization", 
                               font=('Arial', 16, 'bold'), bg='white', fg='#2c3e50')
        results_title.pack(pady=10)
        
        # Notebook for different result views
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Plot tab
        self.plot_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.plot_frame, text="📈 Plots")
        
        # Set up matplotlib figure
        self.figure = Figure(figsize=(8, 6), dpi=100, facecolor='white')
        self.canvas = FigureCanvasTkAgg(self.figure, self.plot_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=5, pady=5)
        
        # Log tab
        self.log_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.log_frame, text="📋 Log")
        
        # Log text widget
        log_text_frame = tk.Frame(self.log_frame, bg='white')
        log_text_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.log_text = tk.Text(log_text_frame, font=('Courier', 10), bg='#2c3e50', 
                               fg='#ecf0f1', wrap='word')
        log_scrollbar = tk.Scrollbar(log_text_frame, orient='vertical', command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scrollbar.set)
        
        self.log_text.pack(side='left', fill='both', expand=True)
        log_scrollbar.pack(side='right', fill='y')
        
        # Data tab
        self.data_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.data_frame, text="📊 Data")
        
        self.setup_data_display()
        
        # Initial welcome message
        self.add_log("🚀 QUSIM Experiment GUI Ready!")
        self.add_log("Select an experiment and click 'Run Experiment' to start.")
        self.add_log("=" * 50)
        
    def setup_data_display(self):
        """Set up data display widgets."""
        
        # Results summary
        summary_frame = tk.LabelFrame(self.data_frame, text="Experiment Summary", 
                                    font=('Arial', 12, 'bold'), bg='white', fg='#34495e')
        summary_frame.pack(fill='x', padx=10, pady=5)
        
        self.summary_text = tk.Text(summary_frame, height=8, font=('Courier', 10),
                                   bg='#ecf0f1', fg='#2c3e50', wrap='word')
        self.summary_text.pack(fill='x', padx=5, pady=5)
        
        # Key metrics
        metrics_frame = tk.LabelFrame(self.data_frame, text="Key Metrics", 
                                    font=('Arial', 12, 'bold'), bg='white', fg='#34495e')
        metrics_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.metrics_tree = ttk.Treeview(metrics_frame, columns=('Value', 'Unit'), show='tree headings')
        self.metrics_tree.heading('#0', text='Parameter')
        self.metrics_tree.heading('Value', text='Value')
        self.metrics_tree.heading('Unit', text='Unit')
        
        metrics_scrollbar = ttk.Scrollbar(metrics_frame, orient='vertical', 
                                        command=self.metrics_tree.yview)
        self.metrics_tree.configure(yscrollcommand=metrics_scrollbar.set)
        
        self.metrics_tree.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        metrics_scrollbar.pack(side='right', fill='y', pady=5)
        
    def setup_status_bar(self):
        """Set up the status bar."""
        
        self.status_frame = tk.Frame(self.root, bg='#34495e', height=30)
        self.status_frame.pack(side='bottom', fill='x')
        self.status_frame.pack_propagate(False)
        
        self.status_label = tk.Label(self.status_frame, text="Ready", 
                                   font=('Arial', 10), bg='#34495e', fg='white')
        self.status_label.pack(side='left', padx=10, pady=5)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.status_frame, variable=self.progress_var,
                                          maximum=100, length=200)
        self.progress_bar.pack(side='right', padx=10, pady=5)
        
    def on_experiment_select(self):
        """Handle experiment selection change."""
        self.setup_parameters()  # Refresh parameters based on selection
        self.add_log(f"Selected experiment: {self.experiment_var.get()}")
        
    def add_log(self, message):
        """Add a message to the log."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert('end', full_message)
        self.log_text.see('end')
        self.root.update_idletasks()
        
    def update_status(self, message):
        """Update the status bar."""
        self.status_label.config(text=message)
        self.root.update_idletasks()
        
    def update_progress(self, value):
        """Update the progress bar."""
        self.progress_var.set(value)
        self.root.update_idletasks()
        
    def run_experiment(self):
        """Run the selected experiment."""
        
        if self.is_running:
            messagebox.showwarning("Warning", "An experiment is already running!")
            return
            
        # Validate parameters
        try:
            b_field = float(self.b_field_var.get())
            n_measurements = int(self.n_measurements_var.get())
            rabi_freq = float(self.rabi_freq_var.get())
            
            if b_field <= 0 or n_measurements <= 0 or rabi_freq <= 0:
                raise ValueError("All parameters must be positive")
                
        except ValueError as e:
            messagebox.showerror("Parameter Error", f"Invalid parameter: {e}")
            return
            
        # Update GUI state
        self.is_running = True
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.save_button.config(state='disabled')
        
        # Clear previous results
        self.clear_plots()
        
        experiment_type = self.experiment_var.get()
        self.add_log(f"🚀 Starting {experiment_type} experiment...")
        self.update_status("Running experiment...")
        
        # Start experiment in separate thread
        self.experiment_thread = threading.Thread(target=self._run_experiment_thread, 
                                                 args=(experiment_type,))
        self.experiment_thread.daemon = True
        self.experiment_thread.start()
        
    def _run_experiment_thread(self, experiment_type):
        """Run experiment in separate thread to avoid blocking GUI."""
        
        try:
            # Get parameters
            b_field = self.b_field_var.get() / 1000  # Convert mT to T
            n_measurements = self.n_measurements_var.get()
            rabi_freq = self.rabi_freq_var.get() * 1e6 * 2 * np.pi  # Convert MHz to rad/s
            enable_noise = self.noise_var.get()
            
            self.add_log(f"Parameters: B={b_field*1000:.1f} mT, N={n_measurements}, Ω={rabi_freq/(2*np.pi*1e6):.1f} MHz")
            
            # Update progress
            self.update_progress(10)
            
            if experiment_type == "minimal":
                self.results = self._run_minimal_experiment(b_field, n_measurements)
                
            elif experiment_type == "photons":
                readout_time = self.readout_time_var.get() * 1e-6  # Convert μs to s
                self.results = self._run_photons_experiment(b_field, rabi_freq, readout_time)
                
            elif experiment_type == "quick":
                self.results = self._run_quick_experiment(b_field, n_measurements, enable_noise)
                
            elif experiment_type == "full":
                self.results = self._run_full_experiment(b_field, n_measurements, rabi_freq, enable_noise)
                
            self.update_progress(90)
            
            # Update GUI with results
            self.root.after(0, self._experiment_completed_gui_update)
            
        except Exception as e:
            self.add_log(f"❌ Experiment failed: {str(e)}")
            self.root.after(0, self._experiment_failed_gui_update)
            
    def _run_minimal_experiment(self, b_field, n_measurements):
        """Run the minimal experiment using real QUSIM physics."""
        
        self.add_log("Running minimal π-pulse demo with QUSIM...")
        self.update_progress(30)
        
        # Create minimal noise configuration
        noise_config = NoiseConfiguration()
        # Only enable essential noise for minimal demo
        noise_config.enable_c13_bath = True
        noise_config.enable_optical = True
        noise_config.enable_external_field = False
        noise_config.enable_johnson = False
        noise_config.enable_charge_noise = False
        noise_config.enable_temperature = False
        noise_config.enable_strain = False
        noise_config.enable_microwave = False
        
        # Create noise generator
        noise_gen = NoiseGenerator(noise_config)
        
        # Create NV system
        B_field = np.array([0, 0, b_field])
        nv_system = NVSystem(B_field=B_field, noise_gen=noise_gen)
        
        # Get parameters from system.json
        bright_rate = SYSTEM.get_noise_param('optical', 'readout', 'bright_state_rate')
        dark_rate = SYSTEM.get_noise_param('optical', 'readout', 'dark_state_rate')
        collection_eff = SYSTEM.get_empirical_param('optical_system', 'collection_efficiency')
        readout_contrast = SYSTEM.get_empirical_param('optical_system', 'readout_contrast')
        readout_time = 1e-6  # 1 μs readout
        
        # Effective rates after collection
        effective_bright = bright_rate * collection_eff
        effective_dark = dark_rate * collection_eff
        
        self.update_progress(50)
        
        # Simulate measurements
        bright_counts = np.zeros(n_measurements)
        dark_counts = np.zeros(n_measurements)
        
        for i in range(n_measurements):
            # Initial state |0⟩ (bright)
            state_pops_bright = {'ms=0': 1.0, 'ms=+1': 0.0, 'ms=-1': 0.0}
            bright_counts[i] = noise_gen.process_optical_readout(
                state_pops_bright, readout_time, n_shots=1
            )[0]
            
            # After π-pulse (dark)
            state_pops_dark = {'ms=0': 1.0 - readout_contrast, 'ms=+1': readout_contrast, 'ms=-1': 0.0}
            dark_counts[i] = noise_gen.process_optical_readout(
                state_pops_dark, readout_time, n_shots=1
            )[0]
            
            if i % (n_measurements // 10) == 0:
                self.update_progress(50 + 40 * i // n_measurements)
        
        self.update_progress(90)
        
        # Calculate metrics
        mean_bright = np.mean(bright_counts)
        mean_dark = np.mean(dark_counts)
        contrast = (mean_bright - mean_dark) / (mean_bright + mean_dark)
        threshold = (mean_bright + mean_dark) / 2
        
        correct_bright = np.sum(bright_counts > threshold)
        correct_dark = np.sum(dark_counts < threshold)
        fidelity = (correct_bright + correct_dark) / (2 * n_measurements)
        
        self.add_log(f"Contrast: {contrast:.3f}, Fidelity: {fidelity:.3f}")
        self.add_log(f"Bright: {mean_bright:.1f} counts, Dark: {mean_dark:.1f} counts")
        
        return {
            'type': 'minimal',
            'bright_counts': bright_counts,
            'dark_counts': dark_counts,
            'contrast': contrast,
            'fidelity': fidelity,
            'threshold': threshold,
            'parameters': {
                'b_field': b_field,
                'n_measurements': n_measurements,
                'bright_rate': bright_rate,
                'dark_rate': dark_rate,
                'collection_eff': collection_eff
            }
        }
        
    def _run_photons_experiment(self, b_field, rabi_freq, readout_time):
        """Run the time-resolved photon counting experiment using real QUSIM."""
        
        self.add_log("Running QUSIM time-resolved photon counting...")
        self.update_progress(30)
        
        try:
            # Create noise generator based on selected model
            noise_model = self.noise_model_var.get()
            nv_type = self.nv_type_var.get()
            
            if noise_model == "advanced":
                self.add_log(f"Using advanced noise model for {nv_type} NV")
                noise_gen = create_advanced_realistic_generator(nv_type, 300.0, True)
            elif noise_model == "precision":
                self.add_log("Using precision noise model")
                noise_gen = create_precision_experiment_generator()
            else:
                self.add_log("Using basic noise model")
                noise_gen = create_realistic_noise_generator(300.0, b_field, 10e-9)
            
            self.update_progress(40)
            
            # Create NV system
            self.add_log("Initializing NV system...")
            nv_system = NVSystem(noise_generator=noise_gen)
            
            # Calculate MW frequency using system.json value
            gamma_e = SYSTEM.get_constant('nv_center', 'gamma_e')
            mw_frequency = gamma_e * b_field
            pi_pulse_duration = np.pi / rabi_freq
            
            self.add_log(f"MW frequency: {mw_frequency/1e9:.3f} GHz")
            self.add_log(f"π-pulse duration: {pi_pulse_duration*1e9:.1f} ns")
            
            self.update_progress(50)
            
            # Setup quantum evolution
            evolution = LindLadEvo(nv_system)
            
            # Initialize in bright state |0⟩
            initial_state = nv_system.get_state(0, 0)  # |ms=0⟩
            
            self.add_log("Applying π-pulse...")
            
            # Apply π-pulse (simplified - rotate to dark state)
            evolution.reset_state(initial_state)
            evolution.apply_mw_pulse(pi_pulse_duration, rabi_freq, 0.0)
            
            self.update_progress(60)
            
            # Get final state populations
            final_state = evolution.get_current_state()
            
            # Calculate state populations (simplified)
            # In reality would need to calculate overlaps with |0⟩, |+1⟩, |-1⟩
            final_pop_0 = abs(final_state[0, 0])**2  # |0⟩ population 
            final_pop_dark = 1.0 - final_pop_0       # Dark state population
            
            self.add_log(f"Final populations: |0⟩={final_pop_0:.3f}, dark={final_pop_dark:.3f}")
            
            self.update_progress(70)
            
            # Generate photon emission trace using optical noise source
            self.add_log("Generating photon emission trace...")
            
            # Time bins for readout (adaptive based on readout time)
            if readout_time < 1e-6:  # < 1 μs
                time_bin = 50e-9     # 50 ns bins
            else:  # >= 1 μs
                time_bin = 100e-9    # 100 ns bins for longer readouts
                
            n_bins = int(readout_time / time_bin)
            time_ns = np.arange(n_bins) * time_bin * 1e9  # Convert to ns
            
            self.add_log(f"Time binning: {n_bins} bins of {time_bin*1e9:.0f} ns each")
            
            # Get emission rates from system.json
            bright_rate = SYSTEM.get_noise_param('optical', 'readout', 'bright_state_rate')
            dark_rate = SYSTEM.get_noise_param('optical', 'readout', 'dark_state_rate')
            detector_dark = SYSTEM.get_empirical_param('optical_system', 'detector_dark_counts')
            
            # Get collection efficiency from empirical parameters
            collection_eff = SYSTEM.get_empirical_param('optical_system', 'collection_efficiency')
            detector_eff = SYSTEM.get_noise_param('optical', 'readout', 'detector_efficiency')
            total_eff = collection_eff * detector_eff
            
            # Effective detected rates
            effective_bright = bright_rate * total_eff
            effective_dark = dark_rate * total_eff
            effective_bg = detector_dark
            
            self.add_log(f"Detection rates: bright={effective_bright/1e6:.1f} Mcps, dark={effective_dark/1e3:.0f} kcps")
            
            # Process optical readout with noise
            state_populations = {
                'ms=0': final_pop_0,
                'ms=+1': final_pop_dark / 2,
                'ms=-1': final_pop_dark / 2
            }
            
            self.update_progress(80)
            
            # Generate time-resolved photon counts using noise generator
            photon_counts = np.zeros(n_bins)
            
            # Use noise generator to simulate readout for each time bin
            for i in range(n_bins):
                # Process one time bin with proper noise
                bin_counts = noise_gen.process_optical_readout(
                    state_populations, 
                    time_bin,
                    n_shots=1
                )[0]
                photon_counts[i] = bin_counts
                
                if i % (n_bins // 10) == 0 and n_bins > 10:
                    self.update_progress(80 + 10 * i // n_bins)
            
            total_photons = np.sum(photon_counts)
            average_rate = total_photons / readout_time
            
            self.add_log(f"Total photons: {total_photons:.0f}")
            self.add_log(f"Average rate: {average_rate/1e6:.3f} Mcps")
            
            return {
                'type': 'photons',
                'time_ns': time_ns,
                'photon_counts': photon_counts,
                'total_photons': total_photons,
                'average_rate': average_rate,
                'mw_frequency': mw_frequency,
                'pi_pulse_duration': pi_pulse_duration,
                'final_pop_0': final_pop_0,
                'final_pop_dark': final_pop_dark,
                'noise_model': noise_model,
                'nv_type': nv_type,
                'parameters': {
                    'b_field': b_field,
                    'readout_time': readout_time,
                    'rabi_freq': rabi_freq
                }
            }
            
        except Exception as e:
            self.add_log(f"QUSIM photon experiment fehlgeschlagen: {e}")
            raise RuntimeError(f"Photonen-Experiment konnte nicht durchgeführt werden: {e}")
    
    def _run_quick_experiment(self, b_field, n_measurements, enable_noise):
        """Run the quick experiment using QUSIM."""
        
        self.add_log("Running QUSIM quick π-pulse experiment...")
        self.update_progress(30)
        
        try:
            # Create noise generator
            if enable_noise:
                noise_model = self.noise_model_var.get()
                nv_type = self.nv_type_var.get()
                
                if noise_model == "advanced":
                    noise_gen = create_advanced_realistic_generator(nv_type, 300.0, True)
                elif noise_model == "precision":
                    noise_gen = create_precision_experiment_generator()
                else:
                    noise_gen = create_realistic_noise_generator(300.0, b_field, 10e-9)
                
                self.add_log(f"Using {noise_model} noise model for {nv_type} NV")
            else:
                # Minimal realistic noise configuration
                noise_config = NoiseConfiguration()
                noise_config.enable_c13_bath = True
                noise_config.enable_optical = True
                # Disable controllable noise sources
                noise_config.enable_external_field = False  # Magnetic shielding
                noise_config.enable_johnson = False         # Room temperature operation
                noise_config.enable_charge_noise = False    # Good surface
                noise_config.enable_temperature = False     # Stable temperature
                noise_config.enable_strain = False          # Stress-free mounting
                noise_config.enable_microwave = False       # High-quality MW source
                
                noise_gen = NoiseGenerator(noise_config)
                self.add_log("Using minimal realistic noise configuration")
            
            # Create NV system
            nv_system = NVSystem(noise_generator=noise_gen)
            evolution = LindLadEvo(nv_system)
            
            self.update_progress(50)
            
            # Run measurements
            bright_counts = []
            dark_counts = []
            
            readout_time = 1e-6  # 1 μs readout
            
            for i in range(n_measurements):
                if i % 10 == 0:
                    progress = 50 + (i / n_measurements) * 30
                    self.update_progress(progress)
                
                # Bright measurement (no pulse)
                evolution.reset_state(nv_system.get_state(0, 0))  # |ms=0⟩
                
                state_pops = {'ms=0': 1.0, 'ms=+1': 0.0, 'ms=-1': 0.0}
                bright_count = noise_gen.process_optical_readout(state_pops, readout_time, 1)[0]
                bright_counts.append(bright_count)
                
                # Dark measurement (after π-pulse)
                evolution.reset_state(nv_system.get_state(0, 0))
                pi_pulse_duration = np.pi / (15e6 * 2 * np.pi)  # 15 MHz Rabi
                evolution.apply_mw_pulse(pi_pulse_duration, 15e6 * 2 * np.pi, 0.0)
                
                final_state = evolution.get_current_state()
                pop_0 = abs(final_state[0, 0])**2
                pop_dark = 1.0 - pop_0
                
                state_pops = {'ms=0': pop_0, 'ms=+1': pop_dark/2, 'ms=-1': pop_dark/2}
                dark_count = noise_gen.process_optical_readout(state_pops, readout_time, 1)[0]
                dark_counts.append(dark_count)
            
            bright_counts = np.array(bright_counts)
            dark_counts = np.array(dark_counts)
            
            contrast = (np.mean(bright_counts) - np.mean(dark_counts)) / (np.mean(bright_counts) + np.mean(dark_counts))
            
            self.update_progress(90)
            self.add_log(f"QUSIM quick experiment completed. Contrast: {contrast:.3f}")
            
            return {
                'type': 'quick',
                'bright_counts': bright_counts,
                'dark_counts': dark_counts,
                'contrast': contrast,
                'noise_model': noise_model if enable_noise else 'minimal',
                'nv_type': nv_type if enable_noise else 'bulk',
                'parameters': {
                    'b_field': b_field,
                    'n_measurements': n_measurements,
                    'enable_noise': enable_noise
                }
            }
            
        except Exception as e:
            self.add_log(f"QUSIM simulation fehlgeschlagen: {e}")
            raise RuntimeError(f"Experiment konnte nicht durchgeführt werden: {e}")
    
    def _run_full_experiment(self, b_field, n_measurements, rabi_freq, enable_noise):
        """Run the full experiment using complete QUSIM simulation."""
        
        self.add_log("Running full π-pulse readout experiment with QUSIM...")
        self.update_progress(20)
        
        # Create noise configuration based on settings
        if enable_noise:
            noise_model = self.noise_model_var.get()
            nv_type = self.nv_type_var.get()
            
            if noise_model == "advanced":
                noise_gen = create_advanced_realistic_generator(nv_type, 300.0, True)
            elif noise_model == "precision":
                noise_gen = create_precision_experiment_generator()
            else:
                noise_gen = create_realistic_noise_generator(300.0, b_field, 10e-9)
                
            self.add_log(f"Using {noise_model} noise model for {nv_type} NV")
        else:
            # Minimal noise configuration
            noise_config = NoiseConfiguration()
            noise_config.enable_c13_bath = True
            noise_config.enable_optical = True
            # Disable all other noise sources
            noise_config.enable_external_field = False
            noise_config.enable_johnson = False
            noise_config.enable_charge_noise = False
            noise_config.enable_temperature = False
            noise_config.enable_strain = False
            noise_config.enable_microwave = False
            noise_gen = NoiseGenerator(noise_config)
            self.add_log("Using minimal noise configuration")
        
        # Create NV system
        B_field = np.array([0, 0, b_field])
        nv_system = NVSystem(B_field=B_field, noise_gen=noise_gen)
        
        self.update_progress(30)
        
        # Initialize measurements arrays
        bright_counts = np.zeros(n_measurements)
        dark_counts = np.zeros(n_measurements)
        
        # Get readout parameters from system.json
        readout_time = 1e-6  # 1 μs readout
        readout_contrast = SYSTEM.get_empirical_param('optical_system', 'readout_contrast')
        
        # Calculate π-pulse duration
        pi_pulse_duration = np.pi / (2 * rabi_freq)
        self.add_log(f"π-pulse duration: {pi_pulse_duration*1e9:.1f} ns")
        
        self.update_progress(40)
        
        # Run measurements
        for i in range(n_measurements):
            if i % (n_measurements // 10) == 0:
                progress = 40 + 40 * i // n_measurements
                self.update_progress(progress)
            
            # Bright measurement (no pulse)
            state_pops_bright = {'ms=0': 1.0, 'ms=+1': 0.0, 'ms=-1': 0.0}
            bright_counts[i] = noise_gen.process_optical_readout(
                state_pops_bright, readout_time, n_shots=1
            )[0]
            
            # Dark measurement (after π-pulse)
            # Simulate population transfer with realistic contrast
            state_pops_dark = {
                'ms=0': 1.0 - readout_contrast, 
                'ms=+1': readout_contrast * 0.9,  # Most goes to |+1⟩
                'ms=-1': readout_contrast * 0.1   # Small amount to |-1⟩
            }
            dark_counts[i] = noise_gen.process_optical_readout(
                state_pops_dark, readout_time, n_shots=1
            )[0]
        
        self.update_progress(80)
        
        # Calculate metrics
        mean_bright = np.mean(bright_counts)
        mean_dark = np.mean(dark_counts)
        contrast = (mean_bright - mean_dark) / (mean_bright + mean_dark)
        
        # Calculate threshold and fidelity
        threshold = (mean_bright + mean_dark) / 2
        correct_bright = np.sum(bright_counts > threshold)
        correct_dark = np.sum(dark_counts < threshold)
        fidelity = (correct_bright + correct_dark) / (2 * n_measurements)
        
        # Estimate T2* from noise generator
        t2_star = noise_gen.estimate_t2_star(evolution_time=10e-6, n_samples=1000)
        
        self.update_progress(90)
        
        self.add_log(f"Full experiment completed.")
        self.add_log(f"Bright: {mean_bright:.1f} counts, Dark: {mean_dark:.1f} counts")
        self.add_log(f"Contrast: {contrast:.3f}, Fidelity: {fidelity:.3f}")
        self.add_log(f"T2*: {t2_star*1e6:.1f} μs")
        
        return {
            'type': 'full',
            'bright_counts': bright_counts,
            'dark_counts': dark_counts,
            'contrast': contrast,
            'fidelity': fidelity,
            'threshold': threshold,
            't2_star': t2_star,
            'parameters': {
                'b_field': b_field,
                'n_measurements': n_measurements,
                'rabi_freq': rabi_freq,
                'enable_noise': enable_noise,
                'noise_model': noise_model if enable_noise else 'minimal',
                'nv_type': nv_type if enable_noise else 'bulk'
            }
        }
        
    def _experiment_completed_gui_update(self):
        """Update GUI after experiment completion."""
        
        self.is_running = False
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.save_button.config(state='normal')
        
        self.update_status("Experiment completed")
        self.update_progress(100)
        self.add_log("✅ Experiment completed successfully!")
        
        # Plot results
        self.plot_results()
        self.update_data_display()
        
        # Auto-switch to plots tab
        self.notebook.select(0)
        
    def _experiment_failed_gui_update(self):
        """Update GUI after experiment failure."""
        
        self.is_running = False
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        self.update_status("Experiment failed")
        self.update_progress(0)
        
    def stop_experiment(self):
        """Stop the running experiment."""
        
        if self.experiment_thread and self.experiment_thread.is_alive():
            # Note: In a real implementation, you'd need proper thread termination
            self.add_log("⏹ Stopping experiment...")
            
        self.is_running = False
        self.run_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.update_status("Stopped")
        self.update_progress(0)
        
    def plot_results(self):
        """Plot the experimental results."""
        
        if not self.results:
            return
            
        self.figure.clear()
        
        exp_type = self.results['type']
        
        if exp_type == 'photons':
            self.plot_photon_results()
        else:
            self.plot_standard_results()
            
        self.canvas.draw()
        
    def plot_photon_results(self):
        """Plot time-resolved photon counting results."""
        
        # Create 2x2 subplot layout
        gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        time_ns = self.results['time_ns']
        counts = self.results['photon_counts']
        
        # Main photon trace
        ax1 = self.figure.add_subplot(gs[0, :])
        ax1.plot(time_ns, counts, 'b-', linewidth=0.8, alpha=0.8)
        ax1.set_xlabel('Time (ns)')
        ax1.set_ylabel('Photon Counts/ns')
        ax1.set_title('Time-Resolved Photon Emission')
        ax1.grid(True, alpha=0.3)
        
        # Add average line
        avg_counts = self.results['average_rate'] * 1e-9
        ax1.axhline(avg_counts, color='red', linestyle='--', 
                   label=f'Average: {avg_counts:.4f} counts/ns')
        ax1.legend()
        
        # Zoomed view (first 100 ns)
        ax2 = self.figure.add_subplot(gs[1, 0])
        mask = time_ns <= 100
        ax2.plot(time_ns[mask], counts[mask], 'ro-', markersize=2, linewidth=1)
        ax2.set_xlabel('Time (ns)')
        ax2.set_ylabel('Counts/ns')
        ax2.set_title('First 100 ns')
        ax2.grid(True, alpha=0.3)
        
        # Cumulative counts
        ax3 = self.figure.add_subplot(gs[1, 1])
        cumulative = np.cumsum(counts)
        ax3.plot(time_ns, cumulative, 'g-', linewidth=2)
        ax3.set_xlabel('Time (ns)')
        ax3.set_ylabel('Cumulative Counts')
        ax3.set_title('Cumulative Photons')
        ax3.grid(True, alpha=0.3)
        
        self.figure.suptitle('Time-Resolved Photon Counting Results', fontsize=14)
        
    def plot_standard_results(self):
        """Plot standard experimental results (histograms, etc.)."""
        
        # Create 2x2 subplot layout
        gs = self.figure.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        bright_counts = self.results['bright_counts']
        dark_counts = self.results['dark_counts']
        
        # Photon count histograms
        ax1 = self.figure.add_subplot(gs[0, 0])
        bins = np.linspace(min(np.min(bright_counts), np.min(dark_counts)), 
                          max(np.max(bright_counts), np.max(dark_counts)), 25)
        
        ax1.hist(bright_counts, bins=bins, alpha=0.7, label='Bright (|0⟩)', color='orange', density=True)
        ax1.hist(dark_counts, bins=bins, alpha=0.7, label='Dark (|±1⟩)', color='blue', density=True)
        
        if 'threshold' in self.results:
            ax1.axvline(self.results['threshold'], color='red', linestyle='--', label='Threshold')
            
        ax1.set_xlabel('Photon Counts')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Photon Count Distributions')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Time traces (first 50 measurements)
        ax2 = self.figure.add_subplot(gs[0, 1])
        n_show = min(50, len(bright_counts))
        x = np.arange(n_show)
        ax2.plot(x, bright_counts[:n_show], 'o-', color='orange', alpha=0.7, label='Bright')
        ax2.plot(x, dark_counts[:n_show], 's-', color='blue', alpha=0.7, label='Dark')
        
        if 'threshold' in self.results:
            ax2.axhline(self.results['threshold'], color='red', linestyle='--', label='Threshold')
            
        ax2.set_xlabel('Measurement #')
        ax2.set_ylabel('Photon Counts')
        ax2.set_title(f'Single-Shot Traces (first {n_show})')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Contrast and fidelity bar chart
        ax3 = self.figure.add_subplot(gs[1, 0])
        metrics = ['Contrast']
        values = [self.results['contrast']]
        
        if 'fidelity' in self.results:
            metrics.append('Fidelity')
            values.append(self.results['fidelity'])
            
        bars = ax3.bar(metrics, values, color=['green', 'purple'][:len(metrics)], alpha=0.7)
        ax3.set_ylabel('Value')
        ax3.set_title('Key Metrics')
        ax3.set_ylim(0, 1)
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Statistics summary
        ax4 = self.figure.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        # Create summary text
        summary_lines = [
            f"Measurements: {len(bright_counts)}",
            f"Bright mean: {np.mean(bright_counts):.1f} ± {np.std(bright_counts):.1f}",
            f"Dark mean: {np.mean(dark_counts):.1f} ± {np.std(dark_counts):.1f}",
            f"Contrast: {self.results['contrast']:.3f}",
        ]
        
        if 'fidelity' in self.results:
            summary_lines.append(f"Fidelity: {self.results['fidelity']:.3f}")
            
        if 't2_star' in self.results:
            summary_lines.append(f"T2*: {self.results['t2_star']*1e6:.1f} μs")
            
        summary_text = '\n'.join(summary_lines)
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        self.figure.suptitle(f'{self.results["type"].capitalize()} Experiment Results', fontsize=14)
        
    def update_data_display(self):
        """Update the data display tab."""
        
        if not self.results:
            return
            
        # Clear existing data
        self.summary_text.delete(1.0, 'end')
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
            
        # Update summary
        exp_type = self.results['type']
        params = self.results['parameters']
        
        summary_lines = [
            f"Experiment Type: {exp_type.capitalize()}",
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "Parameters:",
            f"  Magnetic Field: {params['b_field']*1000:.1f} mT",
        ]
        
        if 'n_measurements' in params:
            summary_lines.append(f"  Measurements: {params['n_measurements']}")
            
        if 'rabi_freq' in params:
            summary_lines.append(f"  Rabi Frequency: {params['rabi_freq']/(2*np.pi*1e6):.1f} MHz")
            
        if 'readout_time' in params:
            if params['readout_time'] >= 1e-6:
                summary_lines.append(f"  Readout Time: {params['readout_time']*1e6:.1f} μs")
            else:
                summary_lines.append(f"  Readout Time: {params['readout_time']*1e9:.0f} ns")
            
        summary_lines.extend([
            "",
            "Results:",
        ])
        
        if exp_type == 'photons':
            summary_lines.extend([
                f"  Total Photons: {self.results['total_photons']:.0f}",
                f"  Average Rate: {self.results['average_rate']/1e6:.3f} Mcps",
                f"  MW Frequency: {self.results['mw_frequency']/1e9:.3f} GHz",
                f"  π-Pulse Duration: {self.results['pi_pulse_duration']*1e9:.1f} ns",
            ])
            if 'final_pop_0' in self.results:
                summary_lines.extend([
                    f"  Final |0⟩ population: {self.results['final_pop_0']:.3f}",
                    f"  Final dark population: {self.results['final_pop_dark']:.3f}",
                ])
        else:
            summary_lines.extend([
                f"  Contrast: {self.results['contrast']:.3f}",
            ])
            
        # Add noise model information
        if 'noise_model' in self.results:
            summary_lines.append(f"  Noise Model: {self.results['noise_model']}")
        if 'nv_type' in self.results:
            summary_lines.append(f"  NV Type: {self.results['nv_type']}")
            
            if 'fidelity' in self.results:
                summary_lines.append(f"  Fidelity: {self.results['fidelity']:.3f}")
                
            if 't2_star' in self.results:
                summary_lines.append(f"  T2*: {self.results['t2_star']*1e6:.1f} μs")
        
        self.summary_text.insert(1.0, '\n'.join(summary_lines))
        
        # Update metrics tree
        if exp_type == 'photons':
            metrics_data = [
                ("Total Photons", f"{self.results['total_photons']:.0f}", "counts"),
                ("Average Rate", f"{self.results['average_rate']/1e6:.3f}", "Mcps"),
                ("MW Frequency", f"{self.results['mw_frequency']/1e9:.3f}", "GHz"),
                ("π-Pulse Duration", f"{self.results['pi_pulse_duration']*1e9:.1f}", "ns"),
            ]
        else:
            metrics_data = [
                ("Contrast", f"{self.results['contrast']:.3f}", ""),
                ("Bright Mean", f"{np.mean(self.results['bright_counts']):.1f}", "counts"),
                ("Dark Mean", f"{np.mean(self.results['dark_counts']):.1f}", "counts"),
            ]
            
            if 'fidelity' in self.results:
                metrics_data.append(("Fidelity", f"{self.results['fidelity']:.3f}", ""))
                
            if 't2_star' in self.results:
                metrics_data.append(("T2*", f"{self.results['t2_star']*1e6:.1f}", "μs"))
        
        for param, value, unit in metrics_data:
            self.metrics_tree.insert('', 'end', text=param, values=(value, unit))
            
    def clear_results(self):
        """Clear all results and plots."""
        
        self.results = {}
        self.clear_plots()
        
        # Clear data display
        self.summary_text.delete(1.0, 'end')
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
            
        self.save_button.config(state='disabled')
        self.add_log("🗑 Results cleared")
        self.update_status("Ready")
        
    def clear_plots(self):
        """Clear the plots."""
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, 'No results to display\n\nRun an experiment to see plots here', 
               ha='center', va='center', transform=ax.transAxes, fontsize=14,
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        self.canvas.draw()
        
    def save_results(self):
        """Save the experimental results."""
        
        if not self.results:
            messagebox.showwarning("Warning", "No results to save!")
            return
            
        # Ask user for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".npz",
            filetypes=[("NumPy Archive", "*.npz"), ("All files", "*.*")],
            title="Save Experiment Results"
        )
        
        if filename:
            try:
                # Save results as NumPy archive
                np.savez_compressed(filename, **self.results)
                self.add_log(f"💾 Results saved to: {filename}")
                messagebox.showinfo("Success", f"Results saved to:\n{filename}")
                
            except Exception as e:
                self.add_log(f"❌ Save failed: {str(e)}")
                messagebox.showerror("Save Error", f"Failed to save results:\n{str(e)}")


def main():
    """Main function to run the GUI."""
    
    # Create the main window
    root = tk.Tk()
    
    # Create the GUI application
    app = QUSIMExperimentGUI(root)
    
    # Start the GUI event loop
    root.mainloop()


if __name__ == "__main__":
    main()