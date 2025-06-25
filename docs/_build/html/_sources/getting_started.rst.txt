Getting Started
===============

This guide will help you get up and running with QUSIM for NV center quantum simulations.

Installation
------------

Prerequisites
~~~~~~~~~~~~~

QUSIM requires Python 3.8 or later and the following packages:

* NumPy (≥1.20.0)
* SciPy (≥1.7.0)
* Matplotlib (≥3.3.0)

Optional dependencies:

* Sphinx (for building documentation)
* pytest (for running tests)

Setup
~~~~~

1. Clone or download the QUSIM package
2. Navigate to the QUSIM directory
3. Install dependencies:

.. code-block:: bash

   pip install numpy scipy matplotlib

Or if you have a requirements.txt file:

.. code-block:: bash

   pip install -r requirements.txt

Basic Concepts
--------------

NV Center Physics
~~~~~~~~~~~~~~~~~

The nitrogen-vacancy (NV) center is a point defect in diamond consisting of a nitrogen atom 
adjacent to a vacancy. The electronic ground state is a spin-1 triplet with the following 
key properties:

* **Zero-field splitting**: D ≈ 2.87 GHz separates the ms=0 from ms=±1 states
* **Zeeman effect**: Linear response to magnetic fields with γe ≈ 28 GHz/T
* **Optical transitions**: Can be initialized and read out optically
* **Long coherence times**: T2* ~ 1-100 μs, T2 ~ 1 ms, T1 ~ 1-10 ms

Noise Sources
~~~~~~~~~~~~~

QUSIM models various environmental noise sources:

* **C13 nuclear bath**: Magnetic noise from ¹³C nuclear spins
* **Charge noise**: Electric field fluctuations affecting strain
* **Temperature fluctuations**: Thermal noise in the diamond lattice
* **Johnson noise**: Electronic noise from measurement apparatus
* **Magnetic field noise**: External field fluctuations

Your First Simulation
---------------------

Let's start with a simple example to simulate the evolution of an NV center:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from nvcore import NVSystem, NoiseConfiguration

   # Create a noise configuration
   config = NoiseConfiguration()
   config.enable_c13_bath = True    # Enable nuclear spin bath
   config.enable_charge_noise = True  # Enable charge noise
   
   # Create NV system with 10 mT magnetic field along z
   B_field = np.array([0, 0, 0.01])  # Tesla
   nv = NVSystem(B_field=B_field, noise_config=config)
   
   # Prepare initial state (ms=0)
   rho0 = nv.create_initial_state('ms0')
   
   # Evolve for 1 microsecond
   times, rhos = nv.evolve(rho0, (0, 1e-6))
   
   # Extract populations
   populations = []
   for rho in rhos:
       pops = nv.get_state_populations(rho)
       populations.append([pops['ms-1'], pops['ms0'], pops['ms+1']])
   
   populations = np.array(populations)
   
   # Plot results
   plt.figure(figsize=(10, 6))
   plt.plot(times*1e6, populations[:, 0], label='ms=-1')
   plt.plot(times*1e6, populations[:, 1], label='ms=0')
   plt.plot(times*1e6, populations[:, 2], label='ms=+1')
   plt.xlabel('Time (μs)')
   plt.ylabel('Population')
   plt.legend()
   plt.title('NV Center Evolution')
   plt.show()

Understanding the Results
~~~~~~~~~~~~~~~~~~~~~~~~~

The simulation shows how the NV center evolves under the influence of its environment:

1. **Population dynamics**: How probability flows between the three spin states
2. **Decoherence**: Loss of quantum coherence due to noise
3. **Relaxation**: Population redistribution toward thermal equilibrium

Pulse Sequences
---------------

QUSIM supports arbitrary microwave pulse sequences. Here's a Ramsey interferometry example:

.. code-block:: python

   # Create Ramsey sequence: π/2 - wait - π/2
   ramsey_sequence = nv.create_ramsey_sequence(T_ramsey=5e-6)  # 5 μs wait time
   
   # Apply sequence
   times, rhos = nv.evolve_with_pulses(rho0, ramsey_sequence)
   
   # Measure final coherence
   final_rho = rhos[-1]
   coherence = nv.measure_observable(final_rho, nv.hamiltonian.spin_ops.Sx)
   print(f"Final coherence: {coherence:.3f}")

T2* Measurement
~~~~~~~~~~~~~~~

Simulate a complete T2* measurement:

.. code-block:: python

   # Scan Ramsey wait times
   tau_values, coherences = nv.simulate_t2_measurement(
       tau_max=10e-6,     # Maximum wait time
       n_points=50,       # Number of points
       sequence='ramsey'  # Use Ramsey sequence
   )
   
   # Fit exponential decay
   from scipy.optimize import curve_fit
   
   def exp_decay(t, A, T2_star):
       return A * np.exp(-t / T2_star)
   
   popt, _ = curve_fit(exp_decay, tau_values, np.abs(coherences))
   T2_star_fitted = popt[1]
   
   print(f"Fitted T2*: {T2_star_fitted*1e6:.1f} μs")

Performance Considerations
--------------------------

Fast Mode
~~~~~~~~~

For large parameter sweeps or long simulations, use the fast mode:

.. code-block:: python

   from nvcore_fast import FastNVSystem
   
   # Create fast system (simplified physics for speed)
   fast_nv = FastNVSystem(B_field=B_field, enable_noise=True)
   
   # Same interface as regular system
   times, rhos = fast_nv.evolve_unitary(rho0, (0, 1e-6))

The fast mode provides:

* 2-10x speedup for typical simulations
* Reduced memory usage
* Simplified noise models
* Same basic physics

When to Use Each Mode
~~~~~~~~~~~~~~~~~~~~~

**Regular mode** for:
* High-accuracy simulations
* Detailed noise analysis
* Research applications
* Method development

**Fast mode** for:
* Parameter sweeps
* Real-time applications
* Educational demonstrations
* Preliminary studies

Next Steps
----------

Now that you've run your first simulation, explore:

* :doc:`tutorial` - Detailed tutorials for common experiments
* :doc:`examples` - Gallery of example simulations
* :doc:`api_reference` - Complete API documentation
* :doc:`performance` - Performance optimization guide

Command Line Interface
----------------------

QUSIM also provides a command-line interface through `core.py`:

.. code-block:: bash

   # Run fast simulation
   python core.py --fast --time 1e-6
   
   # Ramsey sequence with specific noise
   python core.py --demo --pulse-sequence ramsey --noise c13_bath
   
   # Generate plots
   python core.py --time 1e-6 --plot --output results/

See the :doc:`api/core` documentation for all available options.

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import errors**: Make sure all dependencies are installed and the QUSIM directory is in your Python path.

**Slow performance**: Try the fast mode or reduce the simulation time/resolution.

**Memory usage**: For long simulations, avoid saving the full trajectory with `--save-trajectory`.

**Numerical errors**: Check that your initial state is a valid density matrix (Hermitian, trace=1).

Getting Help
~~~~~~~~~~~~

If you encounter issues:

1. Check the examples in this documentation
2. Review the API reference for proper function usage
3. Run the test suite to verify your installation
4. Submit issues to the project repository

The test suite can be run with:

.. code-block:: bash

   python test.py