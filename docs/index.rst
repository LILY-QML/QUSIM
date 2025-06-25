QUSIM: Quantum Simulation of NV Centers
=======================================

Welcome to QUSIM, a comprehensive quantum simulation package for nitrogen-vacancy (NV) center dynamics. 
QUSIM provides physically accurate modeling of NV centers including noise sources, open quantum system evolution, 
and experimental protocols commonly used in quantum sensing and quantum information processing.

.. image:: https://img.shields.io/badge/python-3.8%2B-blue
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green
   :target: https://opensource.org/licenses/MIT
   :alt: License

Features
--------

* **Complete NV Center Physics**: Zero-field splitting, Zeeman effect, strain interactions
* **Comprehensive Noise Modeling**: C13 nuclear bath, charge noise, temperature fluctuations, and more
* **Open Quantum System Evolution**: Lindblad master equation with adaptive integration
* **Pulse Sequence Control**: Support for arbitrary microwave pulse sequences
* **Common Protocols**: Built-in Ramsey, echo, CPMG, and Rabi sequences
* **Performance Optimized**: Fast simulation mode for large-scale studies
* **Extensive Documentation**: Detailed API reference and examples

Quick Start
-----------

Basic usage example:

.. code-block:: python

   import numpy as np
   from nvcore import create_room_temperature_nv
   
   # Create NV system with realistic noise
   nv = create_room_temperature_nv(B_field=[0, 0, 0.01])  # 10 mT field
   
   # Prepare initial state
   rho0 = nv.create_initial_state('ms0')
   
   # Simulate T2* measurement
   tau_values, coherences = nv.simulate_t2_measurement()
   
   # Visualize evolution
   nv.visualize_evolution(rho0, t_max=1e-6)

Installation
------------

Clone the repository and install dependencies:

.. code-block:: bash

   git clone <repository-url>
   cd QUSIM
   pip install -r requirements.txt

Requirements:
* Python 3.8+
* NumPy
* SciPy
* Matplotlib
* Optional: Sphinx (for documentation)

User Guide
----------

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   tutorial
   examples
   api_reference
   performance
   development

API Reference
-------------

.. toctree::
   :maxdepth: 3
   :caption: API Documentation:

   api/core
   api/nvcore
   api/noise
   api/utils

Examples
--------

.. toctree::
   :maxdepth: 2
   :caption: Example Gallery:

   examples/basic_simulation
   examples/pulse_sequences
   examples/noise_analysis
   examples/parameter_studies
   examples/advanced_protocols

Performance Guide
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Performance:

   performance/optimization
   performance/benchmarks
   performance/scaling

Development
-----------

.. toctree::
   :maxdepth: 2
   :caption: Development:

   development/contributing
   development/testing
   development/architecture

Citation
--------

If you use QUSIM in your research, please cite:

.. code-block:: bibtex

   @software{qusim2024,
     title={QUSIM: Quantum Simulation of NV Centers},
     author={QUSIM Development Team},
     year={2024},
     url={https://github.com/your-org/qusim}
   }

License
-------

QUSIM is released under the MIT License. See LICENSE file for details.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`