API Reference
=============

This section provides detailed documentation of all QUSIM modules, classes, and functions.

Core Module
-----------

.. automodule:: core
   :members:
   :undoc-members:
   :show-inheritance:

Main NV System
--------------

.. automodule:: nvcore
   :members:
   :undoc-members:
   :show-inheritance:

Spin Operators
~~~~~~~~~~~~~~

.. autoclass:: nvcore.NVSpinOperators
   :members:
   :undoc-members:
   :show-inheritance:

System Hamiltonian
~~~~~~~~~~~~~~~~~~

.. autoclass:: nvcore.NVSystemHamiltonian
   :members:
   :undoc-members:
   :show-inheritance:

Lindblad Evolution
~~~~~~~~~~~~~~~~~~

.. autoclass:: nvcore.NVLindblad
   :members:
   :undoc-members:
   :show-inheritance:

Complete NV System
~~~~~~~~~~~~~~~~~~

.. autoclass:: nvcore.NVSystem
   :members:
   :undoc-members:
   :show-inheritance:

Fast NV System
~~~~~~~~~~~~~~

.. automodule:: nvcore_fast
   :members:
   :undoc-members:
   :show-inheritance:

Noise Modeling
--------------

.. automodule:: noise
   :members:
   :undoc-members:
   :show-inheritance:

Noise Configuration
~~~~~~~~~~~~~~~~~~~

.. autoclass:: noise.NoiseConfiguration
   :members:
   :undoc-members:
   :show-inheritance:

Noise Generator
~~~~~~~~~~~~~~~

.. autoclass:: noise.NoiseGenerator
   :members:
   :undoc-members:
   :show-inheritance:

System Constants
----------------

.. automodule:: noise_sources
   :members:
   :undoc-members:
   :show-inheritance:

Test Suite
----------

.. automodule:: test
   :members:
   :undoc-members:
   :show-inheritance:

Convenience Functions
--------------------

Room Temperature NV
~~~~~~~~~~~~~~~~~~~

.. autofunction:: nvcore.create_room_temperature_nv

Cryogenic NV
~~~~~~~~~~~~

.. autofunction:: nvcore.create_cryogenic_nv

Low Noise NV
~~~~~~~~~~~~

.. autofunction:: nvcore.create_low_noise_nv