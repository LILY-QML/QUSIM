"""
QUSIM Noise Architecture - Type-Safe, Fallback-Impossible Noise Sources

This module implements a revolutionary noise architecture that makes it
impossible to return wrong types or fallback values from noise sources.

Author: Leon Kaiser  
Institution: Goethe University Frankfurt, MSQC
"""

from .base_sources import (
    NoiseSourceType,
    TypedNoiseSource,
    PhysicsNoiseInterface
)

from .typed_sources import (
    MagneticNoiseSource,
    ChargeStateNoiseSource, 
    ThermalNoiseSource,
    StrainNoiseSource,
    OpticalNoiseSource
)

__all__ = [
    'NoiseSourceType',
    'TypedNoiseSource', 
    'PhysicsNoiseInterface',
    'MagneticNoiseSource',
    'ChargeStateNoiseSource',
    'ThermalNoiseSource', 
    'StrainNoiseSource',
    'OpticalNoiseSource'
]