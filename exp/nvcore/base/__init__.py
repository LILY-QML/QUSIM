"""
QUSIM Base Architecture - Fallback-Impossible Physics Engine

This module contains the fundamental base classes that make fallbacks
architecturally impossible in QUSIM.

Author: Leon Kaiser
Institution: Goethe University Frankfurt, MSQC
"""

from .physics_engine import (
    NoFallbackPhysicsEngine,
    FallbackViolationError,
    PhysicsValidator,
    FallbackDetector
)

__all__ = [
    'NoFallbackPhysicsEngine',
    'FallbackViolationError', 
    'PhysicsValidator',
    'FallbackDetector'
]