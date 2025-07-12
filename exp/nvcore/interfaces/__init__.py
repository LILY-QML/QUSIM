"""
Interfaces for modular QUSIM architecture
"""

from .noise_interface import NoiseInterface, NoiseGeneratorAdapter

__all__ = ['NoiseInterface', 'NoiseGeneratorAdapter']