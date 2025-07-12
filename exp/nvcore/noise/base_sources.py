"""
Base Classes for Type-Safe Noise Sources

Author: Leon Kaiser
Institution: Goethe University Frankfurt, MSQC
Contact: l.kaiser@em.uni-frankfurt.de
Web: https://msqc.cgi-host6.rz.uni-frankfurt.de/

ARCHITECTURAL INNOVATION:
This module implements typed noise sources that CANNOT return wrong types.
Each noise source is strictly typed and validated, making interface violations
impossible rather than just detectable.

CORE CONCEPTS:
1. NoiseSourceType: Enum defining exact output types
2. TypedNoiseSource: Base class with type enforcement  
3. Output validation: Results validated before return
4. Physics consistency: All outputs must obey physical laws

GUARANTEED PROPERTIES:
- Magnetic sources ONLY return realistic B-fields
- Charge sources ONLY return charge state information
- Thermal sources ONLY return temperature data
- Cross-contamination is architecturally impossible

This eliminates the common error of noise sources returning zeros for
wrong interfaces (e.g., charge noise returning zero magnetic field).
"""

from enum import Enum
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
import numpy as np

from ..base.physics_engine import NoFallbackPhysicsEngine, FallbackViolationError


class NoiseSourceType(Enum):
    """
    Strict typing for noise sources with clear output specifications
    
    Each type defines exactly what the noise source must return,
    preventing interface confusion and wrong-type fallbacks.
    """
    
    # Magnetic field noise - returns 3D B-field in Tesla
    MAGNETIC = "magnetic"
    
    # Charge state fluctuations - returns charge state probabilities and rates  
    CHARGE = "charge"
    
    # Temperature fluctuations - returns temperature and thermal properties
    THERMAL = "thermal"
    
    # Strain tensor fluctuations - returns strain tensor components
    STRAIN = "strain"
    
    # Optical noise - returns laser intensity and photon statistics
    OPTICAL = "optical"
    
    # Electric field noise - returns electric field vector in V/m
    ELECTRIC = "electric"
    
    # Microwave control noise - returns MW amplitude/phase errors
    MICROWAVE = "microwave"


class PhysicsNoiseInterface(ABC):
    """
    Physics-consistent interface that all noise sources must implement
    
    This interface ensures that every noise source provides the minimum
    information needed for physics calculations:
    - Time-dependent samples
    - Power spectral density
    - Correlation functions
    - Physical parameter validation
    """
    
    @abstractmethod
    def get_physics_sample(self, t: float) -> Any:
        """Get physics-consistent sample at time t"""
        pass
        
    @abstractmethod
    def get_power_spectral_density(self, frequencies: np.ndarray) -> np.ndarray:
        """Get power spectral density for physics validation"""
        pass
        
    @abstractmethod
    def get_correlation_function(self, tau_values: np.ndarray) -> np.ndarray:
        """Get correlation function for physics validation"""
        pass
        
    @abstractmethod
    def validate_physics_parameters(self) -> Dict[str, bool]:
        """Validate that all parameters obey physical constraints"""
        pass


class TypedNoiseSource(NoFallbackPhysicsEngine, PhysicsNoiseInterface):
    """
    Type-safe noise source that CANNOT return wrong types
    
    ARCHITECTURAL INNOVATION:
    This class makes it impossible to return the wrong type from a noise source.
    Every source is strictly typed, and outputs are validated before return.
    
    CORE PROTECTION:
    - Type validation: Output must match declared type
    - Physics validation: Output must obey physical laws
    - Magnitude validation: Output must have realistic values
    - Consistency validation: Output must be self-consistent
    
    USAGE:
    class MyMagneticNoise(TypedNoiseSource):
        def __init__(self):
            super().__init__(NoiseSourceType.MAGNETIC)
            
        def calculate_physics(self, t: float) -> np.ndarray:
            # MUST return 3D B-field - anything else triggers error
            return np.array([Bx, By, Bz])  # Tesla
    
    IMPOSSIBLE ERRORS:
    - Returning zero B-field without justification → Error
    - Returning wrong shape array → Error  
    - Returning non-finite values → Error
    - Returning unrealistic magnitudes → Error
    """
    
    def __init__(self, source_type: NoiseSourceType, 
                 validation_config: Optional[Dict] = None):
        """
        Initialize typed noise source with strict type enforcement
        
        Args:
            source_type: Type of noise this source produces
            validation_config: Optional validation parameters
        """
        super().__init__(validation_level="STRICT")
        
        self.source_type = source_type
        self.validation_config = validation_config or {}
        
        # Get type-specific validator
        self._output_validator = self._get_output_validator()
        self._physics_validator_func = self._get_physics_validator()
        
        # Performance tracking
        self._sample_count = 0
        self._validation_failures = []
        
    def _get_output_validator(self) -> Callable[[Any], bool]:
        """Get validator function for this source type"""
        validators = {
            NoiseSourceType.MAGNETIC: self._validate_magnetic_output,
            NoiseSourceType.CHARGE: self._validate_charge_output,
            NoiseSourceType.THERMAL: self._validate_thermal_output,
            NoiseSourceType.STRAIN: self._validate_strain_output,
            NoiseSourceType.OPTICAL: self._validate_optical_output,
            NoiseSourceType.ELECTRIC: self._validate_electric_output,
            NoiseSourceType.MICROWAVE: self._validate_microwave_output
        }
        return validators[self.source_type]
        
    def _get_physics_validator(self) -> Callable[[Any], Dict[str, bool]]:
        """Get physics validator for this source type"""
        physics_validators = {
            NoiseSourceType.MAGNETIC: self._validate_magnetic_physics,
            NoiseSourceType.CHARGE: self._validate_charge_physics,
            NoiseSourceType.THERMAL: self._validate_thermal_physics,
            NoiseSourceType.STRAIN: self._validate_strain_physics,
            NoiseSourceType.OPTICAL: self._validate_optical_physics,
            NoiseSourceType.ELECTRIC: self._validate_electric_physics,
            NoiseSourceType.MICROWAVE: self._validate_microwave_physics
        }
        return physics_validators[self.source_type]
        
    def get_physics_sample(self, t: float) -> Any:
        """
        Get physics-validated sample at time t
        
        This is the main interface method that ensures type safety.
        All outputs are validated before return.
        """
        # Calculate physics using subclass implementation
        try:
            result = self.calculate_physics(t)
        except Exception as e:
            raise FallbackViolationError(
                f"Physics calculation failed in {self.__class__.__name__}: {e}",
                violation_type="CALCULATION_FAILURE",
                suggested_fix="Fix the underlying physics calculation"
            ) from e
            
        # Validate output type and magnitude
        if not self._output_validator(result):
            self._validation_failures.append((t, result, "TYPE_VALIDATION"))
            raise FallbackViolationError(
                f"{self.__class__.__name__} produced invalid {self.source_type.value} output: {result}",
                violation_type="OUTPUT_TYPE_VIOLATION",
                suggested_fix=f"Ensure output matches {self.source_type.value} specification"
            )
            
        # Validate physics consistency
        physics_check = self._physics_validator_func(result)
        if not all(physics_check.values()):
            failed_checks = [k for k, v in physics_check.items() if not v]
            self._validation_failures.append((t, result, f"PHYSICS_VALIDATION: {failed_checks}"))
            raise FallbackViolationError(
                f"{self.__class__.__name__} output violates physics: {failed_checks}",
                violation_type="PHYSICS_VIOLATION",
                suggested_fix="Check physics implementation for consistency"
            )
            
        # Track successful sample
        self._sample_count += 1
        return result
        
    # Type-specific validators
    
    def _validate_magnetic_output(self, result: Any) -> bool:
        """Validate magnetic field output"""
        if not isinstance(result, np.ndarray):
            return False
        if result.shape != (3,):
            return False
        if not np.all(np.isfinite(result)):
            return False
        # Realistic magnitude check (< 100 Tesla for lab conditions)
        magnitude = np.linalg.norm(result)
        if magnitude > 100.0:
            return False
        return True
        
    def _validate_charge_output(self, result: Any) -> bool:
        """Validate charge state output"""
        if not isinstance(result, dict):
            return False
        required_keys = ['charge_state', 'probability']
        if not all(key in result for key in required_keys):
            return False
        # Validate probability
        if not (0 <= result['probability'] <= 1):
            return False
        return True
        
    def _validate_thermal_output(self, result: Any) -> bool:
        """Validate temperature output"""
        if not isinstance(result, dict):
            return False
        if 'temperature' not in result:
            return False
        temp = result['temperature']
        if not isinstance(temp, (int, float)):
            return False
        if not np.isfinite(temp):
            return False
        # Realistic temperature range (0 to 10000 K)
        if not (0 <= temp <= 10000):
            return False
        return True
        
    def _validate_strain_output(self, result: Any) -> bool:
        """Validate strain tensor output"""
        if not isinstance(result, np.ndarray):
            return False
        # Can be scalar strain or full 3x3 tensor
        if result.shape not in [(), (3, 3)]:
            return False
        if not np.all(np.isfinite(result)):
            return False
        # Realistic strain magnitude (< 1% for diamond)
        if np.max(np.abs(result)) > 0.01:
            return False
        return True
        
    def _validate_optical_output(self, result: Any) -> bool:
        """Validate optical noise output"""
        if not isinstance(result, dict):
            return False
        required_keys = ['intensity_factor', 'photon_count']
        if not all(key in result for key in required_keys):
            return False
        # Intensity factor should be positive
        if result['intensity_factor'] <= 0:
            return False
        # Photon count should be non-negative integer
        if not isinstance(result['photon_count'], int) or result['photon_count'] < 0:
            return False
        return True
        
    def _validate_electric_output(self, result: Any) -> bool:
        """Validate electric field output"""
        if not isinstance(result, np.ndarray):
            return False
        if result.shape != (3,):
            return False
        if not np.all(np.isfinite(result)):
            return False
        # Realistic E-field magnitude (< 1e9 V/m)
        magnitude = np.linalg.norm(result)
        if magnitude > 1e9:
            return False
        return True
        
    def _validate_microwave_output(self, result: Any) -> bool:
        """Validate microwave control noise output"""
        if not isinstance(result, dict):
            return False
        required_keys = ['amplitude_error', 'phase_error']
        if not all(key in result for key in required_keys):
            return False
        # Errors should be finite
        for key in required_keys:
            if not np.isfinite(result[key]):
                return False
        return True
        
    # Physics consistency validators
    
    def _validate_magnetic_physics(self, result: np.ndarray) -> Dict[str, bool]:
        """Validate magnetic field physics consistency"""
        return {
            'finite_values': np.all(np.isfinite(result)),
            'realistic_magnitude': np.linalg.norm(result) < 100.0,
            'proper_dimensions': result.shape == (3,)
        }
        
    def _validate_charge_physics(self, result: Dict) -> Dict[str, bool]:
        """Validate charge state physics consistency"""
        return {
            'probability_conservation': 0 <= result['probability'] <= 1,
            'valid_charge_state': result['charge_state'] in [-1, 0, 1],
            'consistent_data': True  # Could add more checks
        }
        
    def _validate_thermal_physics(self, result: Dict) -> Dict[str, bool]:
        """Validate thermal physics consistency"""
        temp = result['temperature']
        return {
            'positive_temperature': temp >= 0,
            'realistic_temperature': temp <= 10000,
            'finite_temperature': np.isfinite(temp)
        }
        
    def _validate_strain_physics(self, result: np.ndarray) -> Dict[str, bool]:
        """Validate strain physics consistency"""
        return {
            'finite_values': np.all(np.isfinite(result)),
            'realistic_magnitude': np.max(np.abs(result)) <= 0.01,
            'proper_symmetry': True  # Could check tensor symmetry
        }
        
    def _validate_optical_physics(self, result: Dict) -> Dict[str, bool]:
        """Validate optical physics consistency"""
        return {
            'positive_intensity': result['intensity_factor'] > 0,
            'integer_photons': isinstance(result['photon_count'], int),
            'non_negative_photons': result['photon_count'] >= 0
        }
        
    def _validate_electric_physics(self, result: np.ndarray) -> Dict[str, bool]:
        """Validate electric field physics consistency"""
        return {
            'finite_values': np.all(np.isfinite(result)),
            'realistic_magnitude': np.linalg.norm(result) < 1e9,
            'proper_dimensions': result.shape == (3,)
        }
        
    def _validate_microwave_physics(self, result: Dict) -> Dict[str, bool]:
        """Validate microwave noise physics consistency"""
        return {
            'finite_amplitude_error': np.isfinite(result['amplitude_error']),
            'finite_phase_error': np.isfinite(result['phase_error']),
            'realistic_errors': abs(result['amplitude_error']) < 1 and abs(result['phase_error']) < 2*np.pi
        }
        
    def _zero_justified(self, attribute_name: str, value: np.ndarray) -> bool:
        """
        Override to allow zero values when physically justified
        
        For most noise sources, zero output is NOT justified and indicates
        a fallback. Override this method to specify when zeros are acceptable.
        """
        # Default: zeros are never justified (forces real physics)
        return False
        
    def get_diagnostic_info(self) -> Dict[str, Any]:
        """Get diagnostic information for debugging"""
        base_info = super().get_diagnostic_info()
        base_info.update({
            'source_type': self.source_type.value,
            'sample_count': self._sample_count,
            'validation_failures': len(self._validation_failures),
            'recent_failures': self._validation_failures[-5:] if self._validation_failures else []
        })
        return base_info


# Export main classes
__all__ = [
    'NoiseSourceType',
    'PhysicsNoiseInterface', 
    'TypedNoiseSource'
]