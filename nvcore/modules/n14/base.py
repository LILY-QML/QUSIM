"""
Base classes for N14 module with ZERO-TOLERANCE fallback detection

These base classes make fallbacks IMPOSSIBLE by detecting and rejecting
any attempt to return fallback values.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Any, Union, Dict, List
import inspect

class FallbackViolationError(Exception):
    """Raised when any fallback pattern is detected"""
    pass

class NoFallbackBase(ABC):
    """
    Base class that makes fallbacks IMPOSSIBLE
    
    Any attempt to return common fallback values will raise FallbackViolationError
    """
    
    def __init__(self):
        self._call_stack = []
        
    def __setattr__(self, name: str, value: Any):
        # Intercept all assignments and check for fallbacks
        if hasattr(self, '_initialized') and self._initialized:
            self._validate_not_fallback(value, f"assignment to {name}")
        super().__setattr__(name, value)
    
    def _validate_not_fallback(self, value: Any, context: str = ""):
        """Check if value is a fallback pattern and raise error if so"""
        
        # Check for zero fallbacks (allow legitimate physical zeros)
        legitimate_zeros = {
            '_eta', 'asymmetry_parameter', 'E', 'strain_parameter', 'phase',
            '_current_time', 'frequency', 'detuning', 'time_start', 'offset'
        }
        is_legitimate_zero = any(param in context for param in legitimate_zeros)
        
        if isinstance(value, (int, float)) and value == 0.0 and not is_legitimate_zero:
            caller = self._get_caller_info()
            raise FallbackViolationError(
                f"ZERO FALLBACK DETECTED in {context}!\n"
                f"Caller: {caller}\n"
                f"Value: {value}\n"
                f"FALLBACKS ARE FORBIDDEN - implement real physics!"
            )
        
        # Check for numpy zero arrays
        if isinstance(value, np.ndarray):
            if np.allclose(value, 0) and value.size > 0:
                raise FallbackViolationError(
                    f"ZERO ARRAY FALLBACK DETECTED in {context}!\n"
                    f"Shape: {value.shape}\n"
                    f"All elements are zero - implement real physics!"
                )
        
        # Check for None fallbacks (allow None for initialization)
        legitimate_nones = {'_current_state', '_evolution_history', 'initial_state', 'nv_state'}
        is_legitimate_none = any(param in context for param in legitimate_nones)
        
        if value is None and not is_legitimate_none:
            raise FallbackViolationError(
                f"NONE FALLBACK DETECTED in {context}!\n"
                f"Returning None is forbidden - provide real values!"
            )
        
        # Check for inf/nan fallbacks
        if isinstance(value, (float, np.floating)):
            if np.isinf(value):
                raise FallbackViolationError(
                    f"INFINITY FALLBACK DETECTED in {context}!\n"
                    f"Value: {value}\n"
                    f"Use finite physical values only!"
                )
            if np.isnan(value):
                raise FallbackViolationError(
                    f"NAN FALLBACK DETECTED in {context}!\n"
                    f"NaN values are forbidden - implement proper error handling!"
                )
        
        # Check for empty containers (allow legitimate empty containers)
        legitimate_empty_containers = {'_evolution_history', 'history', 'results', 'sequence'}
        is_legitimate_empty = any(param in context for param in legitimate_empty_containers)
        
        if isinstance(value, (dict, list, tuple)) and len(value) == 0 and not is_legitimate_empty:
            raise FallbackViolationError(
                f"EMPTY CONTAINER FALLBACK in {context}!\n"
                f"Type: {type(value)}\n"
                f"Empty containers are fallbacks - provide real data!"
            )
    
    def _get_caller_info(self) -> str:
        """Get information about the calling function"""
        frame = inspect.currentframe()
        try:
            # Go up the stack to find the actual caller
            caller_frame = frame.f_back.f_back.f_back
            if caller_frame:
                filename = caller_frame.f_code.co_filename
                lineno = caller_frame.f_lineno
                function = caller_frame.f_code.co_name
                return f"{filename}:{lineno} in {function}()"
            return "unknown caller"
        finally:
            del frame
    
    def validate_output(self, output: Any, method_name: str = "") -> Any:
        """Validate output before returning - called by all public methods"""
        self._validate_not_fallback(output, f"output from {method_name}")
        return output
    
    def __init_subclass__(cls, **kwargs):
        """Automatically wrap all public methods with validation"""
        super().__init_subclass__(**kwargs)
        
        # Methods to exclude from wrapping (to avoid recursion)
        excluded_methods = {'validate_output', 'require_experimental_validation'}
        
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and attr_name not in excluded_methods:
                attr = getattr(cls, attr_name)
                if callable(attr) and not isinstance(attr, type):
                    # Wrap method with output validation
                    setattr(cls, attr_name, cls._wrap_method(attr, attr_name))
    
    @staticmethod
    def _wrap_method(method, method_name):
        """Wrap method to validate output"""
        def wrapped(self, *args, **kwargs):
            result = method(self, *args, **kwargs)
            return self.validate_output(result, method_name)
        return wrapped
    
    def __post_init__(self):
        """Mark as initialized to enable fallback detection"""
        self._initialized = True


class N14PhysicsEngine(NoFallbackBase):
    """
    Base class for all N14 physics engines
    
    Enforces:
    - Real quantum mechanical calculations only
    - No approximations without explicit justification
    - Experimental parameter validation
    - Physical consistency checks
    """
    
    def __init__(self):
        super().__init__()
        self._physics_validator = None
        self.__post_init__()
    
    @abstractmethod
    def calculate_physics(self, *args, **kwargs):
        """All subclasses must implement real physics calculation"""
        pass
    
    def validate_physics_parameters(self, **params) -> Dict[str, bool]:
        """Validate all physics parameters against experimental bounds"""
        validation = {}
        
        for param_name, value in params.items():
            validation[param_name] = self._validate_single_parameter(param_name, value)
        
        return validation
    
    def _validate_single_parameter(self, name: str, value: Any) -> bool:
        """Validate single parameter against physical bounds"""
        
        # N14 hyperfine coupling validation
        if 'hyperfine' in name.lower():
            if not isinstance(value, (int, float, np.number)):
                return False
            # Typical range: 0.1 MHz to 10 MHz
            return 1e5 <= abs(value) <= 1e7
        
        # N14 quadrupole coupling validation  
        if 'quadrupole' in name.lower():
            if not isinstance(value, (int, float, np.number)):
                return False
            # Typical range: 1 MHz to 10 MHz
            return 1e6 <= abs(value) <= 1e7
        
        # Magnetic field validation
        if 'field' in name.lower():
            if isinstance(value, np.ndarray):
                if value.shape != (3,):
                    return False
                # Reasonable field range: 1 μT to 10 T
                magnitude = np.linalg.norm(value)
                return 1e-6 <= magnitude <= 10.0
        
        return True  # Default: assume valid if not specifically checked
    
    def require_experimental_validation(self, calculated_value: float, 
                                      literature_value: float, 
                                      parameter_name: str,
                                      tolerance: float = 0.1) -> bool:
        """Require calculated values match experimental literature"""
        
        relative_error = abs(calculated_value - literature_value) / abs(literature_value)
        
        if relative_error > tolerance:
            raise FallbackViolationError(
                f"EXPERIMENTAL VALIDATION FAILED for {parameter_name}!\n"
                f"Calculated: {calculated_value:.3e}\n"
                f"Literature: {literature_value:.3e}\n"
                f"Relative error: {relative_error:.3%} > {tolerance:.1%}\n"
                f"Implementation must match experimental data!"
            )
        
        return True