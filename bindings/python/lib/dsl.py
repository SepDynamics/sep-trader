"""
SEP DSL Python Interface

High-level Python wrapper for the SEP DSL engine providing a clean,
Pythonic interface to the quantum pattern analysis capabilities.
"""

import os
import sys
from typing import Any, Dict, Optional, Union

try:
    from ._sep_dsl import DSLInterpreter as _DSLInterpreter
except ImportError as e:
    raise ImportError(
        "SEP DSL C extension not found. Please ensure the package was "
        "installed correctly and libsep.so is available."
    ) from e


class DSLError(Exception):
    """Base exception for SEP DSL errors."""
    pass


class DSLRuntimeError(DSLError):
    """Runtime error during DSL script execution."""
    pass


class DSLVariableError(DSLError):
    """Error accessing DSL variables."""
    pass


class DSLInterpreter:
    """
    SEP DSL Interpreter
    
    High-level interface to the SEP DSL engine for quantum pattern analysis.
    
    Example:
        >>> import sep_dsl
        >>> dsl = sep_dsl.DSLInterpreter()
        >>> dsl.execute('''
        ...     pattern sensor_analysis {
        ...         coherence = measure_coherence("sensor_data")
        ...         entropy = measure_entropy("sensor_data") 
        ...         print("Coherence:", coherence, "Entropy:", entropy)
        ...     }
        ... ''')
        >>> coherence_value = dsl.get_variable("sensor_analysis.coherence")
        >>> print(f"Analysis complete: {coherence_value}")
    """
    
    def __init__(self):
        """Initialize a new DSL interpreter instance."""
        try:
            self._interpreter = _DSLInterpreter()
        except Exception as e:
            raise DSLRuntimeError(f"Failed to initialize DSL interpreter: {e}")
    
    def execute(self, script: str) -> None:
        """
        Execute DSL script.
        
        Args:
            script: DSL script content to execute
            
        Raises:
            DSLRuntimeError: If script execution fails
        """
        if not isinstance(script, str):
            raise TypeError("Script must be a string")
        
        try:
            self._interpreter.execute(script)
        except Exception as e:
            raise DSLRuntimeError(f"Script execution failed: {e}")
    
    def execute_file(self, filepath: str) -> None:
        """
        Execute DSL script from file.
        
        Args:
            filepath: Path to .sep file to execute
            
        Raises:
            DSLRuntimeError: If file reading or execution fails
        """
        if not os.path.exists(filepath):
            raise DSLRuntimeError(f"File not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                script = f.read()
            self.execute(script)
        except IOError as e:
            raise DSLRuntimeError(f"Failed to read file {filepath}: {e}")
    
    def get_variable(self, name: str) -> str:
        """
        Get variable value from DSL context.
        
        Args:
            name: Variable name (supports dot notation like "pattern.variable")
            
        Returns:
            Variable value as string
            
        Raises:
            DSLVariableError: If variable not found
        """
        if not isinstance(name, str):
            raise TypeError("Variable name must be a string")
        
        try:
            return self._interpreter.get_variable(name)
        except Exception as e:
            raise DSLVariableError(f"Variable '{name}' not found: {e}")
    
    def get_pattern_results(self, pattern_name: str) -> Dict[str, str]:
        """
        Get all variables from a pattern as a dictionary.
        
        Args:
            pattern_name: Name of the pattern
            
        Returns:
            Dictionary of variable names to values
            
        Raises:
            DSLVariableError: If pattern not found or variables inaccessible
        """
        # Common pattern variables to check
        common_vars = ["coherence", "entropy", "stability", "rupture", "signal"]
        results = {}
        
        for var in common_vars:
            full_name = f"{pattern_name}.{var}"
            try:
                value = self.get_variable(full_name)
                results[var] = value
            except DSLVariableError:
                # Variable doesn't exist, skip it
                continue
        
        if not results:
            raise DSLVariableError(f"No variables found in pattern '{pattern_name}'")
        
        return results
    
    def analyze_coherence(self, data_name: str = "sensor_data") -> float:
        """
        Quick coherence analysis helper.
        
        Args:
            data_name: Name of data to analyze
            
        Returns:
            Coherence value (0.0 to 1.0)
        """
        script = f"""
        pattern quick_coherence {{
            coherence = measure_coherence("{data_name}")
        }}
        """
        self.execute(script)
        coherence_str = self.get_variable("quick_coherence.coherence")
        try:
            return float(coherence_str)
        except ValueError:
            raise DSLRuntimeError(f"Invalid coherence value: {coherence_str}")
    
    def analyze_entropy(self, data_name: str = "sensor_data") -> float:
        """
        Quick entropy analysis helper.
        
        Args:
            data_name: Name of data to analyze
            
        Returns:
            Entropy value (0.0 to 1.0)
        """
        script = f"""
        pattern quick_entropy {{
            entropy = measure_entropy("{data_name}")
        }}
        """
        self.execute(script)
        entropy_str = self.get_variable("quick_entropy.entropy")
        try:
            return float(entropy_str)
        except ValueError:
            raise DSLRuntimeError(f"Invalid entropy value: {entropy_str}")


def quick_analysis(data_name: str = "sensor_data") -> Dict[str, float]:
    """
    Convenience function for quick pattern analysis.
    
    Args:
        data_name: Name of data to analyze
        
    Returns:
        Dictionary with 'coherence' and 'entropy' values
    """
    dsl = DSLInterpreter()
    script = f"""
    pattern analysis {{
        coherence = measure_coherence("{data_name}")
        entropy = measure_entropy("{data_name}")
    }}
    """
    dsl.execute(script)
    
    return {
        'coherence': float(dsl.get_variable("analysis.coherence")),
        'entropy': float(dsl.get_variable("analysis.entropy"))
    }
