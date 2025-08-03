#!/usr/bin/env python3

import unittest
import sys
import os

# Add the package to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'lib'))

import sep_dsl
from sep_dsl import DSLInterpreter, DSLError, DSLRuntimeError, DSLVariableError


class TestDSLInterpreter(unittest.TestCase):
    """Test suite for SEP DSL Python interface."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dsl = DSLInterpreter()
    
    def test_interpreter_creation(self):
        """Test DSL interpreter can be created."""
        self.assertIsInstance(self.dsl, DSLInterpreter)
    
    def test_simple_pattern_execution(self):
        """Test basic pattern execution."""
        script = """
        pattern test_pattern {
            coherence = measure_coherence("test_data")
            entropy = measure_entropy("test_data")
            print("Test pattern executed")
        }
        """
        # Should not raise exception
        self.dsl.execute(script)
    
    def test_variable_access(self):
        """Test variable access from executed patterns."""
        script = """
        pattern var_test {
            coherence = measure_coherence("sensor_data")
            entropy = measure_entropy("sensor_data")
        }
        """
        self.dsl.execute(script)
        
        # These should return string values
        coherence = self.dsl.get_variable("var_test.coherence")
        entropy = self.dsl.get_variable("var_test.entropy")
        
        self.assertIsInstance(coherence, str)
        self.assertIsInstance(entropy, str)
        
        # Values should be numeric strings
        float(coherence)  # Should not raise ValueError
        float(entropy)    # Should not raise ValueError
    
    def test_variable_not_found(self):
        """Test error handling for non-existent variables."""
        with self.assertRaises(DSLVariableError):
            self.dsl.get_variable("nonexistent.variable")
    
    def test_invalid_script(self):
        """Test error handling for invalid DSL syntax."""
        invalid_script = "this is not valid DSL syntax"
        with self.assertRaises(DSLRuntimeError):
            self.dsl.execute(invalid_script)
    
    def test_pattern_results(self):
        """Test getting all pattern results as dictionary."""
        script = """
        pattern results_test {
            coherence = measure_coherence("data")
            entropy = measure_entropy("data")
        }
        """
        self.dsl.execute(script)
        
        results = self.dsl.get_pattern_results("results_test")
        self.assertIsInstance(results, dict)
        self.assertIn("coherence", results)
        self.assertIn("entropy", results)
    
    def test_quick_analysis_methods(self):
        """Test convenience analysis methods."""
        coherence = self.dsl.analyze_coherence("test_data")
        entropy = self.dsl.analyze_entropy("test_data")
        
        self.assertIsInstance(coherence, float)
        self.assertIsInstance(entropy, float)
        self.assertGreaterEqual(coherence, 0.0)
        self.assertLessEqual(coherence, 1.0)
        self.assertGreaterEqual(entropy, 0.0)
        self.assertLessEqual(entropy, 1.0)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""
    
    def test_quick_analysis(self):
        """Test quick_analysis convenience function."""
        results = sep_dsl.quick_analysis("test_data")
        
        self.assertIsInstance(results, dict)
        self.assertIn("coherence", results)
        self.assertIn("entropy", results)
        self.assertIsInstance(results["coherence"], float)
        self.assertIsInstance(results["entropy"], float)


class TestFileExecution(unittest.TestCase):
    """Test file-based DSL execution."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dsl = DSLInterpreter()
        self.test_file = "test_script.sep"
    
    def tearDown(self):
        """Clean up test files."""
        if os.path.exists(self.test_file):
            os.remove(self.test_file)
    
    def test_execute_file(self):
        """Test executing DSL script from file."""
        script_content = """
        pattern file_test {
            coherence = measure_coherence("file_data")
            print("Executed from file")
        }
        """
        
        # Create test file
        with open(self.test_file, 'w') as f:
            f.write(script_content)
        
        # Execute from file
        self.dsl.execute_file(self.test_file)
        
        # Verify execution
        coherence = self.dsl.get_variable("file_test.coherence")
        self.assertIsInstance(coherence, str)
    
    def test_file_not_found(self):
        """Test error handling for non-existent files."""
        with self.assertRaises(DSLRuntimeError):
            self.dsl.execute_file("nonexistent.sep")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dsl = DSLInterpreter()
    
    def test_invalid_variable_type(self):
        """Test error handling for invalid variable name types."""
        with self.assertRaises(TypeError):
            self.dsl.get_variable(123)  # Should be string
    
    def test_invalid_script_type(self):
        """Test error handling for invalid script types."""
        with self.assertRaises(TypeError):
            self.dsl.execute(123)  # Should be string
    
    def test_empty_script(self):
        """Test handling of empty scripts."""
        # Empty script should execute without error
        self.dsl.execute("")
    
    def test_pattern_not_found(self):
        """Test error handling when pattern has no accessible variables."""
        # Execute a script without creating any patterns
        self.dsl.execute("// Just a comment")
        
        with self.assertRaises(DSLVariableError):
            self.dsl.get_pattern_results("nonexistent_pattern")


if __name__ == '__main__':
    # Check if we can import the C extension
    try:
        from sep_dsl._sep_dsl import DSLInterpreter as _DSLInterpreter
        print("✅ C extension loaded successfully")
    except ImportError as e:
        print(f"❌ Failed to load C extension: {e}")
        print("Make sure to build the extension first:")
        print("  cd /sep/bindings/python")
        print("  python setup.py build_ext --inplace")
        sys.exit(1)
    
    # Run tests
    unittest.main(verbosity=2)
