#!/usr/bin/env python3
"""
Fast SEP Physics Validation - Reduced dataset sizes for quick testing
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import traceback

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Patch the test parameters to use smaller datasets
def patch_test_parameters():
    """Monkey patch test parameters for faster execution"""
    from test_scripts import T1_time_scaling_test
    from test_scripts import T2_maxent_sufficiency_test
    from test_scripts import T3_convolutional_invariance_test
    from test_scripts import T4_retrodictive_reconstruction_test
    from test_scripts import T5_smoothing_beats_filtering_test
    
    # Reduce dataset sizes for all tests
    T1_time_scaling_test.PROCESS_LENGTH = 5000  # Down from 200000
    T2_maxent_sufficiency_test.PROCESS_LENGTH = 5000  # Down from 100000
    T3_convolutional_invariance_test.PROCESS_LENGTH = 5000  # Down from 50000
    T4_retrodictive_reconstruction_test.PROCESS_LENGTH = 2000  # Down from 20000
    T5_smoothing_beats_filtering_test.PROCESS_LENGTH = 5000  # Down from 100000
    
    print("Fast validation mode: Using reduced dataset sizes")
    print(f"  T1 process length: {T1_time_scaling_test.PROCESS_LENGTH}")
    print(f"  T2 process length: {T2_maxent_sufficiency_test.PROCESS_LENGTH}")
    print(f"  T3 process length: {T3_convolutional_invariance_test.PROCESS_LENGTH}")
    print(f"  T4 process length: {T4_retrodictive_reconstruction_test.PROCESS_LENGTH}")
    print(f"  T5 process length: {T5_smoothing_beats_filtering_test.PROCESS_LENGTH}")

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import numpy
        import scipy
        import matplotlib
        import sklearn
        print(f"✓ NumPy {numpy.__version__}")
        print(f"✓ SciPy {scipy.__version__}")
        print(f"✓ Matplotlib {matplotlib.__version__}")
        print(f"✓ Scikit-learn {sklearn.__version__}")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install requirements with: pip install -r requirements.txt")
        return False

def create_results_directory():
    """Create results directory with timestamp."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_UTC")
    results_dir = Path(f"results/fast_validation_run_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlink to latest
    latest_link = Path("results/latest_fast")
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(results_dir.name)
    
    return results_dir

def run_test(test_module, results_dir):
    """Run a single test and capture results."""
    test_name = test_module.replace('.py', '')
    print(f"\n{'='*60}")
    print(f"RUNNING FAST TEST: {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Import and run the test
        module = __import__(test_module.replace('.py', ''))
        
        # Redirect results to our timestamped directory
        test_results_path = results_dir / test_name
        test_results_path.mkdir(exist_ok=True)
        
        # Change working directory to put results in the right place
        original_cwd = os.getcwd()
        try:
            os.chdir(str(test_results_path))
            
            # Run the test
            if hasattr(module, 'main'):
                success = module.main()
            else:
                print(f"Warning: {test_module} does not have a main() function")
                success = False
                
        finally:
            os.chdir(original_cwd)
        
        end_time = time.time()
        duration = end_time - start_time
        
        result = {
            'test_name': test_name,
            'success': success,
            'duration_seconds': duration,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': None,
            'fast_mode': True
        }
        
        status = "PASS" if success else "FAIL"
        print(f"\n{test_name}: {status} ({duration:.2f}s)")
        
        return result
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        
        result = {
            'test_name': test_name,
            'success': False,
            'duration_seconds': duration,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'error': error_msg,
            'traceback': traceback.format_exc(),
            'fast_mode': True
        }
        
        print(f"\n{test_name}: ERROR ({duration:.2f}s)")
        print(f"Error: {error_msg}")
        
        return result

def print_final_report(test_results):
    """Print final validation report to console."""
    print("\n" + "="*80)
    print("SEP PHYSICS FAST VALIDATION REPORT")
    print("="*80)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r['success'])
    failed_tests = total_tests - passed_tests
    total_duration = sum(r['duration_seconds'] for r in test_results)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    print(f"Duration: {total_duration:.2f} seconds")
    print()
    
    print("TEST RESULTS:")
    print(f"  Total Tests: {total_tests}")
    print(f"  Passed: {passed_tests}")
    print(f"  Failed: {failed_tests}")
    print(f"  Success Rate: {success_rate:.1%}")
    print()
    
    # Individual test status
    print("INDIVIDUAL TEST STATUS:")
    for result in test_results:
        status = "PASS" if result['success'] else "FAIL"
        duration = result['duration_seconds']
        print(f"  {result['test_name']:<40} {status:>6} ({duration:6.2f}s)")
        if result['error']:
            print(f"    Error: {result['error']}")
    
    print("="*80)
    
    overall_success = success_rate >= 0.8
    
    if overall_success:
        print("🎉 FAST VALIDATION SUCCESSFUL: Core testing framework is functional")
    else:
        print("⚠️  FAST VALIDATION INCOMPLETE: Some core tests failed")
    
    print("="*80)
    
    return overall_success

def main():
    """Run fast SEP validation suite."""
    print("SEP Physics Fast Validation Framework")
    print("=" * 50)
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        return False
    
    # Patch test parameters
    print("\nPatching test parameters for fast execution...")
    patch_test_parameters()
    
    # Create results directory
    results_dir = create_results_directory()
    print(f"\nResults will be saved to: {results_dir}")
    
    # List of tests to run
    test_modules = [
        'test_scripts/T1_time_scaling_test.py',
        'test_scripts/T2_maxent_sufficiency_test.py',
        'test_scripts/T3_convolutional_invariance_test.py',
        'test_scripts/T4_retrodictive_reconstruction_test.py',
        'test_scripts/T5_smoothing_beats_filtering_test.py'
    ]
    
    # Run each test
    test_results = []
    
    print(f"\nRunning {len(test_modules)} fast validation tests...")
    
    for test_module in test_modules:
        if Path(test_module).exists():
            result = run_test(test_module, results_dir)
            test_results.append(result)
        else:
            print(f"Warning: Test file {test_module} not found, skipping")
    
    # Save results summary
    summary = {
        'validation_run': {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_tests': len(test_results),
            'passed_tests': sum(1 for r in test_results if r['success']),
            'total_duration_seconds': sum(r['duration_seconds'] for r in test_results),
            'fast_mode': True
        },
        'test_results': test_results
    }
    
    summary_file = results_dir / 'fast_validation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print final report
    overall_success = print_final_report(test_results)
    
    print(f"\nDetailed results saved to: {results_dir}")
    print(f"Summary report: {results_dir}/fast_validation_summary.json")
    
    return overall_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nFast validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nFast validation failed with unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)