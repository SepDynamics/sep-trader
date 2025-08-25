#!/usr/bin/env python3
"""
SEP Physics Validation Framework Runner

This script runs the complete SEP physics validation test suite,
generating reproducible artifacts and comprehensive analysis reports.
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
from datetime import datetime, timezone
import traceback

# Add current directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    results_dir = Path(f"results/validation_run_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create symlink to latest
    latest_link = Path("results/latest")
    if latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(results_dir.name)
    
    return results_dir

def run_test(test_module, results_dir):
    """Run a single test and capture results."""
    test_name = test_module.replace('.py', '')
    print(f"\n{'='*60}")
    print(f"RUNNING TEST: {test_name}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Import and run the test
        module = __import__(test_module.replace('.py', ''))
        
        # Redirect results to our timestamped directory
        original_results_path = Path("results")
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
            'error': None
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
            'traceback': traceback.format_exc()
        }
        
        print(f"\n{test_name}: ERROR ({duration:.2f}s)")
        print(f"Error: {error_msg}")
        
        return result

def generate_summary_report(test_results, results_dir):
    """Generate comprehensive summary report."""
    
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results if r['success'])
    failed_tests = total_tests - passed_tests
    total_duration = sum(r['duration_seconds'] for r in test_results)
    
    summary = {
        'validation_run': {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0,
            'total_duration_seconds': total_duration
        },
        'test_results': test_results,
        'hypothesis_summary': {}
    }
    
    # Analyze hypothesis results
    hypothesis_results = {}
    
    for result in test_results:
        if result['success']:
            test_name = result['test_name']
            
            # Try to load detailed results from each test
            test_result_file = results_dir / test_name / f"{test_name}_summary.json"
            if test_result_file.exists():
                try:
                    with open(test_result_file, 'r') as f:
                        test_data = json.load(f)
                    
                    if 'evaluation' in test_data:
                        evaluation = test_data['evaluation']
                        
                        # Extract hypothesis results
                        for key, value in evaluation.items():
                            if isinstance(value, dict) and 'pass' in value:
                                hypothesis_results[f"{test_name}_{key}"] = {
                                    'test': test_name,
                                    'hypothesis': key,
                                    'pass': value['pass'],
                                    'details': value
                                }
                except Exception as e:
                    print(f"Warning: Could not load detailed results for {test_name}: {e}")
    
    summary['hypothesis_summary'] = hypothesis_results
    
    # Calculate overall hypothesis success rate
    total_hypotheses = len(hypothesis_results)
    passed_hypotheses = sum(1 for h in hypothesis_results.values() if h['pass'])
    
    summary['validation_run']['total_hypotheses'] = total_hypotheses
    summary['validation_run']['passed_hypotheses'] = passed_hypotheses
    summary['validation_run']['hypothesis_success_rate'] = passed_hypotheses / total_hypotheses if total_hypotheses > 0 else 0
    
    # Save summary
    summary_file = results_dir / 'validation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary

def print_final_report(summary):
    """Print final validation report to console."""
    print("\n" + "="*80)
    print("SEP PHYSICS VALIDATION FINAL REPORT")
    print("="*80)
    
    run_info = summary['validation_run']
    
    print(f"Timestamp: {run_info['timestamp']}")
    print(f"Duration: {run_info['total_duration_seconds']:.2f} seconds")
    print()
    
    print("TEST RESULTS:")
    print(f"  Total Tests: {run_info['total_tests']}")
    print(f"  Passed: {run_info['passed_tests']}")
    print(f"  Failed: {run_info['failed_tests']}")
    print(f"  Success Rate: {run_info['success_rate']:.1%}")
    print()
    
    print("HYPOTHESIS VALIDATION:")
    print(f"  Total Hypotheses: {run_info['total_hypotheses']}")
    print(f"  Passed: {run_info['passed_hypotheses']}")
    print(f"  Failed: {run_info['total_hypotheses'] - run_info['passed_hypotheses']}")
    print(f"  Success Rate: {run_info['hypothesis_success_rate']:.1%}")
    print()
    
    # Individual test status
    print("INDIVIDUAL TEST STATUS:")
    for result in summary['test_results']:
        status = "PASS" if result['success'] else "FAIL"
        duration = result['duration_seconds']
        print(f"  {result['test_name']:<40} {status:>6} ({duration:6.2f}s)")
        if result['error']:
            print(f"    Error: {result['error']}")
    print()
    
    # Hypothesis details
    if summary['hypothesis_summary']:
        print("HYPOTHESIS DETAILS:")
        for hyp_key, hyp_data in summary['hypothesis_summary'].items():
            status = "PASS" if hyp_data['pass'] else "FAIL"
            print(f"  {hyp_data['test']:<20} {hyp_data['hypothesis']:<25} {status:>6}")
    
    print("="*80)
    
    overall_success = (run_info['success_rate'] >= 0.8 and 
                      run_info['hypothesis_success_rate'] >= 0.7)
    
    if overall_success:
        print("🎉 VALIDATION SUCCESSFUL: SEP physics claims supported by evidence")
    else:
        print("⚠️  VALIDATION INCOMPLETE: Some tests failed or hypotheses not supported")
    
    print("="*80)
    
    return overall_success

def main():
    """Run complete SEP validation suite."""
    print("SEP Physics Validation Framework")
    print("=" * 40)
    
    # Check dependencies
    print("\nChecking dependencies...")
    if not check_dependencies():
        return False
    
    # Create results directory
    results_dir = create_results_directory()
    print(f"\nResults will be saved to: {results_dir}")
    
    # List of tests to run
    test_modules = [
        'test_T1_time_scaling.py',
        'test_T2_maxent_sufficiency.py',
        'test_T3_convolutional_invariance.py',
        'test_T4_retrodictive_reconstruction.py',
        'test_T5_smoothing_beats_filtering.py'
    ]
    
    # Run each test
    test_results = []
    
    print(f"\nRunning {len(test_modules)} validation tests...")
    
    for test_module in test_modules:
        if Path(test_module).exists():
            result = run_test(test_module, results_dir)
            test_results.append(result)
        else:
            print(f"Warning: Test file {test_module} not found, skipping")
    
    # Generate summary report
    print(f"\nGenerating summary report...")
    summary = generate_summary_report(test_results, results_dir)
    
    # Print final report
    overall_success = print_final_report(summary)
    
    print(f"\nDetailed results saved to: {results_dir}")
    print(f"Summary report: {results_dir}/validation_summary.json")
    
    return overall_success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nValidation failed with unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)