"""
I/O utilities for SEP validation tests.
Handles saving results to JSON, CSV, and figures.
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(data: Dict[str, Any], filepath: str):
    """
    Save data to JSON file with proper formatting.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    ensure_dir(os.path.dirname(filepath))
    
    # Convert numpy types to native Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        else:
            return obj
    
    clean_data = convert(data)
    
    with open(filepath, 'w') as f:
        json.dump(clean_data, f, indent=2)
    
    print(f"Saved JSON to {filepath}")

def save_csv(data: List[Dict[str, Any]], filepath: str):
    """
    Save tabular data to CSV file.
    
    Args:
        data: List of dictionaries with consistent keys
        filepath: Path to save file
    """
    if not data:
        return
    
    ensure_dir(os.path.dirname(filepath))
    
    keys = data[0].keys()
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Saved CSV to {filepath}")

def save_figure(fig: plt.Figure, filepath: str, dpi: int = 150):
    """
    Save matplotlib figure to file.
    
    Args:
        fig: Matplotlib figure object
        filepath: Path to save file
        dpi: Resolution for raster formats
    """
    ensure_dir(os.path.dirname(filepath))
    
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    print(f"Saved figure to {filepath}")

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_csv(filepath: str) -> List[Dict[str, Any]]:
    """Load CSV file as list of dictionaries."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def get_results_dir(test_name: str) -> str:
    """Get results directory for a test."""
    return f"whitepaper/validation/results/{test_name}"

def get_timestamp() -> str:
    """Get timestamp string for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_test_results(test_name: str, summary: Dict[str, Any], 
                     metrics: List[Dict[str, Any]], 
                     fig: Optional[plt.Figure] = None,
                     use_timestamp: bool = False):
    """
    Save all test results with consistent naming.
    
    Args:
        test_name: Name of the test (e.g., 'T1', 'T2')
        summary: Summary dictionary
        metrics: List of per-seed metrics
        fig: Optional matplotlib figure
        use_timestamp: Whether to include timestamp in filename
    """
    results_dir = get_results_dir(test_name)
    
    if use_timestamp:
        timestamp = get_timestamp()
        base_name = f"{test_name}_{timestamp}"
    else:
        base_name = test_name
    
    # Save summary JSON
    summary_path = os.path.join(results_dir, f"{base_name}_summary.json")
    save_json(summary, summary_path)
    
    # Save metrics CSV
    if metrics:
        metrics_path = os.path.join(results_dir, f"{base_name}_metrics.csv")
        save_csv(metrics, metrics_path)
    
    # Save figure if provided
    if fig is not None:
        fig_path = os.path.join(results_dir, f"{base_name}_plots.png")
        save_figure(fig, fig_path)

def log_test_header(test_name: str, description: str):
    """Print formatted test header."""
    print("\n" + "="*60)
    print(f"{test_name}: {description}")
    print("="*60)

def log_hypothesis(h_name: str, description: str, threshold: float, 
                  metric: float, passed: bool):
    """Print formatted hypothesis result."""
    status = "PASS" if passed else "FAIL"
    symbol = "✓" if passed else "✗"
    
    print(f"\n{h_name}: {description}")
    print(f"  Threshold: {threshold:.4f}")
    print(f"  Metric:    {metric:.4f}")
    print(f"  Status:    {status} {symbol}")

def log_test_summary(test_name: str, hypotheses: Dict[str, bool]):
    """Print test summary."""
    all_passed = all(hypotheses.values())
    
    print("\n" + "-"*40)
    print(f"{test_name} Summary:")
    
    for h_name, passed in hypotheses.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {h_name}: {status}")
    
    overall = "PASS" if all_passed else "FAIL"
    print(f"\nOverall: {overall}")
    print("-"*40)

def create_results_structure():
    """Create the results directory structure."""
    tests = ['T1', 'T2', 'T3', 'T4', 'T5']
    
    for test in tests:
        ensure_dir(get_results_dir(test))
    
    # Create docs directories
    ensure_dir('docs/figures')
    ensure_dir('docs/summaries')

class TestLogger:
    """Context manager for test logging."""
    
    def __init__(self, test_name: str, description: str):
        self.test_name = test_name
        self.description = description
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        log_test_header(self.test_name, self.description)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            print(f"\n{self.test_name} completed in {elapsed:.2f} seconds")
        else:
            print(f"\n{self.test_name} failed with error: {exc_val}")
        return False

def format_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-safe types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: format_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [format_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(format_json_safe(v) for v in obj)
    else:
        return obj