#!/usr/bin/env python3
import sys
import os
import numpy as np
from scipy.signal import firwin, lfilter

# Add parent directory to path for module imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from common import apply_antialiasing_filter
from sep_core import bit_mapping_D1, triad_series, rmse, generate_chirp, decimate2
import matplotlib.pyplot as plt
import json
import csv

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)

def align(triads_orig, triads_decimated):
    n = len(triads_orig)
    x_orig = np.linspace(0.0, 1.0, n)
    x_dec = np.linspace(0.0, 1.0, len(triads_decimated))
    xq = np.clip(x_orig / 2.0, 0, 1)
    out = np.zeros_like(triads_orig)
    for i in range(3):
        out[:, i] = np.interp(xq, x_dec, triads_decimated[:, i],
                              left=triads_decimated[0, i], right=triads_decimated[-1, i])
    return out

def save_results(joint_rmse, h_rmse, c_rmse, s_rmse):
    """Save results to JSON and CSV files."""
    # JSON summary
    summary = {
        "test": "T3_convolutional_invariance",
        "hypothesis": "H4 (Convolutional invariance in waves)",
        "joint_rmse": joint_rmse,
        "h_rmse": h_rmse,
        "c_rmse": c_rmse,
        "s_rmse": s_rmse,
        "threshold": 0.05,
        "pass": joint_rmse <= 0.05,
        "mapping": "D1",
        "beta": 0.1
    }
    
    with open("results/T3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # CSV metrics
    with open("results/T3_entropy_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Joint RMSE", joint_rmse])
        writer.writerow(["H RMSE", h_rmse])
        writer.writerow(["C RMSE", c_rmse])
        writer.writerow(["S RMSE", s_rmse])
        writer.writerow(["Threshold", 0.05])
        writer.writerow(["Pass", joint_rmse <= 0.05])

def plot_results(tri0, trid_aligned):
    """Plot the triad series for visualization."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    t_orig = np.linspace(0, 1, len(tri0))
    t_aligned = np.linspace(0, 1, len(trid_aligned))
    
    axes[0].plot(t_orig, tri0[:, 0], label="Original", alpha=0.7)
    axes[0].plot(t_aligned, trid_aligned[:, 0], label="Decimated", alpha=0.7)
    axes[0].set_ylabel("Entropy (H)")
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(t_orig, tri0[:, 1], label="Original", alpha=0.7)
    axes[1].plot(t_aligned, trid_aligned[:, 1], label="Decimated", alpha=0.7)
    axes[1].set_ylabel("Coherence (C)")
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(t_orig, tri0[:, 2], label="Original", alpha=0.7)
    axes[2].plot(t_aligned, trid_aligned[:, 2], label="Decimated", alpha=0.7)
    axes[2].set_ylabel("Stability (S)")
    axes[2].set_xlabel("Normalized Time")
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("results/T3_plots.png", dpi=300, bbox_inches="tight")
    plt.close()

def main():
    print("============================================================")
    print("Running Test T3: Convolutional Invariance")
    print("============================================================")
    
    # Generate chirp signal and apply antialiasing filter
    x = generate_chirp(length=200000, f0=10, f1=50, snr_db=20)
    x = apply_antialiasing_filter(x)

    # Map to bits and compute triads for original signal
    bits0 = bit_mapping_D1(x)
    tri0 = triad_series(bits0, beta=0.1)

    # Decimate and compute triads for decimated signal
    xd = decimate2(x)
    bitsd = bit_mapping_D1(xd)
    trid = triad_series(bitsd, beta=0.1)
    
    # Align decimated triads to original time scale
    trid_aligned = align(tri0, trid)
    
    # Compute component RMSEs
    h = rmse(tri0[:,0], trid_aligned[:,0])
    c = rmse(tri0[:,1], trid_aligned[:,1])
    s = rmse(tri0[:,2], trid_aligned[:,2])
    joint = float((h + c + s) / 3.0)
    
    print(f"Joint RMSE: {joint:.4f}  (H={h:.4f}, C={c:.4f}, S={s:.4f})")
    
    # Evaluate hypothesis
    print("\n============================================================")
    print("EVALUATION RESULTS")
    print("============================================================")
    print(f"H4 (Convolutional Invariance): {'PASS' if joint <= 0.05 else 'FAIL'}")
    print(f"  Joint RMSE: {joint:.4f}")
    print(f"  Threshold: 0.05")
    
    # Save results
    save_results(joint, h, c, s)
    plot_results(tri0, trid_aligned)
    
    print(f"\nResults saved to results/T3_summary.json and results/T3_entropy_metrics.csv")
    print(f"Plots saved to results/T3_plots.png")

if __name__ == "__main__":
    main()