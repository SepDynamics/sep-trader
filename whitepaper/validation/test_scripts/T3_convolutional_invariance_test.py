#!/usr/bin/env python3

import argparse
import json
import csv
import os

import numpy as np
from scipy.signal import firwin, lfilter
from sep_core import (
    bit_mapping_D1,
    bit_mapping_D2,
    triad_series,
    rmse,
    generate_chirp,
    decimate2,
)
import matplotlib.pyplot as plt

# Create results directory if it doesn't exist
os.makedirs("results", exist_ok=True)


def antialiased_decimate(signal: np.ndarray, taps: int = 32) -> np.ndarray:
    """Decimate by factor of 2 with an anti-aliasing FIR filter."""
    fir = firwin(taps, 0.5)
    filtered = lfilter(fir, [1.0], signal)
    return filtered[::2]

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

def save_results(mapping: str, beta: float, with_aa: dict, joint_no_aa: float):
    """Save results to JSON and CSV files."""
    summary = {
        "test": "T3_convolutional_invariance",
        "hypothesis": "H4 (Convolutional invariance in waves)",
        "rmse_with_antialiasing": with_aa["joint"],
        "rmse_without_antialiasing": joint_no_aa,
        "h_rmse": with_aa["h"],
        "c_rmse": with_aa["c"],
        "s_rmse": with_aa["s"],
        "threshold": 0.05,
        "pass": with_aa["joint"] <= 0.05,
        "mapping": mapping,
        "beta": beta,
    }

    with open("results/T3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("results/T3_entropy_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Joint RMSE (AA)", with_aa["joint"]])
        writer.writerow(["Joint RMSE (No AA)", joint_no_aa])
        writer.writerow(["H RMSE", with_aa["h"]])
        writer.writerow(["C RMSE", with_aa["c"]])
        writer.writerow(["S RMSE", with_aa["s"]])
        writer.writerow(["Threshold", 0.05])
        writer.writerow(["Pass", with_aa["joint"] <= 0.05])

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

def main() -> None:
    parser = argparse.ArgumentParser(description="Run T3 convolutional invariance test")
    parser.add_argument("--mapping", choices=["D1", "D2"], default="D2",
                        help="Bit mapping strategy")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Triad EMA beta value")
    parser.add_argument("--window-size", type=int, default=1024,
                        help="Window size for D2 mapping")
    args = parser.parse_args()

    print("============================================================")
    print("Running Test T3: Convolutional Invariance")
    print("============================================================")

    x = generate_chirp(length=200000, f0=10, f1=50, snr_db=20)

    if args.mapping == "D1":
        bits0 = bit_mapping_D1(x)
    else:
        bits0 = bit_mapping_D2(x, window_size=args.window_size)
    tri0 = triad_series(bits0, beta=args.beta)

    xd_no_aa = decimate2(x)
    xd_aa = antialiased_decimate(x)

    if args.mapping == "D1":
        bits_no_aa = bit_mapping_D1(xd_no_aa)
        bits_aa = bit_mapping_D1(xd_aa)
    else:
        bits_no_aa = bit_mapping_D2(xd_no_aa, window_size=args.window_size)
        bits_aa = bit_mapping_D2(xd_aa, window_size=args.window_size)

    trid_no_aa = triad_series(bits_no_aa, beta=args.beta)
    trid_aa = triad_series(bits_aa, beta=args.beta)

    trid_no_aa_aligned = align(tri0, trid_no_aa)
    trid_aa_aligned = align(tri0, trid_aa)

    h = rmse(tri0[:, 0], trid_aa_aligned[:, 0])
    c = rmse(tri0[:, 1], trid_aa_aligned[:, 1])
    s = rmse(tri0[:, 2], trid_aa_aligned[:, 2])
    joint_aa = float((h + c + s) / 3.0)

    joint_no_aa = float(
        (rmse(tri0[:, 0], trid_no_aa_aligned[:, 0]) +
         rmse(tri0[:, 1], trid_no_aa_aligned[:, 1]) +
         rmse(tri0[:, 2], trid_no_aa_aligned[:, 2])) / 3.0
    )

    print(f"Joint RMSE (AA): {joint_aa:.4f}  (H={h:.4f}, C={c:.4f}, S={s:.4f})")
    print(f"Joint RMSE (No AA): {joint_no_aa:.4f}")

    print("\n============================================================")
    print("EVALUATION RESULTS")
    print("============================================================")
    print(f"H4 (Convolutional Invariance): {'PASS' if joint_aa <= 0.05 else 'FAIL'}")
    print(f"  Joint RMSE (AA): {joint_aa:.4f}")
    print(f"  Joint RMSE (No AA): {joint_no_aa:.4f}")
    print(f"  Threshold: 0.05")

    save_results(args.mapping, args.beta,
                 {"joint": joint_aa, "h": h, "c": c, "s": s},
                 joint_no_aa)
    plot_results(tri0, trid_aa_aligned)

    print("\nResults saved to results/T3_summary.json and results/T3_entropy_metrics.csv")
    print("Plots saved to results/T3_plots.png")


if __name__ == "__main__":
    main()
