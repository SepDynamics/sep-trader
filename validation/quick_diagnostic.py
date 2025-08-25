#!/usr/bin/env python3
"""Quick diagnostic test to verify SEP core functionality"""

import numpy as np
from sep_core import *

def test_basic_functionality():
    """Test basic SEP core functions with small datasets"""
    print("Testing SEP Core Functionality")
    print("=" * 40)
    
    # Test 1: Basic signal generation
    print("\n1. Testing signal generation...")
    try:
        signal = generate_van_der_pol(length=1000, mu=1.0, seed=42)
        print(f"✓ Van der Pol signal generated: length={len(signal)}, range=[{signal.min():.3f}, {signal.max():.3f}]")
    except Exception as e:
        print(f"✗ Van der Pol generation failed: {e}")
        return False
    
    try:
        poisson = generate_poisson_process(length=1000, rate=1.0, seed=42)
        print(f"✓ Poisson process generated: length={len(poisson)}, range=[{poisson.min():.3f}, {poisson.max():.3f}]")
    except Exception as e:
        print(f"✗ Poisson process generation failed: {e}")
        return False
    
    # Test 2: Bit mapping
    print("\n2. Testing bit mapping...")
    try:
        bits_d1 = bit_mapping_D1(signal[:500])  # Smaller size for speed
        print(f"✓ D1 bit mapping: shape={bits_d1.shape}, unique_values={np.unique(bits_d1)}")
        
        if len(bits_d1) == 0:
            print("✗ D1 mapping produced empty result")
            return False
            
    except Exception as e:
        print(f"✗ D1 bit mapping failed: {e}")
        return False
    
    try:
        bits_d2 = bit_mapping_D2(signal)
        print(f"✓ D2 bit mapping: shape={bits_d2.shape}, unique_values={np.unique(bits_d2)}")
    except Exception as e:
        print(f"✗ D2 bit mapping failed: {e}")
        return False
    
    # Test 3: Triad computation
    print("\n3. Testing triad computation...")
    try:
        if len(bits_d1) > 1:
            triads = triad_series(bits_d1, beta=0.1)
            print(f"✓ Triad series computed: shape={triads.shape}")
            
            if triads.shape[1] != 3:
                print(f"✗ Expected 3 triad components (H,C,S), got {triads.shape[1]}")
                return False
            
            h_mean = np.mean(triads[:, 0])
            c_mean = np.mean(triads[:, 1])  
            s_mean = np.mean(triads[:, 2])
            
            print(f"  Mean H (entropy): {h_mean:.4f}")
            print(f"  Mean C (coherence): {c_mean:.4f}")
            print(f"  Mean S (stability): {s_mean:.4f}")
            
            # Check for reasonable ranges
            if not (0 <= h_mean <= 1):
                print(f"✗ H entropy out of expected range [0,1]: {h_mean}")
                return False
            if not (0 <= c_mean <= 1):
                print(f"✗ C coherence out of expected range [0,1]: {c_mean}")
                return False
            if not (0 <= s_mean <= 1):
                print(f"✗ S stability out of expected range [0,1]: {s_mean}")
                return False
                
        else:
            print("✗ Not enough bit data to compute triads")
            return False
            
    except Exception as e:
        print(f"✗ Triad computation failed: {e}")
        return False
    
    # Test 4: RMSE function
    print("\n4. Testing RMSE calculation...")
    try:
        a = np.array([1, 2, 3, 4, 5])
        b = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        rmse_val = rmse(a, b)
        expected_rmse = np.sqrt(np.mean((a - b)**2))
        
        print(f"✓ RMSE calculation: {rmse_val:.4f} (expected: {expected_rmse:.4f})")
        
        if abs(rmse_val - expected_rmse) > 1e-10:
            print(f"✗ RMSE mismatch")
            return False
            
    except Exception as e:
        print(f"✗ RMSE calculation failed: {e}")
        return False
    
    # Test 5: Time scaling (simplified)
    print("\n5. Testing time scaling...")
    try:
        original = np.sin(np.linspace(0, 4*np.pi, 100))
        
        # Manual time scaling
        gamma = 1.5
        scaled_indices = np.arange(len(original)) / gamma
        scaled = np.interp(np.arange(len(original)), scaled_indices, original)
        
        print(f"✓ Time scaling: original_range=[{original.min():.3f}, {original.max():.3f}], scaled_range=[{scaled.min():.3f}, {scaled.max():.3f}]")
        
    except Exception as e:
        print(f"✗ Time scaling failed: {e}")
        return False
    
    print("\n" + "=" * 40)
    print("✓ ALL BASIC TESTS PASSED")
    return True

def test_potential_issues():
    """Test for potential issues causing 0.000000 RMSE"""
    print("\nTesting for potential issues...")
    
    # Test identical signals
    signal = generate_van_der_pol(length=500, seed=42)
    bits1 = bit_mapping_D1(signal)
    bits2 = bit_mapping_D1(signal)  # Same signal, should be identical
    
    if len(bits1) > 0 and len(bits2) > 0:
        triads1 = triad_series(bits1, beta=0.1)
        triads2 = triad_series(bits2, beta=0.1)
        
        rmse_val = rmse(triads1.flatten(), triads2.flatten())
        print(f"RMSE between identical signals: {rmse_val:.10f}")
        
        if rmse_val == 0.0:
            print("This explains the 0.000000 RMSE - signals may be identical after processing")
        
    # Test different seeds
    signal1 = generate_van_der_pol(length=500, seed=42)
    signal2 = generate_van_der_pol(length=500, seed=123)
    
    bits1 = bit_mapping_D1(signal1)
    bits2 = bit_mapping_D1(signal2)
    
    if len(bits1) > 0 and len(bits2) > 0:
        triads1 = triad_series(bits1, beta=0.1)
        triads2 = triad_series(bits2, beta=0.1)
        
        rmse_val = rmse(triads1.flatten(), triads2.flatten())
        print(f"RMSE between different seeds: {rmse_val:.6f}")

if __name__ == "__main__":
    success = test_basic_functionality()
    if success:
        test_potential_issues()
    else:
        print("Basic functionality tests failed!")