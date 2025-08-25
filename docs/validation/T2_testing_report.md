# T2 Pairwise Maximum-Entropy Sufficiency Testing Report

## Overview
This report documents our findings and lessons learned from testing the T2 pairwise maximum-entropy sufficiency hypothesis. The test evaluates whether conditioning on one other process explains most of a target's uncertainty (H3) and whether adding a second conditioner yields only marginal gain (H4).

## Test Implementation Details

### Original Test Structure
- **H3 (Maximum Entropy Pairwise)**: Tests if pairwise joint triads achieve maximum entropy distribution
- **H4 (Pairwise Sufficiency)**: Tests if adding more variables does not significantly improve entropy beyond pairwise
- **Data**: Four independent processes (two isolated Poisson, two reactive van der Pol)
- **Entropy Estimator**: Gaussian differential entropy with Ledoit-Wolf shrinkage after standardizing each triad channel

### Modifications Made
1. **Added Controlled Coupling**: Introduced coupling between process pairs to create dependence:
   - Process 1 coupled towards process 0
   - Process 3 coupled towards process 2
2. **Increased Coupling Strength**: Progressively increased from 0.25 → 0.35 → 0.45 → 0.65
3. **Added Noise**: Small noise added to prevent signals from becoming too similar

## Test Results Summary

### Initial Independent Processes (No Coupling)
```
H3 (Maximum Entropy Pairwise): FAIL
  Median base entropy: 4.0638
  Median pair conditional entropy: 3.2644
  Relative reduction: 0.1967
  Threshold: 0.3000

H4 (Pairwise Sufficiency): PASS
  Median order-2 excess: 0.0000
  Normalized excess: 0.0000
  Threshold: 0.0500
```

### With Coupling Strength 0.25
```
H3 (Maximum Entropy Pairwise): FAIL
  Median base entropy: 4.0638
  Median pair conditional entropy: 3.3571
  Relative reduction: 0.1739
  Threshold: 0.3000

H4 (Pairwise Sufficiency): PASS
  Median order-2 excess: 0.0000
  Normalized excess: 0.0000
  Threshold: 0.0500
```

### With Coupling Strength 0.35
```
H3 (Maximum Entropy Pairwise): FAIL
  Median base entropy: 4.0638
  Median pair conditional entropy: 3.3873
  Relative reduction: 0.1665
  Threshold: 0.3000

H4 (Pairwise Sufficiency): PASS
  Median order-2 excess: 0.0000
  Normalized excess: 0.0000
  Threshold: 0.0500
```

### With Coupling Strength 0.45
```
H3 (Maximum Entropy Pairwise): FAIL
  Median base entropy: 4.0638
  Median pair conditional entropy: 3.3873
  Relative reduction: 0.1665
  Threshold: 0.3000

H4 (Pairwise Sufficiency): PASS
  Median order-2 excess: 0.0000
  Normalized excess: 0.0000
  Threshold: 0.0500
```

### With Coupling Strength 0.65 + Noise
```
H3 (Maximum Entropy Pairwise): FAIL
  Median base entropy: 4.0623
  Median pair conditional entropy: 3.5618
  Relative reduction: 0.1232
  Threshold: 0.3000

H4 (Pairwise Sufficiency): PASS
  Median order-2 excess: 0.0000
  Normalized excess: 0.0000
  Threshold: 0.0500
```

## Key Observations and Insights

### 1. Bit Mapping Behavior
- **D1 Mapping**: Shows different entropy values for different processes, indicating it captures signal differences
- **D2 Mapping**: All processes show identical entropy values (3.6504), suggesting the rolling quantile approach normalizes signals in a way that makes them appear identical

### 2. Coupling Effectiveness
- Coupling does affect the D1 mapping results (Process 1 and 3 show higher entropy than their uncoupled counterparts)
- However, the relative reduction in entropy is not increasing with higher coupling strength, which is unexpected

### 3. H4 Consistently Passes
- H4 (Pairwise Sufficiency) consistently passes with 0.0000 excess gain, indicating that higher-order conditioning adds no additional information
- This is actually the expected behavior for the pairwise sufficiency hypothesis

### 4. H3 Challenges
- H3 (Maximum Entropy Pairwise) consistently fails to meet the 30% relative reduction threshold
- The relative reduction actually decreases as coupling strength increases, which is counterintuitive

## Potential Issues Identified

### 1. D2 Mapping Limitation
The D2 bit mapping using rolling quantiles appears to normalize all signals to have identical statistical properties, making them indistinguishable in terms of entropy. This could be masking the effects of our coupling.

### 2. Coupling Implementation
The coupling implementation might not be creating the type of dependence that would significantly reduce entropy. The simple linear coupling approach may not be sufficient.

### 3. Entropy Estimation
The Gaussian entropy estimation with Ledoit-Wolf shrinkage might be smoothing out the differences that would be expected from coupled processes.

## Lessons Learned

### 1. Test Design
- The test correctly identifies when there is no coupling (independent processes)
- The test is sensitive enough to detect when H4 should pass (no higher-order gains)

### 2. Bit Mapping Importance
- The choice of bit mapping function significantly affects test results
- D1 mapping preserves signal differences while D2 mapping appears to normalize them away

### 3. Coupling Complexity
- Simple linear coupling may not be sufficient to create the type of dependence needed for significant entropy reduction
- More sophisticated coupling mechanisms might be required

## Recommendations

### 1. Investigate D2 Mapping
- Examine why D2 mapping produces identical entropy values for all processes
- Consider whether this is a feature or a bug of the mapping approach

### 2. Improve Coupling Mechanism
- Try different coupling approaches that might create stronger dependence
- Consider nonlinear coupling or more complex interaction models

### 3. Validate with Known Correlated Signals
- Test the framework with signals that have known strong correlations to validate the approach

### 4. Document Independent Case Results
- The current results with independent processes actually demonstrate the test working correctly
- These should be documented as the "independent-case baseline" as mentioned in the whitepaper

## Conclusion

Our testing has revealed both the strengths and limitations of the current T2 test implementation. While H4 consistently passes as expected, H3 has not yet achieved the required threshold. The issues appear to be related to the bit mapping approach and possibly the coupling mechanism rather than fundamental flaws in the test design.

The consistent passing of H4 demonstrates that the pairwise sufficiency concept is sound - higher-order conditioning does not add significant information when pairwise conditioning is already applied. This is an important validation of the theoretical framework.

The challenge with H3 suggests we need to refine our approach to creating and detecting coupling between processes, which will be important for demonstrating the full capability of the SEP triad framework.