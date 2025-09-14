# Convergence Analysis Documentation

## Overview

This document provides comprehensive guidance on understanding and interpreting the convergence analysis tools used in the Solana Quantum Vulnerability Monte Carlo (SOL-QV-MC) simulation. Proper convergence analysis ensures that our simulation results are statistically robust and suitable for executive decision-making.

## Table of Contents

1. [Monte Carlo Convergence Theory](#monte-carlo-convergence-theory)
2. [Mathematical Formulation](#mathematical-formulation)
3. [Convergence Metrics](#convergence-metrics)
4. [Interpretation Guide](#interpretation-guide)
5. [Statistical Significance Criteria](#statistical-significance-criteria)
6. [Standard Error Methodology](#standard-error-methodology)
7. [Confidence Intervals](#confidence-intervals)
8. [Iteration Count Guidelines](#iteration-count-guidelines)
9. [Quality Assessment](#quality-assessment)

## Monte Carlo Convergence Theory

### What is Convergence?

In Monte Carlo simulations, convergence refers to the stabilization of statistical estimates as the number of iterations increases. A simulation has converged when additional iterations do not significantly change the estimated values of key metrics.

### Central Limit Theorem

The theoretical foundation for Monte Carlo convergence is the Central Limit Theorem (CLT), which states that the sample mean of independent random variables approaches a normal distribution as the sample size increases:

$$\bar{X}_n \sim N\left(\mu, \frac{\sigma^2}{n}\right)$$

Where:
- $\bar{X}_n$ is the sample mean after $n$ iterations
- $\mu$ is the true population mean
- $\sigma^2$ is the population variance
- $n$ is the number of iterations

### Law of Large Numbers

The Strong Law of Large Numbers guarantees that:

$$\lim_{n \to \infty} \bar{X}_n = \mu \quad \text{(almost surely)}$$

This ensures that with sufficient iterations, our estimates will converge to the true values.

## Mathematical Formulation

### Running Mean

The running mean after $n$ iterations is calculated as:

$$\bar{X}_n = \frac{1}{n} \sum_{i=1}^{n} X_i$$

### Running Standard Deviation

The unbiased sample standard deviation is:

$$s_n = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (X_i - \bar{X}_n)^2}$$

### Standard Error

The standard error of the mean is:

$$SE(\bar{X}_n) = \frac{s_n}{\sqrt{n}}$$

For autocorrelated data, we adjust using the effective sample size:

$$SE_{adj}(\bar{X}_n) = \frac{s_n}{\sqrt{ESS}}$$

### Effective Sample Size (ESS)

For time series with autocorrelation, the effective sample size is:

$$ESS = \frac{n}{1 + 2\sum_{k=1}^{K} \rho_k}$$

Where $\rho_k$ is the autocorrelation at lag $k$, and $K$ is the first lag where $\rho_k < 0$.

## Convergence Metrics

### 1. Coefficient of Variation (CV)

The CV measures relative variability:

$$CV = \frac{s_n}{|\bar{X}_n|}$$

**Convergence Criterion**: CV < 0.01 (1% relative standard deviation)

### 2. Gelman-Rubin Statistic ($\hat{R}$)

The Gelman-Rubin statistic compares within-chain and between-chain variance:

$$\hat{R} = \sqrt{\frac{\hat{V}}{W}}$$

Where:
- $W$ is the within-chain variance
- $\hat{V}$ is the pooled variance estimate

**Convergence Criterion**: $\hat{R} < 1.1$

### 3. Autocorrelation Function (ACF)

The autocorrelation at lag $k$ is:

$$\rho_k = \frac{\sum_{i=1}^{n-k}(X_i - \bar{X})(X_{i+k} - \bar{X})}{\sum_{i=1}^{n}(X_i - \bar{X})^2}$$

**Healthy Range**: $|\rho_1| < 0.5$ (low autocorrelation at lag 1)

### 4. Heidelberger-Welch Test

Tests for stationarity and calculates the required burn-in period.

**Pass Criteria**: p-value > 0.05 for stationarity test

## Interpretation Guide

### Reading Convergence Plots

1. **Running Mean Plot**
   - Should stabilize to a horizontal line
   - Oscillations should decrease in amplitude
   - Look for systematic trends (indicates non-convergence)

2. **Standard Error Plot**
   - Should decrease as $1/\sqrt{n}$
   - Flattening indicates convergence
   - Sudden jumps suggest outliers or regime changes

3. **Confidence Interval Plot**
   - Bands should narrow over iterations
   - Should contain the running mean
   - Width proportional to standard error

4. **Autocorrelation Plot**
   - Should decay quickly to near zero
   - Persistent autocorrelation indicates poor mixing
   - Negative autocorrelation at early lags is acceptable

### Common Convergence Patterns

| Pattern | Interpretation | Action |
|---------|---------------|--------|
| Steady decrease in variance | Normal convergence | Continue to target iterations |
| Oscillating mean | Multimodal distribution | Increase iterations significantly |
| Persistent high variance | Heavy-tailed distribution | Use robust statistics |
| Sudden jumps | Rare events captured | Verify model logic |
| No convergence after many iterations | Model issues | Review model assumptions |

## Statistical Significance Criteria

### Hypothesis Testing Framework

For comparing scenarios or testing specific thresholds:

$$H_0: \mu = \mu_0$$
$$H_1: \mu \neq \mu_0$$

Test statistic:
$$t = \frac{\bar{X}_n - \mu_0}{SE(\bar{X}_n)}$$

**Significance Level**: $\alpha = 0.05$ (95% confidence)

### Multiple Comparisons Correction

When testing multiple hypotheses, apply Bonferroni correction:

$$\alpha_{adjusted} = \frac{\alpha}{m}$$

Where $m$ is the number of comparisons.

## Standard Error Methodology

### Basic Standard Error

For independent samples:
$$SE = \frac{s}{\sqrt{n}}$$

### Batch Means Method

For correlated samples, divide data into $b$ batches:
$$SE_{batch} = \frac{s_{batch}}{\sqrt{b}}$$

Where $s_{batch}$ is the standard deviation of batch means.

### Bootstrap Standard Error

Using $B$ bootstrap resamples:
$$SE_{boot} = \sqrt{\frac{1}{B-1} \sum_{i=1}^{B} (\theta_i^* - \bar{\theta}^*)^2}$$

## Confidence Intervals

*Note: CI_{0.95} denotes the 95% confidence interval (α = 0.05)*

### Normal Approximation

For large samples (n > 30):
$$CI_{0.95} = \bar{X}_n \pm 1.96 \times SE(\bar{X}_n)$$

### T-Distribution (Small Samples)

For small samples (n ≤ 30):
$$CI_{0.95} = \bar{X}_n \pm t_{0.975,n-1} \times SE(\bar{X}_n)$$

### Bootstrap Percentile Method

Using bootstrap distribution:
$$CI_{0.95} = [P_{2.5}, P_{97.5}]$$

Where $P_{\alpha}$ is the $\alpha$-th percentile of bootstrap distribution.

### Interpretation

- **Narrow CI**: High precision, good convergence
- **Wide CI**: Low precision, need more iterations
- **Asymmetric CI**: Skewed distribution, consider transformations

## Iteration Count Guidelines

### Minimum Requirements

| Analysis Type | Minimum Iterations | Recommended | Publication Quality |
|--------------|-------------------|-------------|-------------------|
| Preliminary | 100 | 1,000 | 10,000 |
| Development | 1,000 | 5,000 | 25,000 |
| Production | 5,000 | 10,000 | 50,000 |
| Research | 10,000 | 50,000 | 100,000+ |

### Adaptive Iteration Strategy

1. **Start Small**: Begin with 100-1,000 iterations for debugging
2. **Check Convergence**: Monitor CV and standard error
3. **Scale Up**: Increase by factor of 5-10 if not converged
4. **Validate**: Run multiple seeds to verify stability

### Resource Considerations

Iteration time complexity:
- **Memory**: O(n) for storing results
- **CPU Time**: O(n × m) where m is model complexity
- **Disk Space**: O(n × k) where k is output variables

Rule of thumb:
```
Required Iterations ≈ (Current SE / Target SE)² × Current Iterations × (n / ESS)
```

## Quality Assessment

### Simulation Quality Grades

| Grade | Criteria | Suitable For |
|-------|----------|--------------|
| **A** | All variables converged, CV < 0.01, $\hat{R}$ < 1.05, ESS/n > 0.5 | Publication, executive presentation |
| **B** | Most variables converged, CV < 0.02, $\hat{R}$ < 1.1, ESS/n > 0.3 | Internal reports, decision support |
| **C** | Key variables converged, CV < 0.05, $\hat{R}$ < 1.2, ESS/n > 0.1 | Development, preliminary analysis |
| **D** | Some convergence, CV < 0.1, $\hat{R}$ < 1.5 | Testing only |
| **F** | No convergence, high variance | Not usable |

### Convergence Checklist

- [ ] **Coefficient of Variation**: CV < 0.01 for all key metrics
- [ ] **Gelman-Rubin Statistic**: $\hat{R}$ < 1.1 for all variables
- [ ] **Effective Sample Size**: ESS > 0.1 × n
- [ ] **Visual Inspection**: Stable running mean plots
- [ ] **Autocorrelation**: ACF(1) < 0.5
- [ ] **Standard Error**: Decreasing as expected
- [ ] **Multiple Seeds**: Consistent results across different random seeds
- [ ] **Outlier Analysis**: No single iteration dominating results
- [ ] **Time Stability**: Results stable over simulation time horizon

## Best Practices

### 1. Progressive Refinement
```python
iterations = [100, 1000, 5000, 10000, 50000]
for n in iterations:
    run_simulation(n)
    if check_convergence():
        break
```

### 2. Parallel Chain Validation
Run multiple chains with different seeds:
```python
chains = [run_simulation(seed=i) for i in range(4)]
check_gelman_rubin(chains)
```

### 3. Variance Reduction Techniques
- **Antithetic Variables**: Use negatively correlated samples
- **Control Variates**: Reduce variance using known quantities
- **Importance Sampling**: Focus on critical regions
- **Stratified Sampling**: Ensure coverage of parameter space

### 4. Convergence Monitoring
```python
analyzer = ConvergenceAnalyzer(
    convergence_threshold=0.01,
    confidence_level=0.95,
    check_interval=100
)
```

## Troubleshooting

### Problem: Slow Convergence

**Symptoms**: High variance after many iterations

**Solutions**:
1. Increase iterations by factor of 10
2. Check for rare events dominating results
3. Consider variance reduction techniques
4. Review model for numerical instabilities

### Problem: Oscillating Estimates

**Symptoms**: Mean alternates between values

**Solutions**:
1. Check for multimodal distributions
2. Increase burn-in period
3. Use longer runs to capture all modes
4. Consider mixture model approach

### Problem: High Autocorrelation

**Symptoms**: ESS << n, slow mixing

**Solutions**:
1. Thin the samples (use every k-th sample)
2. Reparameterize the model
3. Use different sampling algorithm
4. Increase time between samples

### Problem: Failed Gelman-Rubin Test

**Symptoms**: $\hat{R}$ > 1.1 despite many iterations

**Solutions**:
1. Run longer chains
2. Use different starting values
3. Check for label switching
4. Verify model identifiability

## References

1. Gelman, A., & Rubin, D. B. (1992). "Inference from iterative simulation using multiple sequences." Statistical Science, 7(4), 457-472.

2. Geyer, C. J. (1992). "Practical Markov chain Monte Carlo." Statistical Science, 7(4), 473-483.

3. Robert, C., & Casella, G. (2004). "Monte Carlo Statistical Methods." Springer.

4. Brooks, S., Gelman, A., Jones, G., & Meng, X. L. (Eds.). (2011). "Handbook of Markov chain Monte Carlo." CRC Press.

5. Flegal, J. M., Haran, M., & Jones, G. L. (2008). "Markov chain Monte Carlo: Can we trust the third significant figure?" Statistical Science, 23(2), 250-260.

## Appendix: Implementation Details

### Using the Convergence Analyzer

```python
from src.analysis.convergence_analyzer import ConvergenceAnalyzer

# Initialize analyzer
analyzer = ConvergenceAnalyzer(
    convergence_threshold=0.01,  # 1% CV threshold
    confidence_level=0.95,        # 95% confidence intervals
    min_iterations=100,           # Minimum before checking
    check_interval=100            # Check every 100 iterations
)

# Track metrics during simulation
for i in range(n_iterations):
    results = run_iteration()
    analyzer.track(i, {
        'crqc_year': results.crqc_year,
        'economic_loss': results.total_loss,
        'attack_probability': results.attack_prob
    })
    
    # Check convergence periodically
    if i % 1000 == 0:
        print(analyzer.get_convergence_summary())

# Generate final report
report = analyzer.generate_report(Path("convergence_report.json"))
print(f"Quality Score: {report.quality_score}")
print(f"Recommended Iterations: {report.recommended_iterations:,}")
```

### Interpreting Results

The convergence analyzer provides multiple views:

1. **Real-time Monitoring**: Track convergence during simulation
2. **Final Report**: Comprehensive analysis after completion
3. **Quality Score**: Letter grade for publication readiness
4. **Recommendations**: Suggested iteration count for desired precision

### Integration with Simulation Pipeline

The convergence analyzer is integrated into:
- `src/core/simulation.py`: Real-time monitoring
- `visualization/convergence_plots.py`: Visual analysis
- `reports/statistical_validation.md`: Automated reporting

For questions or issues, please refer to the project documentation or contact the development team.
