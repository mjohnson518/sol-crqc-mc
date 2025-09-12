"""
Statistical analysis tools for Monte Carlo simulation results.

This module provides comprehensive statistical analysis of simulation outputs,
including descriptive statistics, time series analysis, and distribution fitting.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import pandas as pd
from scipy import stats
from scipy.stats import norm, lognorm, gamma, expon
import warnings
from collections import defaultdict


@dataclass
class StatisticalSummary:
    """Container for statistical summary metrics."""
    
    mean: float
    median: float
    std: float
    var: float
    min: float
    max: float
    percentiles: Dict[int, float]
    skewness: float
    kurtosis: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    
    @property
    def coefficient_of_variation(self) -> float:
        """Calculate coefficient of variation."""
        return self.std / self.mean if self.mean != 0 else float('inf')
    
    @property
    def iqr(self) -> float:
        """Calculate interquartile range."""
        return self.percentiles[75] - self.percentiles[25]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'mean': self.mean,
            'median': self.median,
            'std': self.std,
            'var': self.var,
            'min': self.min,
            'max': self.max,
            'percentiles': self.percentiles,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'confidence_interval': self.confidence_interval,
            'cv': self.coefficient_of_variation,
            'iqr': self.iqr,
            'sample_size': self.sample_size
        }


@dataclass
class DistributionFit:
    """Results of distribution fitting."""
    
    distribution_name: str
    parameters: Dict[str, float]
    ks_statistic: float
    ks_pvalue: float
    aic: float
    bic: float
    log_likelihood: float
    
    @property
    def is_good_fit(self) -> bool:
        """Check if distribution is a good fit (p-value > 0.05)."""
        return self.ks_pvalue > 0.05


class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis for simulation results.
    
    Provides tools for analyzing distributions, calculating risk metrics,
    and extracting key insights from Monte Carlo simulations.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize statistical analyzer.
        
        Args:
            confidence_level: Confidence level for intervals (default 0.95)
        """
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def analyze_variable(
        self,
        data: Union[List[float], np.ndarray],
        variable_name: Optional[str] = None
    ) -> StatisticalSummary:
        """
        Perform comprehensive statistical analysis on a variable.
        
        Args:
            data: Data to analyze
            variable_name: Optional name for the variable
            
        Returns:
            StatisticalSummary with comprehensive metrics
        """
        # Convert to numpy array and remove NaN values
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            raise ValueError("No valid data points to analyze")
        
        # Calculate basic statistics
        mean = np.mean(data)
        median = np.median(data)
        std = np.std(data)
        var = np.var(data)
        min_val = np.min(data)
        max_val = np.max(data)
        
        # Calculate percentiles
        percentiles = {}
        for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
            percentiles[p] = np.percentile(data, p)
        
        # Calculate higher moments
        skewness = stats.skew(data)
        kurtosis = stats.kurtosis(data)
        
        # Calculate confidence interval
        se = std / np.sqrt(len(data))
        ci = stats.t.interval(
            self.confidence_level,
            len(data) - 1,
            loc=mean,
            scale=se
        )
        
        return StatisticalSummary(
            mean=mean,
            median=median,
            std=std,
            var=var,
            min=min_val,
            max=max_val,
            percentiles=percentiles,
            skewness=skewness,
            kurtosis=kurtosis,
            confidence_interval=ci,
            sample_size=len(data)
        )
    
    def calculate_var(
        self,
        data: Union[List[float], np.ndarray],
        confidence_level: Optional[float] = None
    ) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            data: Loss data
            confidence_level: Confidence level (uses instance default if None)
            
        Returns:
            VaR at specified confidence level
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        return np.percentile(data, confidence_level * 100)
    
    def calculate_cvar(
        self,
        data: Union[List[float], np.ndarray],
        confidence_level: Optional[float] = None
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR/Expected Shortfall).
        
        Args:
            data: Loss data
            confidence_level: Confidence level (uses instance default if None)
            
        Returns:
            CVaR at specified confidence level
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
        
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        var = self.calculate_var(data, confidence_level)
        return np.mean(data[data >= var])
    
    def fit_distribution(
        self,
        data: Union[List[float], np.ndarray],
        distributions: Optional[List[str]] = None
    ) -> Dict[str, DistributionFit]:
        """
        Fit various distributions to data and evaluate goodness of fit.
        
        Args:
            data: Data to fit
            distributions: List of distribution names to try
                          (default: ['norm', 'lognorm', 'gamma', 'expon'])
        
        Returns:
            Dictionary of distribution fits
        """
        if distributions is None:
            distributions = ['norm', 'lognorm', 'gamma', 'expon']
        
        data = np.array(data)
        data = data[~np.isnan(data)]
        
        results = {}
        
        for dist_name in distributions:
            try:
                # Get distribution
                dist = getattr(stats, dist_name)
                
                # Fit distribution
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    params = dist.fit(data)
                
                # Perform KS test
                ks_stat, ks_pval = stats.kstest(data, lambda x: dist.cdf(x, *params))
                
                # Calculate log-likelihood
                log_likelihood = np.sum(dist.logpdf(data, *params))
                
                # Calculate AIC and BIC
                n_params = len(params)
                n_data = len(data)
                aic = 2 * n_params - 2 * log_likelihood
                bic = n_params * np.log(n_data) - 2 * log_likelihood
                
                # Store results
                param_dict = {}
                param_names = dist.shapes.split(',') if dist.shapes else []
                param_names += ['loc', 'scale']
                for i, name in enumerate(param_names[:len(params)]):
                    param_dict[name.strip()] = params[i]
                
                results[dist_name] = DistributionFit(
                    distribution_name=dist_name,
                    parameters=param_dict,
                    ks_statistic=ks_stat,
                    ks_pvalue=ks_pval,
                    aic=aic,
                    bic=bic,
                    log_likelihood=log_likelihood
                )
            except Exception as e:
                # Skip distributions that fail to fit
                continue
        
        return results
    
    def analyze_time_series(
        self,
        data: Dict[int, List[float]],
        metric_name: str = "metric"
    ) -> Dict[str, Any]:
        """
        Analyze time series data from simulations.
        
        Args:
            data: Dictionary mapping years to lists of values
            metric_name: Name of the metric being analyzed
            
        Returns:
            Time series analysis results
        """
        years = sorted(data.keys())
        
        # Calculate statistics for each year
        yearly_stats = {}
        for year in years:
            yearly_stats[year] = self.analyze_variable(data[year])
        
        # Extract trends
        means = [yearly_stats[year].mean for year in years]
        medians = [yearly_stats[year].median for year in years]
        stds = [yearly_stats[year].std for year in years]
        
        # Calculate trend statistics
        mean_trend = np.polyfit(range(len(years)), means, 1)[0]
        median_trend = np.polyfit(range(len(years)), medians, 1)[0]
        volatility_trend = np.polyfit(range(len(years)), stds, 1)[0]
        
        return {
            'years': years,
            'yearly_statistics': yearly_stats,
            'trends': {
                'mean_trend': mean_trend,
                'median_trend': median_trend,
                'volatility_trend': volatility_trend
            },
            'aggregate_statistics': {
                'overall_mean': np.mean(means),
                'overall_std': np.mean(stds),
                'peak_year': years[np.argmax(means)],
                'trough_year': years[np.argmin(means)]
            }
        }
    
    def calculate_correlations(
        self,
        data: Dict[str, List[float]],
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix between variables.
        
        Args:
            data: Dictionary mapping variable names to data
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Correlation matrix as DataFrame
        """
        df = pd.DataFrame(data)
        return df.corr(method=method)
    
    def identify_outliers(
        self,
        data: Union[List[float], np.ndarray],
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> Dict[str, Any]:
        """
        Identify outliers in data.
        
        Args:
            data: Data to analyze
            method: Method to use ('iqr', 'zscore', 'mad')
            threshold: Threshold for outlier detection
            
        Returns:
            Dictionary with outlier information
        """
        data = np.array(data)
        data_clean = data[~np.isnan(data)]
        
        outliers = []
        outlier_indices = []
        
        if method == 'iqr':
            q1 = np.percentile(data_clean, 25)
            q3 = np.percentile(data_clean, 75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            for i, val in enumerate(data):
                if not np.isnan(val) and (val < lower_bound or val > upper_bound):
                    outliers.append(val)
                    outlier_indices.append(i)
        
        elif method == 'zscore':
            mean = np.mean(data_clean)
            std = np.std(data_clean)
            
            for i, val in enumerate(data):
                if not np.isnan(val):
                    z = abs((val - mean) / std)
                    if z > threshold:
                        outliers.append(val)
                        outlier_indices.append(i)
        
        elif method == 'mad':  # Median Absolute Deviation
            median = np.median(data_clean)
            mad = np.median(np.abs(data_clean - median))
            
            for i, val in enumerate(data):
                if not np.isnan(val):
                    deviation = abs((val - median) / mad) if mad != 0 else 0
                    if deviation > threshold:
                        outliers.append(val)
                        outlier_indices.append(i)
        
        return {
            'n_outliers': len(outliers),
            'outlier_percentage': len(outliers) / len(data_clean) * 100,
            'outlier_values': outliers,
            'outlier_indices': outlier_indices,
            'method': method,
            'threshold': threshold
        }
    
    def perform_sensitivity_analysis(
        self,
        base_results: Dict[str, float],
        sensitivity_results: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform sensitivity analysis on simulation results.
        
        Args:
            base_results: Base case results
            sensitivity_results: Dictionary of parameter variations and their results
            
        Returns:
            Sensitivity metrics for each parameter
        """
        sensitivity_metrics = {}
        
        for param_name, param_results in sensitivity_results.items():
            sensitivities = {}
            
            for metric_name, base_value in base_results.items():
                if metric_name in param_results:
                    # Calculate percentage change
                    param_value = param_results[metric_name]
                    if base_value != 0:
                        pct_change = (param_value - base_value) / base_value * 100
                    else:
                        pct_change = float('inf') if param_value != 0 else 0
                    
                    sensitivities[metric_name] = {
                        'base_value': base_value,
                        'param_value': param_value,
                        'absolute_change': param_value - base_value,
                        'percentage_change': pct_change
                    }
            
            sensitivity_metrics[param_name] = sensitivities
        
        return sensitivity_metrics
