import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox

def plot_residual_histogram(residuals):
    """
    Plot histogram of residuals.
    
    Parameters:
    -----------
    residuals : pandas.Series
        Residual values from time series decomposition
        
    Returns:
    --------
    None
        Displays plot
    """
    # Drop NaN values
    cleaned_residuals = residuals.dropna()
    
    plt.figure(figsize=(10, 6))
    plt.hist(cleaned_residuals, bins=60, density=True, alpha=0.7)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

def plot_qq(residuals):
    """
    Create Q-Q plot for residuals.
    
    Parameters:
    -----------
    residuals : pandas.Series
        Residual values from time series decomposition
        
    Returns:
    --------
    None
        Displays plot
    """
    # Drop NaN values
    cleaned_residuals = residuals.dropna()
    
    plt.figure(figsize=(10, 6))
    stats.probplot(cleaned_residuals, dist="norm", plot=plt)
    plt.title("Q-Q Plot of Residuals")
    plt.grid(True)
    plt.show()

def plot_acf_pacf(residuals, lags=50):
    """
    Plot ACF and PACF for residuals.
    
    Parameters:
    -----------
    residuals : pandas.Series
        Residual values from time series decomposition
    lags : int, optional
        Number of lags to include in plots
        
    Returns:
    --------
    None
        Displays plot
    """
    # Drop NaN values
    cleaned_residuals = residuals.dropna()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    plot_acf(cleaned_residuals, ax=axes[0], lags=lags)
    axes[0].set_title('Autocorrelation Function (ACF)')
    axes[0].grid(True)
    
    plot_pacf(cleaned_residuals, ax=axes[1], lags=lags)
    axes[1].set_title('Partial Autocorrelation Function (PACF)')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

def perform_adf_test(series):
    """
    Perform Augmented Dickey-Fuller test for stationarity.
    
    Parameters:
    -----------
    series : pandas.Series
        Time series data
        
    Returns:
    --------
    dict
        Dictionary containing test results
    """
    # Drop NaN values
    cleaned_series = series.dropna()
    
    result = adfuller(cleaned_series)
    
    adf_result = {
        'adf_statistic': result[0],
        'p_value': result[1],
        'critical_values': result[4],
        'is_stationary': result[1] <= 0.05
    }
    
    # Print results
    print('ADF Statistic: %f' % adf_result['adf_statistic'])
    print('p-value: %f' % adf_result['p_value'])
    print('Critical Values:')
    for key, value in adf_result['critical_values'].items():
        print(f'   {key}: {value}')
    
    if adf_result['is_stationary']:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data is non-stationary")
    
    return adf_result

def check_residual_autocorrelation(residuals, lags=[10, 20, 30]):
    """
    Check for autocorrelation in residuals using Ljung-Box test.
    
    Parameters:
    -----------
    residuals : pandas.Series
        Residual values from time series decomposition
    lags : list, optional
        List of lags to test
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with test results
    """
    # Drop NaN values
    cleaned_residuals = residuals.dropna()
    
    lb_test = acorr_ljungbox(cleaned_residuals, lags=lags)
    
    print("Ljung-Box Test for Autocorrelation:")
    print(lb_test)
    
    return lb_test