import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_time_series_with_rolling_stats(df, column, window=30):
    """
    Plot time series with rolling statistics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with time series data
    column : str
        Column name of the time series
    window : int, optional
        Window size for rolling statistics
        
    Returns:
    --------
    None
        Displays plot
    """
    rolling_mean = df[column].rolling(window=window).mean()
    rolling_std = df[column].rolling(window=window).std()
    
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df[column], label=column)
    plt.plot(rolling_mean.index, rolling_mean, label=f'{window}-day Rolling Mean', color='red')
    plt.plot(rolling_std.index, rolling_std, label=f'{window}-day Rolling Std', color='green')
    plt.title(f'{column} Time Series with Rolling Statistics')
    plt.xlabel('Date')
    plt.ylabel(column)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def decompose_time_series(df, column, period=365, model='additive'):
    """
    Decompose time series into trend, seasonal, and residual components.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with time series data
    column : str
        Column name of the time series
    period : int, optional
        Period of seasonality
    model : str, optional
        Type of decomposition model ('additive' or 'multiplicative')
        
    Returns:
    --------
    object
        Decomposition results object with trend, seasonal, and residual attributes
    """
    # Perform decomposition
    decompose_result = seasonal_decompose(df[column], model=model, period=period)
    
    return decompose_result

def plot_decomposition(decompose_result, original_series):
    """
    Plot decomposition components.
    
    Parameters:
    -----------
    decompose_result : object
        Decomposition results from seasonal_decompose
    original_series : pandas.Series
        Original time series data
        
    Returns:
    --------
    None
        Displays plot
    """
    trend = decompose_result.trend
    seasonal = decompose_result.seasonal
    residual = decompose_result.resid
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 16))
    
    axes[0].plot(trend.index, trend)
    axes[0].set_title('Trend Component')
    axes[0].grid(True)
    
    axes[1].plot(seasonal.index, seasonal)
    axes[1].set_title('Seasonal Component')
    axes[1].grid(True)
    
    axes[2].plot(residual.index, residual)
    axes[2].set_title('Residual Component')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return trend, seasonal, residual