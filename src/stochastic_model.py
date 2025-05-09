import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

def fit_arima_model(residuals, order=(1, 0, 0)):
    """
    Fit ARIMA model to residuals.
    
    Parameters:
    -----------
    residuals : pandas.Series
        Residual values from time series decomposition
    order : tuple, optional
        Order of ARIMA model (p, d, q)
        
    Returns:
    --------
    model
        Fitted ARIMA model
    """
    # Drop NaN values
    cleaned_residuals = residuals.dropna()
    
    # Fit ARIMA model
    model = ARIMA(cleaned_residuals, order=order)
    model_fit = model.fit()
    
    print(model_fit.summary())
    
    return model_fit

def simulate_temperature(trend, seasonal, arima_model, n_steps=365, n_simulations=5):
    """
    Simulate future temperature data using trend, seasonal components, and stochastic model.
    
    Parameters:
    -----------
    trend : pandas.Series
        Trend component from time series decomposition
    seasonal : pandas.Series
        Seasonal component from time series decomposition
    arima_model : model
        Fitted ARIMA model for residuals
    n_steps : int, optional
        Number of steps to forecast
    n_simulations : int, optional
        Number of simulations to run
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with simulated temperature data
    """
    # Extract the last values of trend and calculate slope
    last_trend_values = trend.dropna().iloc[-100:].values
    trend_slope = np.polyfit(np.arange(len(last_trend_values)), last_trend_values, 1)[0]
    
    # Get the last trend value
    last_trend = trend.dropna().iloc[-1]
    
    # Create future trend values
    future_trend = [last_trend + i * trend_slope for i in range(1, n_steps + 1)]
    
    # Get one cycle of seasonality
    season_cycle = seasonal.dropna().iloc[:365].values
    
    # Extend seasonality for future dates
    future_seasonal = np.tile(season_cycle, int(np.ceil(n_steps / 365)))[:n_steps]
    
    # Dates for future predictions
    last_date = seasonal.dropna().index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_steps, freq='D')
    
    # Run multiple simulations
    simulations = pd.DataFrame(index=future_dates)
    
    for i in range(n_simulations):
        # Simulate residuals using the ARIMA model
        simulated_residuals = arima_model.simulate(n_steps)
        
        # Combine components
        simulated_temp = future_trend + future_seasonal + simulated_residuals
        
        # Add to simulations dataframe
        simulations[f'sim_{i+1}'] = simulated_temp
    
    # Calculate mean and confidence intervals
    simulations['mean'] = simulations.mean(axis=1)
    simulations['lower_ci'] = simulations.iloc[:, :-1].quantile(0.025, axis=1)
    simulations['upper_ci'] = simulations.iloc[:, :-1].quantile(0.975, axis=1)
    
    return simulations

def plot_simulations(original_data, simulations, column='temperature_2m_mean'):
    """
    Plot original data and simulations.
    
    Parameters:
    -----------
    original_data : pandas.DataFrame
        Original temperature data
    simulations : pandas.DataFrame
        Simulated temperature data
    column : str, optional
        Column name in original data
        
    Returns:
    --------
    None
        Displays plot
    """
    plt.figure(figsize=(14, 7))
    
    # Plot historical data
    plt.plot(original_data.index, original_data[column], 
             label='Historical Data', color='black', linewidth=1.5)
    
    # Plot individual simulations
    sim_cols = [col for col in simulations.columns if col.startswith('sim_')]
    for col in sim_cols:
        plt.plot(simulations.index, simulations[col], 
                 alpha=0.3, linewidth=0.5, color='blue')
    
    # Plot mean and confidence intervals
    plt.plot(simulations.index, simulations['mean'], 
             label='Mean Forecast', color='red', linewidth=1.5)
    plt.fill_between(simulations.index, 
                     simulations['lower_ci'], 
                     simulations['upper_ci'], 
                     color='red', alpha=0.2, label='95% Confidence Interval')
    
    plt.title('Temperature Forecasts with Stochastic Model')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()