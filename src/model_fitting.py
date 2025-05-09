import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

def deterministic_model(x, a, b, alpha, theta):
    """
    Deterministic model for temperature with trend and seasonality.
    
    Parameters:
    -----------
    x : array-like
        Time points (in days)
    a : float
        Intercept
    b : float
        Trend coefficient
    alpha : float
        Amplitude of seasonal component
    theta : float
        Phase shift
        
    Returns:
    --------
    array-like
        Model predictions
    """
    omega = 2 * np.pi / 365.25
    return a + b * x + alpha * np.sin(omega * x + theta)

def fit_deterministic_model(df, column='temperature_2m_mean'):
    """
    Fit deterministic seasonal model to temperature data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with temperature data
    column : str, optional
        Column name of temperature data
        
    Returns:
    --------
    dict
        Dictionary containing model parameters, fitted values, and residuals
    """
    # Convert dates to ordinal numbers
    first_ord = df.index[0].toordinal()
    xdata = np.array([date.toordinal() - first_ord for date in df.index])
    ydata = df[column].values
    
    # Initial parameter guesses
    p0 = [np.mean(ydata), 0, np.std(ydata), 0]
    
    # Fit the model
    params, cov = curve_fit(deterministic_model, xdata, ydata, p0=p0, method='lm')
    
    # Extract parameters
    a, b, alpha, theta = params
    
    # Compute fitted values
    fitted_values = deterministic_model(xdata, *params)
    
    # Compute residuals
    residuals = ydata - fitted_values
    
    # Create result dictionary
    result = {
        'parameters': {
            'a': a,
            'b': b,
            'alpha': alpha,
            'theta': theta
        },
        'xdata': xdata,
        'fitted_values': fitted_values,
        'residuals': residuals,
        'cov': cov
    }
    
    # Print parameter estimates
    print("Estimated parameters:")
    print(f"a (intercept): {a:.4f}")
    print(f"b (trend): {b:.6f}")
    print(f"alpha (amplitude): {alpha:.4f}")
    print(f"theta (phase): {theta:.4f}")
    
    return result

def plot_deterministic_fit(df, fit_result, column='temperature_2m_mean'):
    """
    Plot original data with fitted deterministic model.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with temperature data
    fit_result : dict
        Dictionary from fit_deterministic_model
    column : str, optional
        Column name of temperature data
        
    Returns:
    --------
    None
        Displays plot
    """
    plt.figure(figsize=(14, 7))
    
    # Plot original data
    plt.plot(df.index, df[column], label='Observed Temperature', alpha=0.7)
    
    # Plot fitted values
    plt.plot(df.index, fit_result['fitted_values'], 
             label='Deterministic Model Fit', color='red', linewidth=2)
    
    plt.title('Temperature Data with Deterministic Model Fit')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot residuals
    plt.figure(figsize=(14, 5))
    plt.plot(df.index, fit_result['residuals'], color='green')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.title('Residuals from Deterministic Model')
    plt.xlabel('Date')
    plt.ylabel('Residual (°C)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def fit_ar_model(residuals, max_order=10):
    """
    Fit autoregressive model to residuals and select optimal order using AIC.
    
    Parameters:
    -----------
    residuals : array-like
        Residuals from deterministic model
    max_order : int, optional
        Maximum AR order to consider
        
    Returns:
    --------
    dict
        Dictionary containing AR model results
    """
    # Create a pandas Series from residuals
    residual_series = pd.Series(residuals)
    
    # Initialize variables to store results
    best_order = 1
    best_aic = float('inf')
    best_model = None
    
    # Try different AR orders
    aic_values = []
    for p in range(1, max_order + 1):
        model = AutoReg(residual_series.dropna(), lags=p)
        model_fit = model.fit()
        aic = model_fit.aic
        aic_values.append((p, aic))
        
        if aic < best_aic:
            best_aic = aic
            best_order = p
            best_model = model_fit
    
    # Print results
    print(f"Best AR order (p) based on AIC: {best_order}")
    print(f"AIC value: {best_aic:.4f}")
    print("\nAutoregressive model summary:")
    print(best_model.summary())
    
    # Calculate mean-reversion parameter (kappa)
    ar_coef = best_model.params[1]  # First AR coefficient
    kappa = -np.log(ar_coef)
    half_life = np.log(2) / kappa if kappa > 0 else float('inf')
    
    print(f"\nMean-reversion parameter (κ): {kappa:.4f}")
    print(f"Half-life of mean reversion: {half_life:.2f} days")
    
    # Plot AIC values
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_order + 1), [aic for _, aic in aic_values], 'o-')
    plt.axvline(x=best_order, color='red', linestyle='--', 
                label=f'Best order = {best_order}')
    plt.title('AIC Values for Different AR Orders')
    plt.xlabel('AR Order (p)')
    plt.ylabel('AIC Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Return results
    return {
        'best_order': best_order,
        'best_aic': best_aic,
        'model': best_model,
        'kappa': kappa,
        'half_life': half_life,
        'aic_values': aic_values
    }

def plot_ar_model_diagnostics(ar_model):
    """
    Plot diagnostics for AR model.
    
    Parameters:
    -----------
    ar_model : model
        Fitted AR model
        
    Returns:
    --------
    None
        Displays plots
    """
    # Get residuals from the AR model
    ar_residuals = ar_model.resid
    
    # Plot AR model residuals
    plt.figure(figsize=(14, 5))
    plt.plot(ar_residuals.index, ar_residuals)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('AR Model Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residual')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot ACF and PACF of AR model residuals
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    sm.graphics.tsa.plot_acf(ar_residuals.dropna(), lags=40, ax=axes[0])
    axes[0].set_title('ACF of AR Model Residuals')
    axes[0].grid(True)
    
    sm.graphics.tsa.plot_pacf(ar_residuals.dropna(), lags=40, ax=axes[1])
    axes[1].set_title('PACF of AR Model Residuals')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Q-Q plot
    plt.figure(figsize=(10, 6))
    sm.qqplot(ar_residuals.dropna(), line='45', fit=True)
    plt.title('Q-Q Plot of AR Model Residuals')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Histogram
    plt.figure(figsize=(10, 6))
    plt.hist(ar_residuals.dropna(), bins=40, density=True, alpha=0.7)
    
    # Add normal distribution curve
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = np.exp(-0.5 * ((x - np.mean(ar_residuals.dropna())) / np.std(ar_residuals.dropna()))**2) / (np.std(ar_residuals.dropna()) * np.sqrt(2 * np.pi))
    plt.plot(x, p, 'k', linewidth=2)
    
    plt.title('Distribution of AR Model Residuals')
    plt.xlabel('Residual')
    plt.ylabel('Density')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def simulate_temperature_with_ar(df, det_model_result, ar_model_result, 
                               n_steps=365, n_simulations=10, column='temperature_2m_mean'):
    """
    Simulate future temperature using deterministic model and AR process.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with temperature data
    det_model_result : dict
        Result from fit_deterministic_model
    ar_model_result : dict
        Result from fit_ar_model
    n_steps : int, optional
        Number of days to simulate
    n_simulations : int, optional
        Number of simulations to run
    column : str, optional
        Column name of temperature data
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with simulated temperatures
    """
    # Extract parameters and models
    params = det_model_result['parameters']
    ar_model = ar_model_result['model']
    ar_order = ar_model_result['best_order']
    
    # Get the last date in the dataset
    last_date = df.index[-1]
    
    # Create future dates
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), 
                                periods=n_steps, freq='D')
    
    # Get the last AR_ORDER residuals
    last_residuals = df[column].values[-ar_order:] - det_model_result['fitted_values'][-ar_order:]
    
    # Create dataframe for simulations
    simulations = pd.DataFrame(index=future_dates)
    
    # Calculate future ordinal days
    first_ord = df.index[0].toordinal()
    future_xdata = np.array([date.toordinal() - first_ord for date in future_dates])
    
    # Calculate deterministic component for future dates
    future_deterministic = deterministic_model(future_xdata, 
                                            params['a'], 
                                            params['b'], 
                                            params['alpha'], 
                                            params['theta'])
    
    # Get AR coefficients
    ar_coeffs = ar_model.params[1:ar_order+1].values
    
    # Get standard deviation of AR residuals
    sigma = np.std(ar_model.resid)
    
    # Run simulations
    for sim in range(n_simulations):
        # Initialize with last known residuals
        sim_residuals = last_residuals.copy()
        future_residuals = []
        
        # Simulate future residuals
        for t in range(n_steps):
            # Calculate next residual based on AR model
            next_residual = np.sum(ar_coeffs * sim_residuals[-ar_order:][::-1]) + np.random.normal(0, sigma)
            future_residuals.append(next_residual)
            sim_residuals = np.append(sim_residuals[1:], next_residual)
        
        # Calculate total future temperature
        future_temps = future_deterministic + future_residuals
        
        # Add to simulations dataframe
        simulations[f'sim_{sim+1}'] = future_temps
    
    # Calculate statistics
    simulations['deterministic'] = future_deterministic
    simulations['mean'] = simulations.filter(regex='^sim_').mean(axis=1)
    simulations['std'] = simulations.filter(regex='^sim_').std(axis=1)
    simulations['lower_ci'] = simulations['mean'] - 1.96 * simulations['std']
    simulations['upper_ci'] = simulations['mean'] + 1.96 * simulations['std']
    
    return simulations

def plot_temperature_forecast(df, simulations, column='temperature_2m_mean'):
    """
    Plot historical data and temperature forecasts.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with historical temperature data
    simulations : pandas.DataFrame
        Dataframe with simulated temperatures
    column : str, optional
        Column name of temperature data
        
    Returns:
    --------
    None
        Displays plot
    """
    plt.figure(figsize=(16, 8))
    
    # Plot historical data
    plt.plot(df.index, df[column], 
             label='Historical Temperature', color='black', linewidth=1.5)
    
    # Plot deterministic forecast
    plt.plot(simulations.index, simulations['deterministic'], 
             label='Deterministic Forecast', color='blue', linewidth=1.5)
    
    # Plot simulations
    for col in simulations.filter(regex='^sim_').columns:
        plt.plot(simulations.index, simulations[col], 
                 alpha=0.2, color='gray', linewidth=0.5)
    
    # Plot mean forecast and confidence interval
    plt.plot(simulations.index, simulations['mean'], 
             label='Mean Forecast', color='red', linewidth=1.5)
    plt.fill_between(simulations.index, 
                     simulations['lower_ci'], 
                     simulations['upper_ci'], 
                     alpha=0.3, color='red', 
                     label='95% Confidence Interval')
    
    # Customize plot
    plt.title('Temperature Forecast with Deterministic-AR Model')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend(loc='best')
    plt.grid(True)
    
    # Add vertical line separating historical and forecast data
    plt.axvline(x=df.index[-1], color='black', linestyle='--', alpha=0.7)
    plt.text(df.index[-1], plt.ylim()[0] + 0.05 * (plt.ylim()[1] - plt.ylim()[0]), 
             ' Historical | Forecast ', 
             ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()