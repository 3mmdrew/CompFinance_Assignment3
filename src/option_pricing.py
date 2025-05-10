import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate_temperature_paths(det_model_result, ar_model_result, start_date, end_date, 
                              n_paths=1000, base_temp=18.0):
    """
    Simulate temperature paths using Monte Carlo with Euler discretization.
    Uses a simplified approach without ordinal dates.
    
    Parameters:
    -----------
    det_model_result : dict
        Results from deterministic model fitting
    ar_model_result : dict
        Results from AR model fitting
    start_date : datetime
        Start date for simulation period
    end_date : datetime
        End date for simulation period
    n_paths : int, optional
        Number of paths to simulate
    base_temp : float, optional
        Base temperature for HDD calculations (°C)
        
    Returns:
    --------
    dict
        Dictionary containing simulation results
    """
    # Extract parameters
    params = det_model_result['parameters']
    ar_order = ar_model_result['best_order']
    ar_coeffs = ar_model_result['model'].params[1:ar_order+1].values
    sigma = np.std(ar_model_result['model'].resid)
    
    # Create date range for simulation period
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)
    
    print(f"Simulation period: {start_date} to {end_date} ({n_days} days)")
    
    # SIMPLIFIED APPROACH: Just use day of year for seasonal component
    # For the linear trend, we'll use simple sequential day numbers
    
    # Create days since start (for linear trend)
    days_index = np.arange(n_days)
    
    # Create day of year values (for seasonal component) - normalized to [0, 2π]
    day_of_year = np.array([date.dayofyear for date in dates])
    day_of_year_radians = 2 * np.pi * day_of_year / 365.25
    
    # Calculate deterministic component
    a, b, alpha, theta = params.values()
    
    print(f"Model parameters: a={a}, b={b}, alpha={alpha}, theta={theta}")
    
    # Deterministic component = intercept + trend + seasonal
    det_component = a + b * days_index + alpha * np.sin(day_of_year_radians + theta)
    
    print(f"Deterministic component - Min: {np.min(det_component):.2f}°C, Max: {np.max(det_component):.2f}°C, Mean: {np.mean(det_component):.2f}°C")
    
    # Initialize arrays for simulations
    temp_paths = np.zeros((n_paths, n_days))
    hdd_paths = np.zeros((n_paths, n_days))
    
    # Get last ar_order residuals from historical data to initialize
    last_residuals = det_model_result['residuals'][-ar_order:]
    
    iterator = tqdm(range(n_paths), desc="Simulating temperature paths")
    for i in iterator:
        # Start with last known residuals
        residuals = last_residuals.copy()
        
        # Simulate for each day
        for t in range(n_days):
            # Calculate next residual using AR model
            next_residual = np.sum(ar_coeffs * residuals[-ar_order:][::-1]) + np.random.normal(0, sigma)
            
            # Update residuals history
            residuals = np.append(residuals[1:], next_residual)
            
            # Total temperature is deterministic + stochastic
            temp_paths[i, t] = det_component[t] + next_residual
            
            # Calculate heating degree days for this temperature
            hdd_paths[i, t] = max(base_temp - temp_paths[i, t], 0)
    
    # Calculate statistics on zero HDDs
    zero_hdd_prop = np.mean(hdd_paths == 0)
    print(f"Proportion of days with zero HDDs: {zero_hdd_prop:.2%}")
    
    # Create a Dictionary for easy access to simulated data
    results = {
        'dates': dates,
        'deterministic': det_component,
        'temp_paths': temp_paths,
        'hdd_paths': hdd_paths,
        'base_temp': base_temp,
        'zero_hdd_proportion': zero_hdd_prop
    }
    
    return results
def calculate_cumulative_degree_days(simulation_results):
    """
    Calculate cumulative heating degree days from simulation results.
    
    Parameters:
    -----------
    simulation_results : dict
        Results from simulate_temperature_paths
        
    Returns:
    --------
    numpy.ndarray
        Array of cumulative heating degree days for each path
    """
    daily_hdd = simulation_results['hdd_paths']
    
    # Calculate cumulative sum for each path
    cumulative_hdd = np.sum(daily_hdd, axis=1)
    
    return cumulative_hdd

def call_option_payoff(cumulative_hdd, strike, tick_size=1.0, cap=float('inf')):
    """
    Calculate call option payoff for HDD contract.
    
    Parameters:
    -----------
    cumulative_hdd : array-like
        Cumulative heating degree days
    strike : float
        Strike price (K)
    tick_size : float, optional
        Notional value per degree day (N)
    cap : float, optional
        Maximum payoff
        
    Returns:
    --------
    array-like
        Option payoffs
    """
    return np.minimum(tick_size * np.maximum(cumulative_hdd - strike, 0), cap)

def put_option_payoff(cumulative_hdd, strike, tick_size=1.0, floor=float('inf')):
    """
    Calculate put option payoff for HDD contract.
    
    Parameters:
    -----------
    cumulative_hdd : array-like
        Cumulative heating degree days
    strike : float
        Strike price (K)
    tick_size : float, optional
        Notional value per degree day (N)
    floor : float, optional
        Maximum payoff
        
    Returns:
    --------
    array-like
        Option payoffs
    """
    return np.minimum(tick_size * np.maximum(strike - cumulative_hdd, 0), floor)

def collar_option_payoff(cumulative_hdd, call_strike, put_strike, 
                          call_tick=1.0, put_tick=1.0, cap=float('inf'), floor=float('inf')):
    """
    Calculate collar option payoff for HDD contract.
    
    Parameters:
    -----------
    cumulative_hdd : array-like
        Cumulative heating degree days
    call_strike : float
        Strike price for call component (K1)
    put_strike : float
        Strike price for put component (K2)
    call_tick : float, optional
        Notional value for call component (α)
    put_tick : float, optional
        Notional value for put component (β)
    cap : float, optional
        Maximum payoff for call component
    floor : float, optional
        Maximum payoff for put component
        
    Returns:
    --------
    array-like
        Option payoffs
    """
    call_payoff = np.minimum(call_tick * np.maximum(cumulative_hdd - call_strike, 0), cap)
    put_payoff = np.minimum(put_tick * np.maximum(put_strike - cumulative_hdd, 0), floor)
    
    return call_payoff - put_payoff

def price_option(payoffs, risk_free_rate, time_to_maturity):
    """
    Price an option based on Monte Carlo simulated payoffs.
    Implements equation (26) from the problem statement.
    
    Parameters:
    -----------
    payoffs : array-like
        Option payoffs from simulations
    risk_free_rate : float
        Annual risk-free interest rate (decimal)
    time_to_maturity : float
        Time to maturity in years
        
    Returns:
    --------
    float
        Option price
    """
    # Calculate present value of expected payoff
    discount_factor = np.exp(-risk_free_rate * time_to_maturity)
    option_price = discount_factor * np.mean(payoffs)
    
    # Calculate standard error
    payoff_std = np.std(payoffs)
    n_paths = len(payoffs)
    std_error = payoff_std / np.sqrt(n_paths) * discount_factor
    
    return option_price, std_error

def plot_degree_day_distribution(cumulative_hdd, base_temp):
    """
    Plot distribution of cumulative heating degree days from simulations.
    
    Parameters:
    -----------
    cumulative_hdd : array-like
        Cumulative heating degree days from simulations
    base_temp : float
        Base temperature used for degree day calculation
        
    Returns:
    --------
    None
        Displays plot
    """
    plt.figure(figsize=(12, 7))
    
    # Plot histogram with frequency instead of density
    n, bins, patches = plt.hist(cumulative_hdd, bins=50, alpha=0.7, density=False)
    
    # Add statistics
    mean_dd = np.mean(cumulative_hdd)
    std_dd = np.std(cumulative_hdd)
    
    plt.axvline(x=mean_dd, color='black', linestyle='--', linewidth=1.5,
                label=f'Mean: {mean_dd:.2f}')
    
    # Add text with statistics
    stats_text = f"Mean: {mean_dd:.2f}\nStd Dev: {std_dd:.2f}\nMin: {np.min(cumulative_hdd):.2f}\nMax: {np.max(cumulative_hdd):.2f}"
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 ha='left', va='top')
    
    plt.title(f'Distribution of Cumulative HDD (Base Temp: {base_temp}°C)')
    plt.xlabel(f'Cumulative HDD')
    plt.ylabel('Frequency')  # Changed from 'Probability Density' to 'Frequency'
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_option_payoff_distribution(payoffs, option_type, strike, price, std_error):
    """
    Plot distribution of option payoffs from simulations.
    
    Parameters:
    -----------
    payoffs : array-like
        Option payoffs from simulations
    option_type : str
        Type of option ('Call', 'Put', or 'Collar')
    strike : float or tuple
        Strike price(s)
    price : float
        Calculated option price
    std_error : float
        Standard error of the price estimate
        
    Returns:
    --------
    None
        Displays plot
    """
    plt.figure(figsize=(12, 7))
    
    # Plot histogram with frequency instead of density
    n, bins, patches = plt.hist(payoffs, bins=50, alpha=0.7, density=False)
    
    # Add statistics
    mean_payoff = np.mean(payoffs)
    std_payoff = np.std(payoffs)
    
    plt.axvline(x=mean_payoff, color='black', linestyle='--', linewidth=1.5,
                label=f'Mean: {mean_payoff:.2f}')
    
    # Add text with statistics
    stats_text = (f"Mean: {mean_payoff:.2f}\nStd Dev: {std_payoff:.2f}\n"
                 f"Min: {np.min(payoffs):.2f}\nMax: {np.max(payoffs):.2f}\n"
                 f"Option Price: {price:.2f} ± {std_error:.2f}")
    
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                ha='left', va='top')
    
    # Add strike information
    if option_type == 'Collar':
        strike_info = f"Call Strike: {strike[0]}, Put Strike: {strike[1]}"
    else:
        strike_info = f"Strike: {strike}"
    
    plt.title(f'HDD {option_type} Option Payoff Distribution\n{strike_info}')
    plt.xlabel('Option Payoff')
    plt.ylabel('Frequency')  # Changed from 'Probability Density' to 'Frequency'
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def calculate_option_metrics(payoffs):
    """
    Calculate various risk metrics for an option.
    
    Parameters:
    -----------
    payoffs : array-like
        Option payoffs from simulations
        
    Returns:
    --------
    dict
        Dictionary of risk metrics
    """
    metrics = {
        'mean': np.mean(payoffs),
        'std': np.std(payoffs),
        'min': np.min(payoffs),
        'max': np.max(payoffs),
        'VaR_95': np.percentile(payoffs, 5),  # 95% Value at Risk
        'VaR_99': np.percentile(payoffs, 1),  # 99% Value at Risk
        'zero_probability': np.mean(payoffs == 0)  # Probability of zero payoff
    }
    
    return metrics

def plot_temperature_paths(simulation_results, n_paths_to_show=10):
    """
    Plot simulated temperature paths.
    
    Parameters:
    -----------
    simulation_results : dict
        Results from simulate_temperature_paths
    n_paths_to_show : int, optional
        Number of paths to display
        
    Returns:
    --------
    None
        Displays plot
    """
    plt.figure(figsize=(14, 7))
    
    # Get data
    dates = simulation_results['dates']
    temp_paths = simulation_results['temp_paths']
    det_component = simulation_results['deterministic']
    
    # Plot subset of paths
    for i in range(min(n_paths_to_show, temp_paths.shape[0])):
        plt.plot(dates, temp_paths[i], alpha=0.5, linewidth=0.8)
    
    # Plot deterministic component
    plt.plot(dates, det_component, 'r-', linewidth=2, label='Deterministic Trend')
    
    # Add base temperature line
    base_temp = simulation_results['base_temp']
    plt.axhline(y=base_temp, color='black', linestyle='--', linewidth=1.5,
                label=f'Base Temperature ({base_temp}°C)')
    
    plt.title('Simulated Temperature Paths')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cumulative_hdd_paths(simulation_results, n_paths_to_show=10):
    """
    Plot cumulative heating degree day paths.
    
    Parameters:
    -----------
    simulation_results : dict
        Results from simulate_temperature_paths
    n_paths_to_show : int, optional
        Number of paths to display
        
    Returns:
    --------
    None
        Displays plot
    """
    plt.figure(figsize=(14, 7))
    
    # Get data
    dates = simulation_results['dates']
    hdd_paths = simulation_results['hdd_paths']
    
    # Calculate cumulative sums
    cum_hdd_paths = np.cumsum(hdd_paths, axis=1)
    
    # Plot subset of paths
    for i in range(min(n_paths_to_show, cum_hdd_paths.shape[0])):
        plt.plot(dates, cum_hdd_paths[i], alpha=0.5, linewidth=0.8)
    
    # Calculate and plot mean path
    mean_path = np.mean(cum_hdd_paths, axis=0)
    plt.plot(dates, mean_path, 'r-', linewidth=2, label='Mean Path')
    
    # Calculate and plot 5% and 95% quantiles
    q05_path = np.percentile(cum_hdd_paths, 5, axis=0)
    q95_path = np.percentile(cum_hdd_paths, 95, axis=0)
    
    plt.fill_between(dates, q05_path, q95_path, alpha=0.2, color='red',
                     label='90% Confidence Interval')
    
    plt.title('Cumulative Heating Degree Days (HDD) Paths')
    plt.xlabel('Date')
    plt.ylabel('Cumulative HDD')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
