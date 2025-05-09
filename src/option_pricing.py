import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulate_temperature_paths(det_model_result, ar_model_result, start_date, end_date, 
                              n_paths=1000, base_temp=18.0):
    """
    Simulate temperature paths using Monte Carlo with Euler discretization.
    
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
        Base temperature for degree day calculations (°C)
        
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
    
    # Reference date for ordinal calculation (same as in det_model_result)
    first_ord = det_model_result['xdata'][0] - (det_model_result['xdata'][1] - det_model_result['xdata'][0])
    
    # Calculate ordinal days for simulation period
    xdata = np.array([date.toordinal() - first_ord for date in dates])
    
    # Calculate deterministic component
    a, b, alpha, theta = params.values()
    det_component = a + b * xdata + alpha * np.sin(2 * np.pi / 365.25 * xdata + theta)
    
    # Initialize arrays for simulations
    temp_paths = np.zeros((n_paths, n_days))
    hdd_paths = np.zeros((n_paths, n_days))
    cdd_paths = np.zeros((n_paths, n_days))
    
    # Get last ar_order residuals from historical data to initialize
    last_residuals = det_model_result['residuals'][-ar_order:]
    
    # Run simulations with progress bar
    for i in tqdm(range(n_paths), desc="Simulating temperature paths"):
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
            
            # Calculate degree days for this temperature
            hdd_paths[i, t] = max(base_temp - temp_paths[i, t], 0)  # Heating degree days
            cdd_paths[i, t] = max(temp_paths[i, t] - base_temp, 0)  # Cooling degree days
    
    # Create a DataFrame for easy access to simulated data
    results = {
        'dates': dates,
        'deterministic': det_component,
        'temp_paths': temp_paths,
        'hdd_paths': hdd_paths,
        'cdd_paths': cdd_paths,
        'base_temp': base_temp
    }
    
    return results

def calculate_cumulative_degree_days(simulation_results, degree_day_type='hdd'):
    """
    Calculate cumulative degree days from simulation results.
    
    Parameters:
    -----------
    simulation_results : dict
        Results from simulate_temperature_paths
    degree_day_type : str, optional
        Type of degree days to calculate ('hdd' or 'cdd')
        
    Returns:
    --------
    numpy.ndarray
        Array of cumulative degree days for each path
    """
    if degree_day_type.lower() == 'hdd':
        daily_dd = simulation_results['hdd_paths']
    elif degree_day_type.lower() == 'cdd':
        daily_dd = simulation_results['cdd_paths']
    else:
        raise ValueError("degree_day_type must be 'hdd' or 'cdd'")
    
    # Calculate cumulative sum for each path
    cumulative_dd = np.sum(daily_dd, axis=1)
    
    return cumulative_dd

def call_option_payoff(cumulative_dd, strike, alpha=1.0, cap=float('inf')):
    """
    Calculate call option payoff.
    
    Parameters:
    -----------
    cumulative_dd : array-like
        Cumulative degree days
    strike : float
        Strike price
    alpha : float, optional
        Multiplier
    cap : float, optional
        Maximum payoff
        
    Returns:
    --------
    array-like
        Option payoffs
    """
    return np.minimum(alpha * np.maximum(cumulative_dd - strike, 0), cap)

def put_option_payoff(cumulative_dd, strike, alpha=1.0, floor=float('inf')):
    """
    Calculate put option payoff.
    
    Parameters:
    -----------
    cumulative_dd : array-like
        Cumulative degree days
    strike : float
        Strike price
    alpha : float, optional
        Multiplier
    floor : float, optional
        Maximum payoff
        
    Returns:
    --------
    array-like
        Option payoffs
    """
    return np.minimum(alpha * np.maximum(strike - cumulative_dd, 0), floor)

def collar_option_payoff(cumulative_dd, strike1, strike2, alpha=1.0, beta=1.0, cap=float('inf'), floor=float('inf')):
    """
    Calculate collar option payoff.
    
    Parameters:
    -----------
    cumulative_dd : array-like
        Cumulative degree days
    strike1 : float
        Strike price for call component
    strike2 : float
        Strike price for put component
    alpha : float, optional
        Multiplier for call component
    beta : float, optional
        Multiplier for put component
    cap : float, optional
        Maximum payoff for call component
    floor : float, optional
        Maximum payoff for put component
        
    Returns:
    --------
    array-like
        Option payoffs
    """
    call_payoff = np.minimum(alpha * np.maximum(cumulative_dd - strike1, 0), cap)
    put_payoff = np.minimum(beta * np.maximum(strike2 - cumulative_dd, 0), floor)
    
    return call_payoff - put_payoff

def price_option(payoffs, risk_free_rate, time_to_maturity):
    """
    Price an option based on Monte Carlo simulated payoffs.
    
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
    
    return option_price

def plot_degree_day_distribution(cumulative_dd, option_type, base_temp):
    """
    Plot distribution of cumulative degree days from simulations.
    
    Parameters:
    -----------
    cumulative_dd : array-like
        Cumulative degree days from simulations
    option_type : str
        Type of degree days ('HDD' or 'CDD')
    base_temp : float
        Base temperature used for degree day calculation
        
    Returns:
    --------
    None
        Displays plot
    """
    plt.figure(figsize=(12, 7))
    
    # Plot histogram
    n, bins, patches = plt.hist(cumulative_dd, bins=50, alpha=0.7, density=True)
    
    # Add kernel density estimate
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(cumulative_dd)
    x = np.linspace(min(cumulative_dd), max(cumulative_dd), 1000)
    plt.plot(x, kde(x), 'r-', linewidth=2)
    
    # Add statistics
    mean_dd = np.mean(cumulative_dd)
    std_dd = np.std(cumulative_dd)
    
    plt.axvline(x=mean_dd, color='black', linestyle='--', linewidth=1.5,
                label=f'Mean: {mean_dd:.2f}')
    
    # Add text with statistics
    stats_text = f"Mean: {mean_dd:.2f}\nStd Dev: {std_dd:.2f}\nMin: {np.min(cumulative_dd):.2f}\nMax: {np.max(cumulative_dd):.2f}"
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 ha='left', va='top')
    
    plt.title(f'Distribution of Cumulative {option_type} (Base Temp: {base_temp}°C)')
    plt.xlabel(f'Cumulative {option_type}')
    plt.ylabel('Probability Density')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_option_payoff_distribution(payoffs, option_type, strike, price):
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
        
    Returns:
    --------
    None
        Displays plot
    """
    plt.figure(figsize=(12, 7))
    
    # Plot histogram
    n, bins, patches = plt.hist(payoffs, bins=50, alpha=0.7, density=True)
    
    # Add kernel density estimate where possible (if payoffs are not all zero)
    if np.std(payoffs) > 0:
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(payoffs)
        x = np.linspace(min(payoffs), max(payoffs), 1000)
        plt.plot(x, kde(x), 'r-', linewidth=2)
    
    # Add statistics
    mean_payoff = np.mean(payoffs)
    std_payoff = np.std(payoffs)
    
    plt.axvline(x=mean_payoff, color='black', linestyle='--', linewidth=1.5,
                label=f'Mean: {mean_payoff:.2f}')
    
    # Add text with statistics
    stats_text = f"Mean: {mean_payoff:.2f}\nStd Dev: {std_payoff:.2f}\nMin: {np.min(payoffs):.2f}\nMax: {np.max(payoffs):.2f}\nOption Price: {price:.2f}"
    plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                ha='left', va='top')
    
    # Add strike information
    if option_type == 'Collar':
        strike_info = f"Call Strike: {strike[0]}, Put Strike: {strike[1]}"
    else:
        strike_info = f"Strike: {strike}"
    
    plt.title(f'{option_type} Option Payoff Distribution\n{strike_info}')
    plt.xlabel('Option Payoff')
    plt.ylabel('Probability Density')
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

def plot_cumulative_dd_paths(simulation_results, degree_day_type='hdd', n_paths_to_show=10):
    """
    Plot cumulative degree day paths.
    
    Parameters:
    -----------
    simulation_results : dict
        Results from simulate_temperature_paths
    degree_day_type : str, optional
        Type of degree days to plot ('hdd' or 'cdd')
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
    
    if degree_day_type.lower() == 'hdd':
        dd_paths = simulation_results['hdd_paths']
        title = 'Heating Degree Days (HDD)'
    elif degree_day_type.lower() == 'cdd':
        dd_paths = simulation_results['cdd_paths']
        title = 'Cooling Degree Days (CDD)'
    else:
        raise ValueError("degree_day_type must be 'hdd' or 'cdd'")
    
    # Calculate cumulative sums
    cum_dd_paths = np.cumsum(dd_paths, axis=1)
    
    # Plot subset of paths
    for i in range(min(n_paths_to_show, cum_dd_paths.shape[0])):
        plt.plot(dates, cum_dd_paths[i], alpha=0.5, linewidth=0.8)
    
    # Calculate and plot mean path
    mean_path = np.mean(cum_dd_paths, axis=0)
    plt.plot(dates, mean_path, 'r-', linewidth=2, label='Mean Path')
    
    # Calculate and plot 5% and 95% quantiles
    q05_path = np.percentile(cum_dd_paths, 5, axis=0)
    q95_path = np.percentile(cum_dd_paths, 95, axis=0)
    
    plt.fill_between(dates, q05_path, q95_path, alpha=0.2, color='red',
                     label='90% Confidence Interval')
    
    plt.title(f'Cumulative {title} Paths')
    plt.xlabel('Date')
    plt.ylabel(f'Cumulative {title}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()