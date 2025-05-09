import openmeteo_requests
import requests_cache
import pandas as pd
import numpy as np
from retry_requests import retry

def setup_openmeteo_client():
    """Set up the Open-Meteo API client with caching and retry functionality."""
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)

def get_temperature_data(latitude, longitude, start_date, end_date):
    """
    Retrieve temperature data from Open-Meteo API.
    
    Parameters:
    -----------
    latitude : float
        Latitude of the location
    longitude : float
        Longitude of the location
    start_date : str
        Start date in format 'YYYY-MM-DD'
    end_date : str
        End date in format 'YYYY-MM-DD'
        
    Returns:
    --------
    pandas.DataFrame
        Dataframe with daily mean temperature data
    """
    # Setup the API client
    openmeteo = setup_openmeteo_client()
    
    # Define parameters
    url_params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m",
        "daily": "temperature_2m_mean"
    }
    
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
    
    # Make the API request
    responses = openmeteo.weather_api(url, params=url_params)
    response = responses[0]
    
    # Process daily data
    daily = response.Daily()
    daily_temps = daily.Variables(0).ValuesAsNumpy()
    
    # Create a datetime index
    daily_dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        periods=len(daily_temps),
        freq="D"
    )
    
    # Create a pandas DataFrame
    daily_dataframe = pd.DataFrame({
        "temperature_2m_mean": daily_temps
    }, index=daily_dates)
    
    return daily_dataframe

def clean_temperature_data(df):
    """
    Clean temperature data by handling missing values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe with temperature data
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe with no missing values
    """
    # Handle missing data using interpolation
    cleaned_df = df.interpolate(method='linear')
    
    # Fill any remaining NaNs at the edges with forward/backward fill
    cleaned_df = cleaned_df.fillna(method='ffill').fillna(method='bfill')
    
    return cleaned_df

def get_amsterdam_temperature_data(start_date="2020-08-10", end_date="2024-08-23"):
    """
    Retrieve and clean temperature data for Amsterdam.
    
    Parameters:
    -----------
    start_date : str, optional
        Start date in format 'YYYY-MM-DD'
    end_date : str, optional
        End date in format 'YYYY-MM-DD'
        
    Returns:
    --------
    pandas.DataFrame
        Cleaned dataframe with daily mean temperature data for Amsterdam
    """
    # Amsterdam coordinates
    latitude = 52.37
    longitude = 4.89
    
    # Get temperature data
    temp_data = get_temperature_data(latitude, longitude, start_date, end_date)
    
    # Clean the data
    cleaned_data = clean_temperature_data(temp_data)
    
    return cleaned_data