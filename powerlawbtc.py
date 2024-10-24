import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import linregress, norm
from scipy.optimize import curve_fit
import plotly.graph_objects as go
import pytz

# Ensure timezone awareness
timezone = pytz.UTC

# Function to fetch data from Yahoo Finance
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.index = data.index.tz_localize(timezone)  # Ensure data has timezone-aware index
    return data

# Function to perform linear regression on log-log data
def perform_log_log_regression(data, genesis_date):
    data = data.copy()
    data['Days Since Genesis'] = (data.index - genesis_date).days
    x = np.log(data['Days Since Genesis'].values)
    y = np.log(data.iloc[:, 0].values)
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Inputs must not be empty.")
    
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    
    return intercept, slope

# Function to calculate standard deviation of residuals
def calculate_std_dev(data, intercept, slope, genesis_date):
    data = data.copy()
    data['Days Since Genesis'] = (data.index - genesis_date).days
    x = np.log(data['Days Since Genesis'].values)
    y = np.log(data.iloc[:, 0].values)
    residuals = y - (intercept + slope * x)
    return np.std(residuals)

# Function to calculate 1-year moving average
def calculate_1yr_ma(data):
    return data.iloc[:, 0].rolling(window=365).mean()

# Function to calculate and normalize residuals
def calculate_and_normalize_residuals(data, intercept, slope, genesis_date):
    data['Days Since Genesis'] = (data.index - genesis_date).days
    x = np.log(data['Days Since Genesis'].values)
    y = np.log(data.iloc[:, 0].values)
    residuals = y - (intercept + slope * x)
    
    normalized_residuals = 1 + 99 * (residuals - np.min(residuals)) / (np.max(residuals) - np.min(residuals))
    data['Normalized Residuals'] = normalized_residuals
    data['1yr MA Residuals'] = data['Normalized Residuals'].rolling(window=360).mean()
    
    return data

# Plot functions omitted for brevity but updated in similar manner with timezone handling
