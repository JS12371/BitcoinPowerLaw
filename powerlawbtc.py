import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import linregress, norm
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# Function to plot the power law with percentile lines
def plot_power_law_with_percentile_lines_plotly(data, intercept, slope, std_dev, genesis_date):
    data = data.copy()
    data['Days Since Genesis'] = (data.index - genesis_date).days
    data['Year'] = data['Days Since Genesis'] / 365.25
    days = data['Days Since Genesis'].values
    years = data['Year'].values
    prices = data.iloc[:, 0].values
    
    # Generate future dates up to 2035
    future_dates = pd.date_range(start=data.index[-1], end='2035-12-31')
    future_days = (future_dates - genesis_date).days
    future_years = future_days / 365.25
    
    # Combine current and future data
    all_days = np.concatenate([days, future_days])
    all_years = np.concatenate([years, future_years])
    
    # Calculate the power law (regression line)
    power_law_line = np.exp(intercept + slope * np.log(days))
    
    # Calculate the 90th and 10th percentile lines
    line_90th_percentile = np.exp(intercept + slope * np.log(all_days) + norm.ppf(0.90) * std_dev)
    line_10th_percentile = np.exp(intercept + slope * np.log(all_days) + norm.ppf(0.10) * std_dev)
    
    fig = go.Figure()
    
    # Add scatter plot of BTC prices
    fig.add_trace(go.Scatter(
        x=years,
        y=prices,
        mode='markers',
        marker=dict(color='blue'),
        name='BTC Price'
    ))
    
    # Add power law line
    fig.add_trace(go.Scatter(
        x=all_years,
        y=np.exp(intercept + slope * np.log(all_days)),
        mode='lines',
        name='Power Law',
        line=dict(color='white', dash='dash')
    ))
    
    # Add 90th percentile line
    fig.add_trace(go.Scatter(
        x=all_years,
        y=line_90th_percentile,
        mode='lines',
        name='90th Percentile',
        line=dict(color='red', dash='dash')
    ))
    
    # Add 10th percentile line
    fig.add_trace(go.Scatter(
        x=all_years,
        y=line_10th_percentile,
        mode='lines',
        name='10th Percentile',
        line=dict(color='green', dash='dash')
    ))
    
    fig.update_layout(
        title='BTC Price Power Law with Percentile Lines',
        xaxis_title='Year',
        yaxis_title='BTC Price (USD)',
        yaxis_type="log",
        xaxis_type="log",
        template="plotly_dark"
    )
    
    return fig

# Function to plot power law with 1-year moving average
def plot_power_law_ma_plotly(data, intercept, slope, std_dev, genesis_date):
    data['Days Since Genesis'] = (data.index - genesis_date).days
    data['Year'] = data['Days Since Genesis'] / 365.25
    days = data['Days Since Genesis'].values
    years = data['Year'].values
    ma_prices = data['MA'].values  # Use the MA column
    
    # Generate future dates up to 2035
    future_dates = pd.date_range(start=data.index[-1], end='2035-12-31')
    future_days = (future_dates - genesis_date).days
    future_years = future_days / 365.25
    
    # Combine current and future data
    all_days = np.concatenate([days, future_days])
    all_years = np.concatenate([years, future_years])
    
    # Calculate the power law (regression line)
    power_law_line = np.exp(intercept + slope * np.log(days))
    
    # Calculate the 90th and 10th percentile lines
    line_90th_percentile = np.exp(intercept + slope * np.log(all_days) + norm.ppf(0.90) * std_dev)
    line_10th_percentile = np.exp(intercept + slope * np.log(all_days) + norm.ppf(0.10) * std_dev)
    
    fig = go.Figure()
    
    # Add scatter plot of 1-year moving average prices
    fig.add_trace(go.Scatter(
        x=years,
        y=ma_prices,
        mode='markers',
        marker=dict(color='blue'),
        name='1-Year Moving Average'
    ))
    
    # Add power law line
    fig.add_trace(go.Scatter(
        x=all_years,
        y=np.exp(intercept + slope * np.log(all_days)),
        mode='lines',
        name='Power Law',
        line=dict(color='white', dash='dash')
    ))
    
    # Add 90th percentile line
    fig.add_trace(go.Scatter(
        x=all_years,
        y=line_90th_percentile,
        mode='lines',
        name='90th Percentile',
        line=dict(color='red', dash='dash')
    ))
    
    # Add 10th percentile line
    fig.add_trace(go.Scatter(
        x=all_years,
        y=line_10th_percentile,
        mode='lines',
        name='10th Percentile',
        line=dict(color='green', dash='dash')
    ))
    
    fig.update_layout(
        title='1-Year Moving Average Power Law with Percentile Lines',
        xaxis_title='Year',
        yaxis_title='1-Year Moving Average (USD)',
        yaxis_type="log",
        xaxis_type="log",
        template="plotly_dark"
    )
    
    return fig

# Sinusoidal function for curve fitting
def sinusoidal_function(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

# Function to plot log normalized residuals with sinusoidal fit
def plot_log_normalized_residuals_with_sinusoidal_fit_plotly(data, genesis_date):
    log_normalized_residuals = np.log(data['1yr MA Residuals'].dropna())
    days_since_genesis = data.loc[log_normalized_residuals.index, 'Days Since Genesis'].values
    
    # Fit sinusoidal regression
    amplitude = (np.percentile(log_normalized_residuals, 95) - np.percentile(log_normalized_residuals, 5))
    frequency = 2 * np.pi / (4 * 365)  # Approximately one cycle every 4 years
    vertical_shift = np.median(log_normalized_residuals)
    
    initial_guess = [amplitude, frequency, 0, vertical_shift]
    
    popt, _ = curve_fit(sinusoidal_function, days_since_genesis, log_normalized_residuals, p0=initial_guess)
    sinusoidal_fit = sinusoidal_function(days_since_genesis, *popt)
    
    # Create the figure
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True, vertical_spacing=0.1)
    
    # Plot log normalized residuals
    fig.add_trace(go.Scatter(x=log_normalized_residuals.index, y=log_normalized_residuals, 
                             mode='lines', name='Log of 1yr MA Normalized Residuals', line=dict(color='cyan')))
    fig.add_trace(go.Scatter(x=log_normalized_residuals.index, y=sinusoidal_fit, 
                             mode='lines', name='Sinusoidal Fit', line=dict(color='magenta', dash='dash')))
    
    fig.update_layout(
        title='Log of 1yr MA Normalized Residuals with Sinusoidal Fit',
        xaxis_title='Date',
        yaxis_title='Log of 1yr MA Normalized Residuals',
        template="plotly_dark"
    )
    
    return fig
