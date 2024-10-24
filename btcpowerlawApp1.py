import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz
from scipy.stats import linregress, norm
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
# Import functions from powerlawbtc.py
from powerlawbtc import (
    fetch_data, perform_log_log_regression, calculate_std_dev,
    plot_power_law_with_percentile_lines_plotly, calculate_1yr_ma,
    plot_power_law_ma_plotly, sinusoidal_function,
    plot_log_normalized_residuals_with_sinusoidal_fit_plotly,
    calculate_and_normalize_residuals
)

st.set_page_config(page_title="Bitcoin Power Law Analysis", layout="wide")
st.title("Bitcoin Power Law Analysis")

# Ensure timezone-aware datetime usage
timezone = pytz.UTC

# Fetch and prepare data
@st.cache_data
def load_data():
    btc_ticker = 'BTC-USD'
    end_date = datetime.now(timezone).strftime('%Y-%m-%d')
    
    # Load all available historical data
    btc_historical_data = pd.read_csv('btc_data_reversed.csv', index_col='Date', parse_dates=True)
    btc_historical_data.index = btc_historical_data.index.tz_localize(timezone)  # Make sure index is timezone aware
    
    # Fetch recent data (last 6 months) to ensure up-to-date information
    start_date = (datetime.now(timezone) - timedelta(days=180)).strftime('%Y-%m-%d')
    btc_recent_data = fetch_data(btc_ticker, start_date, end_date)
    btc_recent_data.index = btc_recent_data.index.tz_localize(timezone)
    
    # Concatenate and sort data
    btc_data = pd.concat([btc_historical_data, btc_recent_data[['Close']][~btc_recent_data.index.isin(btc_historical_data.index)]])
    btc_data = btc_data.sort_index()
    
    return btc_data

btc_data = load_data()

genesis_date = datetime(2009, 1, 3, tzinfo=timezone)  # Ensure genesis_date is timezone-aware

# Calculate 1-year moving average
btc_data_ma = calculate_1yr_ma(btc_data)

# Perform power law regression
intercept, slope = perform_log_log_regression(btc_data, genesis_date)
std_dev = calculate_std_dev(btc_data, intercept, slope, genesis_date)

# Sidebar options for plot selection
st.sidebar.header("Select Plot")
plot_option = st.sidebar.selectbox(
    "Choose a plot to display",
    ("Power Law", "Power Law with Moving Average", "Log Normalized Residuals")
)

# Add scale selection for power law plots
if plot_option in ["Power Law", "Power Law with Moving Average"]:
    scale_option = st.sidebar.radio(
        "Choose scale",
        ("Log-Log", "Log-Y Linear-X")
    )

# Modify the plotting functions to accept a scale parameter
def plot_power_law_with_scale(btc_data, intercept, slope, std_dev, genesis_date, scale):
    fig = plot_power_law_with_percentile_lines_plotly(btc_data, intercept, slope, std_dev, genesis_date)
    if scale == "Log-Y Linear-X":
        fig.update_xaxes(type="linear")
    return fig

def plot_power_law_ma_with_scale(btc_data_ma, intercept_ma, slope_ma, std_dev_ma, genesis_date_ma, scale):
    fig = plot_power_law_ma_plotly(btc_data_ma, intercept_ma, slope_ma, std_dev_ma, genesis_date_ma)
    if scale == "Log-Y Linear-X":
        fig.update_xaxes(type="linear")
    return fig

# Update the plotting section
if plot_option == "Power Law":
    st.subheader("Bitcoin Price Power Law")
    fig = plot_power_law_with_scale(btc_data, intercept, slope, std_dev, genesis_date, scale_option)
    st.plotly_chart(fig, use_container_width=True)
elif plot_option == "Power Law with Moving Average":
    st.subheader("Bitcoin 1-Year Moving Average Power Law")
    fig = plot_power_law_ma_with_scale(btc_data_ma, intercept_ma, slope_ma, std_dev_ma, genesis_date_ma, scale_option)
    st.plotly_chart(fig, use_container_width=True)
elif plot_option == "Log Normalized Residuals":
    st.subheader("Log of 1yr MA Normalized Residuals and 1yr MA of Pct Change")
    fig = plot_log_normalized_residuals_with_sinusoidal_fit_plotly(btc_data, genesis_date)
    st.plotly_chart(fig, use_container_width=True)

st.sidebar.info("This app analyzes Bitcoin's price history using power law models and various statistical techniques.")
