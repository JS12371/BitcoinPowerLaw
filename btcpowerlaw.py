import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Constants for the regression
FAIR_SLOPE_BTC = 5.703334904144785
FAIR_INTERCEPT_BTC = -38.11129934443797
STD_DEV_BTC = 0.9389097246887748

GENESIS_DATE = datetime(2009, 1, 3)  # Arbitrary genesis date for BTC
END_DATE = datetime(2034, 12, 31)

# Function to fetch data from Yahoo Finance
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to update data with new Yahoo Finance data
def update_data(existing_data, ticker):
    last_date = existing_data.index[-1]
    today = datetime.now()
    if last_date < today - timedelta(days=1):
        new_data = fetch_data(ticker, last_date + timedelta(days=1), today)
        updated_data = pd.concat([existing_data, new_data])
        return updated_data
    return existing_data

# Load the BTC data from the CSV file
btc_data = pd.read_csv('btc_data_reversed.csv', index_col='Date', parse_dates=True)
btc_data = update_data(btc_data, 'BTC-USD')

# Function to plot results with heatmap and dates on x-axis using Plotly
def plot_results(data, fair_slope, fair_intercept, std_dev):
    data['Days Since Genesis'] = (data.index - GENESIS_DATE).days
    dates = data.index
    prices = data['Close'].values
    days = data['Days Since Genesis'].values

    # Generate future dates up to 2035
    future_dates = pd.date_range(start=dates[-1], end='2035-12-31')
    future_days = (future_dates - GENESIS_DATE).days

    # Combine current and future data
    all_days = np.concatenate([days, future_days])
    all_dates = dates.append(future_dates)

    # Calculate the regression lines
    regression_line = np.exp(fair_intercept) * all_days ** fair_slope
    support_line = np.exp(fair_intercept - std_dev) * all_days ** fair_slope
    resistance_line = np.exp(fair_intercept + std_dev) * all_days ** fair_slope

    # Calculate percent deviations and percentiles
    percent_deviation = (prices - np.exp(fair_intercept) * days ** fair_slope) / (np.exp(fair_intercept) * days ** fair_slope) * 100
    percentiles = pd.qcut(percent_deviation, 100, labels=False)
    percentile_now = percentiles[-1]

    fig = go.Figure()

    # Scatter plot with percentiles as color
    fig.add_trace(go.Scatter(x=dates, y=prices, mode='markers', 
                             marker=dict(color=percentiles, colorscale='turbo', showscale=True),
                             name='Price'))

    # Add regression lines
    fig.add_trace(go.Scatter(x=all_dates, y=regression_line, mode='lines', line=dict(color='white', dash='dash'), name='Fair Line'))
    fig.add_trace(go.Scatter(x=all_dates, y=support_line, mode='lines', line=dict(color='green', dash='dash'), name='Support Line'))
    fig.add_trace(go.Scatter(x=all_dates, y=resistance_line, mode='lines', line=dict(color='red', dash='dash'), name='Resistance Line'))

    fig.update_layout(
        title='BTC-USD Price Data with Power Law Regression (Extended to 2035)',
        xaxis_title='Date',
        yaxis_title='Close Price',
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='gray'),
        yaxis=dict(showgrid=True, gridcolor='gray', type='log'),
        legend=dict(
            x=0,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(0,0,0,0)'
        )
    )

    st.plotly_chart(fig)

    return percentile_now

# Function to plot log-log graph with regression and bands using Plotly
def plot_log_log(data, fair_slope, fair_intercept, std_dev):
    data['Days Since Genesis'] = (data.index - GENESIS_DATE).days
    days = data['Days Since Genesis'].values
    prices = data['Close'].values

    percent_deviation = (prices - np.exp(fair_intercept) * days ** fair_slope) / (np.exp(fair_intercept) * days ** fair_slope) * 100
    percentiles = pd.qcut(percent_deviation, 100, labels=False)
    percentile_now = percentiles[-1]

    # Generate future days up to 2035
    future_days = (pd.date_range(start=data.index[-1], end='2035-12-31') - GENESIS_DATE).days

    # Combine current and future data
    all_days = np.concatenate([days, future_days])
    all_prices = np.concatenate([prices, [np.nan] * len(future_days)])

    # Calculate the regression lines
    regression_line = np.exp(fair_intercept) * all_days ** fair_slope
    support_line = np.exp(fair_intercept - std_dev) * all_days ** fair_slope
    resistance_line = np.exp(fair_intercept + std_dev) * all_days ** fair_slope

    fig = go.Figure()

    # Scatter plot with percentiles as color
    fig.add_trace(go.Scatter(x=days, y=prices, mode='markers', 
                             marker=dict(color=percentiles, colorscale='turbo', showscale=True),
                             name='Price'))

    # Add regression lines
    fig.add_trace(go.Scatter(x=all_days, y=regression_line, mode='lines', line=dict(color='white', dash='dash'), name='Fair Line'))
    fig.add_trace(go.Scatter(x=all_days, y=support_line, mode='lines', line=dict(color='green', dash='dash'), name='Support Line'))
    fig.add_trace(go.Scatter(x=all_days, y=resistance_line, mode='lines', line=dict(color='red', dash='dash'), name='Resistance Line'))

    # Add vertical lines and labels for each new year starting from 2023
    for year in range(2011, 2036):
        year_days = (datetime(year, 1, 1) - GENESIS_DATE).days
        fig.add_trace(go.Scatter(x=[year_days, year_days], y=[prices.min(), prices.max()*2000], mode='lines', 
                                 line=dict(color='mediumseagreen', dash='dot'), 
                                 hovertext=f'{year}', 
                                 hoverinfo='text',
                                 showlegend=False))

    fig.update_layout(
        title='Log-Log Plot of Days Since Genesis vs. Close Price with Regression Lines (Extended to 2035)',
        xaxis_title='Days Since Genesis',
        yaxis_title='Close Price',
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(showgrid=True, gridcolor='gray', type='log'),
        yaxis=dict(showgrid=True, gridcolor='gray', type='log'),
        legend=dict(
            x=0,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(0,0,0,0)'
        )
    )

    st.plotly_chart(fig)

    return percentile_now

# Function to calculate predicted values for a given date
def calculate_predicted_values(target_date, fair_slope, fair_intercept, std_dev):
    days_since_genesis = (pd.to_datetime(target_date) - GENESIS_DATE).days
    fair_value = np.exp(fair_intercept) * days_since_genesis ** fair_slope
    support_value = np.exp(fair_intercept - std_dev) * days_since_genesis ** fair_slope
    resistance_value = np.exp(fair_intercept + std_dev) * days_since_genesis ** fair_slope
    return fair_value, support_value, resistance_value

# Streamlit App
st.set_page_config(page_title="BTC Power Law Analysis", layout="wide", initial_sidebar_state="expanded")
st.title("BTC Power Law Analysis")

# Sidebar settings
st.sidebar.header("Settings")
graph_type = st.sidebar.radio("Select Graph Type", ('Log-Log', 'Log-Linear'))

if not btc_data.empty:
    # Log-transform the data
    btc_data['Days Since Genesis'] = (btc_data.index - GENESIS_DATE).days
    log_days = np.log(btc_data['Days Since Genesis'].values).reshape(-1, 1)
    log_prices = np.log(btc_data['Close'].values).reshape(-1, 1)

    # Perform linear regression on the log-transformed data
    model = LinearRegression()
    model.fit(log_days, log_prices)
    log_predicted_prices = model.predict(log_days)

    # Calculate R² value
    r_squared = r2_score(log_prices, log_predicted_prices)

    if graph_type == 'Log-Linear':
        percentile_now = plot_results(btc_data, FAIR_SLOPE_BTC, FAIR_INTERCEPT_BTC, STD_DEV_BTC)
    elif graph_type == 'Log-Log':
        percentile_now = plot_log_log(btc_data, FAIR_SLOPE_BTC, FAIR_INTERCEPT_BTC, STD_DEV_BTC)

    st.write("### Regression Stats")
    data_dict = {
        "R² Value": [f"{r_squared:.4f}"],
        "Current Residual Risk Value": [f"{percentile_now:.0f}%"],
        "Date of Regression": [datetime.now().strftime("%Y-%m-%d")]
    }

    # Create a dataframe for display
    df = pd.DataFrame(data_dict)

    # Display the dataframe
    st.table(df)

    # Input date for prediction
    st.sidebar.header("Prediction Settings")
    target_date = st.sidebar.date_input("Select a date for prediction", value=datetime.now())
    
    if st.sidebar.button("Calculate Predicted Values"):
        fair_value, support_value, resistance_value = calculate_predicted_values(target_date, FAIR_SLOPE_BTC, FAIR_INTERCEPT_BTC, STD_DEV_BTC)
        
        prediction_data_dict = {
            "Date": [target_date],
            "Fair Value": [f"{fair_value:.8f}"],
            "Support Value": [f"{support_value:.8f}"],
            "Resistance Value": [f"{resistance_value:.8f}"]
        }
        
        prediction_df = pd.DataFrame(prediction_data_dict)
        
        st.write("### Prediction Data")
        st.table(prediction_df)
else:
    st.write("No data found for the given ticker and date range.")

