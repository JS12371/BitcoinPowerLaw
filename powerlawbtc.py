import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import linregress, norm
from scipy.optimize import curve_fit
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

# Function to fetch data from Yahoo Finance
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Function to perform linear regression on log-log data
def perform_log_log_regression(data, genesis_date):
    data = data.copy()  # Create a copy to avoid SettingWithCopyWarning
    data['Days Since Genesis'] = (data.index - genesis_date).days
    x = np.log(data['Days Since Genesis'].values)
    y = np.log(data.iloc[:, 0].values)  # Use the first (and only) column, whatever it's named

    if len(x) == 0 or len(y) == 0:
        raise ValueError("Inputs must not be empty.")

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    
    # Calculate R-squared and adjusted R-squared
    r_squared = r_value ** 2
    n = len(x)
    p = 1  # number of predictors (excluding intercept)
    adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)
    
    # Calculate standard error of the estimate
    residuals = y - (intercept + slope * x)
    std_error = np.sqrt(np.sum(residuals**2) / (n - 2))
    
    # Calculate t-statistic for slope
    t_stat = slope / std_err

    # Print regression statistics
    print("\nRegression Statistics:")
    print(f"Intercept: {intercept:.4f}")
    print(f"Slope: {slope:.4f}")
    print(f"R-squared: {r_squared:.4f}")
    print(f"Adjusted R-squared: {adjusted_r_squared:.4f}")
    print(f"Standard Error: {std_error:.4f}")
    print(f"t-statistic (slope): {t_stat:.4f}")
    print(f"p-value (slope): {p_value:.4e}")

    return intercept, slope

# Function to calculate the standard deviation of residuals
def calculate_std_dev(data, intercept, slope, genesis_date):
    data = data.copy()  # Create a copy to avoid SettingWithCopyWarning
    data['Days Since Genesis'] = (data.index - genesis_date).days
    x = np.log(data['Days Since Genesis'].values)
    y = np.log(data.iloc[:, 0].values)  # Use the first (and only) column, whatever it's named
    residuals = y - (intercept + slope * x)
    std_dev = np.std(residuals)
    return std_dev

# New Plotly version of plot_power_law_with_percentile_lines
def plot_power_law_with_percentile_lines_plotly(data, intercept, slope, std_dev, genesis_date):
    data = data.copy()  # Create a copy to avoid SettingWithCopyWarning
    data['Days Since Genesis'] = (data.index - genesis_date).days
    data['Year'] = data['Days Since Genesis'] / 365.25
    days = data['Days Since Genesis'].values
    years = data['Year'].values
    prices = data.iloc[:, 0].values  # Use the first (and only) column, whatever it's named

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

    # Calculate percent deviation
    percent_deviation = (prices - power_law_line) / power_law_line * 100

    # Calculate percentile ranks of percent deviations
    percentiles = pd.qcut(percent_deviation, 100, labels=False)

    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=years,
        y=prices,
        mode='markers',
        marker=dict(color=percentiles, colorscale='Turbo', showscale=True, colorbar=dict(title='Percentile')),
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

    # Add 90th and 10th percentile lines
    fig.add_trace(go.Scatter(
        x=all_years,
        y=line_90th_percentile,
        mode='lines',
        name='90th Percentile',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=all_years,
        y=line_10th_percentile,
        mode='lines',
        name='10th Percentile',
        line=dict(color='green', dash='dash')
    ))

    fig.update_layout(
        title='BTC Price vs. Year on Log-Log Scale with Percentile Lines',
        xaxis_title='Year',
        yaxis_title='Close Price (USD)',
        yaxis_type="log",
        xaxis_type="log",
        legend_title="Legend",
        template="plotly_dark"
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=np.arange(0, 27, 1),
        ticktext=[f'{int(year + genesis_date.year)}' for year in np.arange(0, 27, 1)]
    )

    return fig

# New function to calculate 1-year moving average
def calculate_1yr_ma(data):
    return data.iloc[:, 0].rolling(window=365).mean()

# New Plotly version of plot_power_law_ma
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

    # Calculate percent deviation
    percent_deviation = (ma_prices - power_law_line) / power_law_line * 100

    # Calculate percentile ranks of percent deviations
    percentiles = pd.qcut(percent_deviation, 100, labels=False)

    fig = go.Figure()

    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=years,
        y=ma_prices,
        mode='markers',
        marker=dict(color=percentiles, colorscale='Turbo', showscale=True, colorbar=dict(title='Percentile')),
        name='BTC 1-Year MA'
    ))

    # Add power law line
    fig.add_trace(go.Scatter(
        x=all_years,
        y=np.exp(intercept + slope * np.log(all_days)),
        mode='lines',
        name='Power Law',
        line=dict(color='white', dash='dash')
    ))

    # Add 90th and 10th percentile lines
    fig.add_trace(go.Scatter(
        x=all_years,
        y=line_90th_percentile,
        mode='lines',
        name='90th Percentile',
        line=dict(color='red', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=all_years,
        y=line_10th_percentile,
        mode='lines',
        name='10th Percentile',
        line=dict(color='green', dash='dash')
    ))

    fig.update_layout(
        title='BTC 1-Year MA vs. Year on Log-Log Scale with Percentile Lines',
        xaxis_title='Year',
        yaxis_title='1-Year MA Price (USD)',
        yaxis_type="log",
        xaxis_type="log",
        legend_title="Legend",
        template="plotly_dark"
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=np.arange(0, 27, 1),
        ticktext=[f'{int(year + genesis_date.year)}' for year in np.arange(0, 27, 1)]
    )

    return fig

# Sinusoidal function for fitting
def sinusoidal_function(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

# New Plotly version of plot_log_normalized_residuals_with_sinusoidal_fit
def plot_log_normalized_residuals_with_sinusoidal_fit_plotly(data, genesis_date):
    log_normalized_residuals = np.log(data['1yr MA Residuals'].dropna())
    days_since_genesis = data.loc[log_normalized_residuals.index, 'Days Since Genesis'].values
    
    # Calculate initial parameters for log normalized residuals sinusoidal fit
    amplitude = (np.percentile(log_normalized_residuals, 95) - np.percentile(log_normalized_residuals, 5))
    frequency = 2 * np.pi / (4 * 365)  # Approximately one cycle every 4 years
    vertical_shift = np.median(log_normalized_residuals)
    
    initial_guess = [amplitude, frequency, 0, vertical_shift]
    
    # Fit sinusoidal regression for log normalized residuals
    popt, _ = curve_fit(sinusoidal_function, days_since_genesis, log_normalized_residuals, p0=initial_guess)
    sinusoidal_fit = sinusoidal_function(days_since_genesis, *popt)
    
    # Calculate R^2 value for log normalized residuals sinusoidal fit
    r_squared = 1 - (np.sum((log_normalized_residuals - sinusoidal_fit) ** 2) / np.sum((log_normalized_residuals - np.mean(log_normalized_residuals)) ** 2))
    
    # Print the sinusoidal function formula
    a, b, c, d = popt
    print(f"Sinusoidal Function (Log Normalized Residuals): y = {a:.4f} * sin({b:.6f} * x + {c:.4f}) + {d:.4f}")
    
    # Calculate median, 95th percentile, and 5th percentile
    median = np.median(log_normalized_residuals)
    percentile_95 = np.percentile(log_normalized_residuals, 95)
    percentile_5 = np.percentile(log_normalized_residuals, 5)

    # Calculate percentage change of the log of the 1yr MA of normalized residuals
    pct_change_log_residuals = log_normalized_residuals.pct_change().dropna()

    # Calculate 30-day moving average of the percentage change
    ma_30day_pct_change = pct_change_log_residuals.rolling(window=360).mean().dropna()
    ma_days_since_genesis = data.loc[ma_30day_pct_change.index, 'Days Since Genesis'].values
    
    # Initial guess for sinusoidal fit for 30-day MA of percentage change
    amplitude_ma = (np.percentile(ma_30day_pct_change, 95) - np.percentile(ma_30day_pct_change, 5))
    vertical_shift_ma = np.median(ma_30day_pct_change)
    initial_guess_ma = [amplitude_ma, frequency, 0, vertical_shift_ma]

    # Fit sinusoidal regression for 30-day MA of percentage change
    popt_ma, _ = curve_fit(sinusoidal_function, ma_days_since_genesis, ma_30day_pct_change, p0=initial_guess_ma)
    sinusoidal_fit_ma = sinusoidal_function(ma_days_since_genesis, *popt_ma)
    
    # Calculate R^2 value for 30-day MA sinusoidal fit
    r_squared_ma = 1 - (np.sum((ma_30day_pct_change - sinusoidal_fit_ma) ** 2) / np.sum((ma_30day_pct_change - np.mean(ma_30day_pct_change)) ** 2))
    
    # Print the sinusoidal function formula for 30-day MA
    a_ma, b_ma, c_ma, d_ma = popt_ma
    print(f"Sinusoidal Function (30-Day MA Pct Change): y = {a_ma:.4f} * sin({b_ma:.6f} * x + {c_ma:.4f}) + {d_ma:.4f}")

    # Create subplots
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)

    # Add traces for log normalized residuals
    fig.add_trace(go.Scatter(x=log_normalized_residuals.index, y=log_normalized_residuals, 
                             mode='lines', name='Log of 1yr MA Normalized Residuals', line=dict(color='cyan')), row=1, col=1)
    fig.add_trace(go.Scatter(x=log_normalized_residuals.index, y=sinusoidal_fit, 
                             mode='lines', name=f'Sinusoidal Fit (R^2 = {r_squared:.4f})', line=dict(color='magenta', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[log_normalized_residuals.index[0], log_normalized_residuals.index[-1]], y=[median, median], 
                             mode='lines', name='Median', line=dict(color='yellow', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[log_normalized_residuals.index[0], log_normalized_residuals.index[-1]], y=[percentile_95, percentile_95], 
                             mode='lines', name='95th Percentile', line=dict(color='red', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=[log_normalized_residuals.index[0], log_normalized_residuals.index[-1]], y=[percentile_5, percentile_5], 
                             mode='lines', name='5th Percentile', line=dict(color='green', dash='dash')), row=1, col=1)

    # Add traces for 30-day MA of percentage change
    fig.add_trace(go.Scatter(x=ma_30day_pct_change.index, y=ma_30day_pct_change, 
                             mode='lines', name='30-Day MA of Pct Change', line=dict(color='orange')), row=2, col=1)
    fig.add_trace(go.Scatter(x=ma_30day_pct_change.index, y=sinusoidal_fit_ma, 
                             mode='lines', name=f'Sinusoidal Fit (R^2 = {r_squared_ma:.4f})', line=dict(color='magenta', dash='dash')), row=2, col=1)
    fig.add_trace(go.Scatter(x=[ma_30day_pct_change.index[0], ma_30day_pct_change.index[-1]], y=[0, 0], 
                             mode='lines', name='Zero Line', line=dict(color='white', dash='dash')), row=2, col=1)

    fig.update_layout(
        title='Log of 1yr MA Normalized Residuals and 30-Day MA of Pct Change with Sinusoidal Fits',
        xaxis_title='Date',
        yaxis_title='Log of 1yr MA Normalized Residuals',
        yaxis2_title='30-Day MA of Pct Change',
        legend_title="Legend",
        template="plotly_dark"
    )

    fig.update_xaxes(
        tickmode='array',
        tickvals=np.arange(0, 27, 1),
        ticktext=[f'{int(year + genesis_date.year)}' for year in np.arange(0, 27, 1)]
    )

    return fig

# Function to calculate and normalize residuals
def calculate_and_normalize_residuals(data, intercept, slope, genesis_date):
    data['Days Since Genesis'] = (data.index - genesis_date).days
    x = np.log(data['Days Since Genesis'].values)
    y = np.log(data.iloc[:, 0].values)  # Use the first (and only) column, whatever it's named
    residuals = y - (intercept + slope * x)
    
    # Normalize residuals between 1 and 100
    normalized_residuals = 1 + 99 * (residuals - np.min(residuals)) / (np.max(residuals) - np.min(residuals))
    
    # Calculate 1yr moving average of residuals
    data['Normalized Residuals'] = normalized_residuals
    data['1yr MA Residuals'] = data['Normalized Residuals'].rolling(window=360).mean()
    
    return data

# Main function
if __name__ == "__main__":
    btc_ticker = 'BTC-USD'
    start_date = '2022-06-01'
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Fetch recent data from Yahoo Finance
    btc_recent_data = fetch_data(btc_ticker, start_date, end_date)

    # Load historical data from the provided CSV file
    btc_historical_data = pd.read_csv('btc_data_reversed.csv', index_col='Date', parse_dates=True)
    
    # Select only necessary columns and rename them to match the Yahoo Finance data format
    btc_historical_data = btc_historical_data[['Close']]
    
    # Concatenate historical and recent data
    btc_data = pd.concat([btc_historical_data, btc_recent_data[['Close']][~btc_recent_data.index.isin(btc_historical_data.index)]])
    btc_data = btc_data.sort_index()  # Ensure data is sorted by date
    
    # Define the genesis date
    genesis_date = datetime(2009, 1, 3)
    
    print("\nPower Law Regression Statistics:")
    intercept, slope = perform_log_log_regression(btc_data, genesis_date)
    
    # Calculate the standard deviation of residuals
    std_dev = calculate_std_dev(btc_data, intercept, slope, genesis_date)
    
    # Plot the power law on a log-log scale with 90th and 10th percentile lines
    fig_power_law = plot_power_law_with_percentile_lines_plotly(btc_data, intercept, slope, std_dev, genesis_date)
    fig_power_law.show()

    # Calculate 1-year moving average
    btc_data['MA'] = calculate_1yr_ma(btc_data)

    # Remove the first year of data (NaN values in MA)
    btc_data_ma = btc_data.dropna()

    # Adjust the genesis date for MA data
    genesis_date_ma = genesis_date + timedelta(days=365)

    # Perform regression on MA data
    print("\nMoving Average Power Law Regression Statistics:")
    intercept_ma, slope_ma = perform_log_log_regression(btc_data_ma['MA'].to_frame(), genesis_date_ma)

    # Calculate the standard deviation of residuals for MA data
    std_dev_ma = calculate_std_dev(btc_data_ma['MA'].to_frame(), intercept_ma, slope_ma, genesis_date_ma)

    # Plot the power law on a log-log scale with 90th and 10th percentile lines for MA data
    fig_power_law_ma = plot_power_law_ma_plotly(btc_data_ma, intercept_ma, slope_ma, std_dev_ma, genesis_date_ma)
    fig_power_law_ma.show()

    # Calculate and normalize residuals
    btc_data = calculate_and_normalize_residuals(btc_data, intercept, slope, genesis_date)

    # Plot log of 1yr MA normalized residuals with sinusoidal fit and percentage change
    fig_residuals = plot_log_normalized_residuals_with_sinusoidal_fit_plotly(btc_data, genesis_date)
    fig_residuals.show()
