import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.dates import DateFormatter, date2num, AutoDateLocator
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tseries_patterns import AmplitudeBasedLabeler
import time
import math
from typing import Dict, List, Optional, Tuple, Union
import hmac

# Authentication
def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password
        else:
            st.session_state["password_correct"] = False

    if "password_correct" not in st.session_state:
        # First run, show input for password
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        return False
    elif not st.session_state["password_correct"]:
        # Password incorrect, show input + error
        st.text_input(
            "Password", type="password", on_change=password_entered, key="password"
        )
        st.error("😕 Password incorrect")
        return False
    else:
        # Password correct
        return True

# Configure page
st.set_page_config(
    page_title="Quantavius Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply dark theme
st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .sidebar .sidebar-content {
        background-color: #262730;
    }
    .Widget>label {
        color: #FAFAFA;
    }
</style>
""", unsafe_allow_html=True)

# Configuration settings
CONFIG = {
    'trading': {
        'timeframe': '1m',
        'exchange': 'binanceusdm',  # Using Binance USD-M Futures
        'market_type': 'future'     # Using futures market
    },
    'visualization': {
        'figure_size': (16, 8),
        'price_color': '#c0c0c0',
        'vwap_above_color': '#3399ff',
        'vwap_below_color': '#ff4d4d',
        'vwma_color': '#90EE90',
        'up_color': '#3399ff',
        'down_color': '#ff4d4d',
        'volume_colors': {
            'high': '#3399ff',
            'medium': '#cccccc',
            'low': '#ff4d4d'
        },
        'base_bubble_size': 35,
        'volume_bins': 50,
        'watermark': {
            'size': 15,
            'color': '#999999',
            'alpha': 0.25
        }
    },
    'analysis': {
        'amplitude_threshold': 20,
        'inactive_period': 10,
        'vwma_period': 20
    }
}

def initialize_exchange():
    """Initialize the cryptocurrency exchange connection."""
    exchange_class = getattr(ccxt, CONFIG['trading']['exchange'])
    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {
            'defaultType': CONFIG['trading']['market_type'],
            'fetchOHLCVWarning': False,
            'adjustForTimeDifference': True,
            'recvWindow': 60000,
            'warnOnFetchOHLCVLimitArgument': False,
        },
        # Add proxy configuration
        'proxies': {
            'http': 'http://proxy.freemyip.com:8080',  # This is a free proxy example
            'https': 'http://proxy.freemyip.com:8080'
        }
    })
    
    # Set custom URLs for API endpoints
    exchange.urls['api'] = {
        'public': 'https://fapi.binance.com/fapi/v1',
        'private': 'https://fapi.binance.com/fapi/v1',
    }
    
    return exchange

def fetch_market_data(exchange, symbol, lookback):
    """Fetch market data using CCXT."""
    try:
        current_time = pd.Timestamp.now(tz='UTC')
        start_time = int((current_time - pd.Timedelta(minutes=lookback+10)).timestamp() * 1000)
        
        # Fetch OHLCV data
        ohlcv = exchange.fetch_ohlcv(
            symbol=symbol,
            timeframe=CONFIG['trading']['timeframe'],
            since=start_time,
            limit=lookback + 10  # Add buffer
        )
        
        if not ohlcv:
            raise Exception("No data received from exchange")
            
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        df['time'] = pd.to_datetime(df['time'], unit='ms')
        df.set_index('time', inplace=True)
        
        # Convert timezone and sort
        df.index = df.index.tz_localize('UTC').tz_convert('America/Toronto')
        df = df.sort_index()
        
        # Take the required number of bars
        df = df.tail(lookback)
        
        if len(df) < lookback * 0.9:  # If we got less than 90% of requested data
            raise Exception(f"Insufficient data: got {len(df)} bars, expected {lookback}")
        
        return df
        
    except Exception as e:
        raise Exception(f"Failed to fetch data: {str(e)}")

def calculate_vwma(df, period):
    """Calculate Volume Weighted Moving Average"""
    df['vwma'] = (df['close'] * df['volume']).rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
    return df

def calculate_metrics(df, exchange, symbol, orderbook_depth):
    """Calculate various market metrics including VWAP and order book imbalance."""
    try:
        # Calculate VWAP - reset at the start of each session
        df['cum_vol'] = df['volume'].cumsum()
        df['cum_vol_price'] = (df['close'] * df['volume']).cumsum()
        df['vwap'] = df['cum_vol_price'] / df['cum_vol']
        
        # Calculate VWMA
        df = calculate_vwma(df, CONFIG['analysis']['vwma_period'])
        
        # Calculate order book metrics using CCXT
        order_book = exchange.fetch_order_book(symbol, orderbook_depth)
        bids_volume = sum(bid[1] for bid in order_book['bids'][:orderbook_depth])
        asks_volume = sum(ask[1] for ask in order_book['asks'][:orderbook_depth])
        
        df['orderbook_imbalance'] = bids_volume / (bids_volume + asks_volume)
        
        # Calculate bubble sizes
        df['bubble_size'] = df['volume'] * df['orderbook_imbalance']
        df['bubble_size'] = df['bubble_size'] / df['bubble_size'].max()
        
        # Reset index to make 'time' a column for AmplitudeBasedLabeler
        df = df.reset_index()
        
        # Create AmplitudeBasedLabeler instance
        labeler = AmplitudeBasedLabeler(
            minamp=CONFIG['analysis']['amplitude_threshold'],
            Tinactive=CONFIG['analysis']['inactive_period']
        )
        
        # Label the data
        labels_df = labeler.label(df)
        
        # Extract momentum values
        if 'label' in labels_df.columns:
            momentum_labels = labels_df['label']
        else:
            momentum_labels = labels_df.iloc[:, 0]
        
        # Add labels to DataFrame and set index back
        df['momentum_label'] = momentum_labels
        df.set_index('time', inplace=True)
        
        # Clean up intermediate columns
        df = df.drop(['cum_vol', 'cum_vol_price'], axis=1)
        
        return df
    except Exception as e:
        print(f"Error in calculate_metrics: {str(e)}")
        raise e

# Visualization functions
def plot_price_action(ax, df, x_values):
    """Plot price action including price line, VWAP, VWMA, and bubbles."""
    # Plot main price line
    ax.plot(x_values, df['close'].values,
            color=CONFIG['visualization']['price_color'],
            linestyle='-', alpha=1.0, linewidth=1.0,  # Increased visibility
            zorder=2, marker='', markersize=0)  # Decreased zorder to be behind bubbles
    
    # Plot VWAP with dynamic coloring
    above_vwap = df['close'] >= df['vwap']
    
    # Plot VWAP segments
    for i in range(1, len(df)):
        if above_vwap.iloc[i]:
            color = CONFIG['visualization']['vwap_above_color']
        else:
            color = CONFIG['visualization']['vwap_below_color']
        
        ax.plot(x_values[i-1:i+1], df['vwap'].iloc[i-1:i+1],
                color=color, linestyle='-', alpha=0.5, linewidth=0.8,
                zorder=1)
    
    # Plot VWMA
    ax.plot(x_values, df['vwma'].values,
            color=CONFIG['visualization']['vwma_color'],
            linestyle='--', alpha=0.5, linewidth=0.8,
            zorder=1)
    
    # Plot bubbles
    up_mask = df['momentum_label'] > 0
    down_mask = df['momentum_label'] <= 0
    base_size = CONFIG['visualization']['base_bubble_size']
    
    ax.scatter(x_values[up_mask], df.loc[up_mask, 'close'],
              c=CONFIG['visualization']['up_color'],
              s=df.loc[up_mask, 'bubble_size'] * base_size,
              alpha=0.8, zorder=3, edgecolors='none')  # Increased alpha and zorder
    ax.scatter(x_values[down_mask], df.loc[down_mask, 'close'],
              c=CONFIG['visualization']['down_color'],
              s=df.loc[down_mask, 'bubble_size'] * base_size,
              alpha=0.8, zorder=3, edgecolors='none')  # Increased alpha and zorder

def plot_volume_profile(ax, df):
    """Plot the volume profile with momentum-based coloring."""
    price_bins = np.linspace(df['close'].min(), df['close'].max(),
                           CONFIG['visualization']['volume_bins'])
    
    # Create separate histograms for different momentum labels
    volume_hist_up = np.histogram(df.loc[df['momentum_label'] > 0, 'close'].values,
                                bins=price_bins,
                                weights=df.loc[df['momentum_label'] > 0, 'volume'].values)[0]
    volume_hist_down = np.histogram(df.loc[df['momentum_label'] < 0, 'close'].values,
                                  bins=price_bins,
                                  weights=df.loc[df['momentum_label'] < 0, 'volume'].values)[0]
    volume_hist_neutral = np.histogram(df.loc[df['momentum_label'] == 0, 'close'].values,
                                     bins=price_bins,
                                     weights=df.loc[df['momentum_label'] == 0, 'volume'].values)[0]
    
    # Calculate total volume for normalization
    total_volume = volume_hist_up + volume_hist_down + volume_hist_neutral
    max_total = total_volume.max() if total_volume.max() > 0 else 1
    
    # Calculate bar width
    max_width = (price_bins[1] - price_bins[0]) * 20
    
    # Plot horizontal volume bars
    for i in range(len(price_bins) - 1):
        price = (price_bins[i] + price_bins[i+1]) / 2
        height = (price_bins[i+1] - price_bins[i]) * 0.95
        
        # Find dominant momentum with priority (up > down > neutral)
        volumes = [
            (volume_hist_up[i], CONFIG['visualization']['volume_colors']['high']),
            (volume_hist_down[i], CONFIG['visualization']['volume_colors']['low']),
            (volume_hist_neutral[i], CONFIG['visualization']['volume_colors']['medium'])
        ]
        # Sort by volume and give priority to up/down over neutral
        sorted_volumes = sorted(volumes, key=lambda x: (x[0], x[1] != CONFIG['visualization']['volume_colors']['medium']), reverse=True)
        max_vol, color = sorted_volumes[0]
        
        # Only plot if there's volume
        if max_vol > 0:
            normalized_vol = -(max_vol / max_total * max_width)
            ax.barh(price, normalized_vol, height=height,
                   color=color, alpha=0.8,  # Slightly reduced alpha for better visibility
                   edgecolor='black', linewidth=0.5)
    
    # Style the volume profile box
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)
        spine.set_visible(True)
    
    # Remove all ticks and labels from volume profile
    ax.set_xticks([])
    ax.set_yticks([])
    
    return max_width

def style_plot(fig, ax_price, ax_vol, df, x_values, max_width):
    """Apply styling to the plot."""
    # Calculate padding
    x_padding = (x_values.max() - x_values.min()) * 0.02
    y_padding = (df['close'].max() - df['close'].min()) * 0.05
    
    # Configure time axis with smaller font
    ax_price.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax_price.xaxis.set_major_locator(AutoDateLocator())
    ax_price.tick_params(axis='x', labelsize=6)
    
    # Remove all spines from price plot
    for spine in ax_price.spines.values():
        spine.set_visible(False)
    
    # Style volume profile spines
    for spine in ax_vol.spines.values():
        spine.set_color('black')
        spine.set_linewidth(0.5)
        spine.set_visible(True)
    
    # Customize ticks
    ax_price.tick_params(axis='both', colors='#666666', length=3, width=0.5)
    ax_vol.tick_params(axis='x', colors='#666666', length=2, width=0.5)
    ax_vol.tick_params(axis='y', colors='#666666', length=0, width=0)
    
    # Set plot limits with padding
    ax_price.set_xlim(x_values.min() - x_padding, x_values.max() + x_padding)
    ax_price.set_ylim(df['close'].min() - y_padding, df['close'].max() + y_padding)
    ax_vol.set_xlim(-max_width * 1.1, 0)  # Added 10% padding
    
    # Final adjustments
    plt.tight_layout()

def create_price_table(df):
    """Create a formatted price data table."""
    # Take the 5 most recent bars
    sampled_df = df.sort_index().tail(5)
    
    # Format the data
    table_data = {
        'Time/Price': [f"{t.strftime('%H:%M')} - ${p:,.2f}"  # Added thousands separator and removed seconds
                      for t, p in zip(sampled_df.index, sampled_df['close'])],
        'Momentum': sampled_df['momentum_label'].map({1: '↑', 0: '•', -1: '↓'}),  # Changed neutral symbol to dot
        'OB Imbalance': (sampled_df['orderbook_imbalance'] * 100).round(1).astype(str) + '%',
        'Volume': sampled_df['volume'].round(2).map('{:,.2f}'.format),  # Added thousands separator
        'VWAP Distance': ((sampled_df['close'] / sampled_df['vwap'] - 1) * 100).round(2).astype(str) + '%'
    }
    
    # Create DataFrame and reverse the order so most recent is at top
    return pd.DataFrame(table_data).iloc[::-1]

# Streamlit app
def main():
    # Add custom CSS for better spacing
    st.markdown("""
        <style>
        .main .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h1 {
            margin-bottom: 2rem;
        }
        .stTable {
            margin-top: 1rem;
        }
        .element-container {
            margin-bottom: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title with custom styling
    st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Quantavius Dashboard</h1>", unsafe_allow_html=True)
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("Enter Ticker Symbol")
        symbol = st.text_input("Symbol (USDT Perpetual)", value="BTC/USDT", key="symbol_input").strip()
        
        st.subheader("Select Lookback Period")
        lookback_options = {
            "1 Day (1440 1m bars)": 1440,
            "4 Hours (240 1m bars)": 240
        }
        lookback = st.selectbox("Lookback", options=list(lookback_options.keys()), key="lookback_select")
        lookback_value = lookback_options[lookback]
        
        st.subheader("Select Order Book Depth")
        orderbook_depth = st.slider("Depth", min_value=10, max_value=100, value=50, key="depth_slider")
        
        st.subheader("Label Settings")
        amplitude_threshold = st.slider("Amplitude Threshold (BPS)", 
                                     min_value=5, 
                                     max_value=50, 
                                     value=20, 
                                     key="amplitude_slider")
        
        inactive_period = st.slider("Inactive Period (minutes)", 
                                  min_value=1, 
                                  max_value=30, 
                                  value=10, 
                                  key="inactive_slider")
        
        # Add note about market type
        st.markdown("""
        ---
        **Note:** This dashboard uses Binance USD-M Perpetual Futures.
        All pairs must be USDT-margined perpetual contracts (e.g., BTC/USDT, ETH/USDT).
        """)
        
        # Update frequency note
        st.subheader("Update Frequency")
        st.text("Data updates every 30 seconds")

    # Main content
    try:
        placeholder = st.empty()
        last_minute = None
        
        while True:
            current_time = pd.Timestamp.now(tz='US/Eastern')
            current_minute = current_time.floor('1min')
            
            with placeholder.container():
                with st.spinner("Fetching data..."):
                    exchange = initialize_exchange()
                    df = fetch_market_data(exchange, symbol, lookback_value)
                    
                    # If we're in a new minute or this is the first run
                    if last_minute is None or current_minute > last_minute:
                        # Update analysis settings from user input
                        CONFIG['analysis']['amplitude_threshold'] = amplitude_threshold
                        CONFIG['analysis']['inactive_period'] = inactive_period
                        
                        df = calculate_metrics(df, exchange, symbol, orderbook_depth)
                        last_minute = current_minute
                    
                    # Create matplotlib figure with increased size
                    fig = plt.figure(figsize=CONFIG['visualization']['figure_size'])
                    gs = gridspec.GridSpec(1, 2, width_ratios=[8, 1])  # Changed ratio from 6:1 to 8:1
                    gs.update(wspace=0.02)  # Added small spacing
                    
                    ax_price = fig.add_subplot(gs[0])
                    ax_vol = fig.add_subplot(gs[1], sharey=ax_price)
                    
                    x_values = date2num(df.index.to_pydatetime())
                    plot_price_action(ax_price, df, x_values)
                    max_width = plot_volume_profile(ax_vol, df)
                    
                    # Update watermark to show ticker
                    ax_price.text(0.5, 0.5, symbol,
                                fontsize=CONFIG['visualization']['watermark']['size'] * 1.5,
                                color=CONFIG['visualization']['watermark']['color'],
                                alpha=CONFIG['visualization']['watermark']['alpha'],
                                ha='center', va='center',
                                transform=ax_price.transAxes,
                                zorder=-1)
                    
                    style_plot(fig, ax_price, ax_vol, df, x_values, max_width)
                    
                    # Add spacing before plot
                    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
                    
                    # Display the plot
                    st.pyplot(fig)
                    
                    # Add spacing after plot
                    st.markdown("<div style='margin: 1.5rem 0;'></div>", unsafe_allow_html=True)
                    
                    # Display price data table
                    price_table = create_price_table(df)
                    st.table(price_table)
                    
                    # Add spacing before timestamp
                    st.markdown("<div style='margin: 1rem 0;'></div>", unsafe_allow_html=True)
                    
                    # Display last data timestamp in EST with custom styling
                    st.markdown(f"<p style='color: #666666; font-size: 0.8em;'>Last data timestamp (EST): {df.index[-1].strftime('%Y-%m-%d %H:%M:%S %Z')}</p>", unsafe_allow_html=True)
                
                # Clear matplotlib figure to prevent memory leak
                plt.close(fig)
                
            # Wait for 30 seconds before the next update
            time.sleep(30)
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    if check_password():
        main() 