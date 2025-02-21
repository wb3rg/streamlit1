"""
Real-Time Cryptocurrency Market Analysis Visualization
Author: [Your Name]
Description: Advanced visualization tool for cryptocurrency market analysis including price action,
volume profile, and order book dynamics.
"""

import matplotlib
# Configure matplotlib for notebook display
matplotlib.use('Agg')  # Use Agg backend
import matplotlib.pyplot as plt

# Patch matplotlib's internal code to use np.inf instead of np.Inf
import matplotlib.axes._base
def _patched_update_title_position(self, renderer):
    """Patch to replace np.Inf with np.inf"""
    if not hasattr(self, 'title'):
        return

    titles = (self.title, self._left_title, self._right_title)
    if not any(title.get_visible() for title in titles):
        return

    top = float('-inf')
    bottom = float('inf')

    for title in titles:
        if title.get_visible():
            x, y = title.get_position()
            if y > top:
                top = y
            if y < bottom:
                bottom = y

    for title in titles:
        if title.get_visible():
            x, y = title.get_position()
            title.set_position((x, y))

# Apply the patch
matplotlib.axes._base._AxesBase._update_title_position = _patched_update_title_position

# Reset to clean style
plt.style.use('default')

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tseries_patterns import AmplitudeBasedLabeler
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.dates import DateFormatter, date2num, AutoDateLocator
import matplotlib.gridspec as gridspec

# =============================================
# Configuration Settings
# =============================================
CONFIG = {
    'trading': {
        'symbol': 'BTC/USDT',
        'timeframe': '1m',
        'limit': 1440,  # 24 hours of 1-minute data
        'exchange': 'binance',
    },
    'visualization': {
        'figure_size': (15, 8),
        'price_color': '#dddddd',
        'vwap_color': '#cc99ff',
        'up_color': '#3399ff',
        'down_color': '#ff4d4d',
        'volume_colors': {
            'high': '#99ccff',
            'medium': '#cccccc',
            'low': '#ffb3b3'
        },
        'base_bubble_size': 35,
        'volume_bins': 50,
        'watermark': {
            'size': 15,
            'color': '#dddddd',
            'alpha': 0.3
        }
    },
    'analysis': {
        'orderbook_depth': 10,
        'amplitude_threshold': 20,
        'inactive_period': 10
    }
}

# =============================================
# Matplotlib Configuration
# =============================================
def configure_matplotlib():
    """Configure matplotlib settings for optimal visualization."""
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
    except:
        matplotlib.use('TkAgg')
    
    plt.style.use('default')
    plt.rcParams.update({
        'figure.figsize': CONFIG['visualization']['figure_size'],
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.grid': False,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'font.family': 'sans-serif'
    })

# =============================================
# Data Collection
# =============================================
def initialize_exchange():
    """Initialize the cryptocurrency exchange connection."""
    exchange_id = CONFIG['trading']['exchange']
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class()

def fetch_market_data(exchange):
    """Fetch market data from the exchange."""
    end_time = exchange.milliseconds()
    start_time = end_time - (CONFIG['trading']['limit'] * 60 * 1000)
    
    ohlcv = exchange.fetch_ohlcv(
        CONFIG['trading']['symbol'],
        CONFIG['trading']['timeframe'],
        start_time,
        limit=CONFIG['trading']['limit']
    )
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('time', inplace=True)
    return df

# =============================================
# Market Analysis
# =============================================
def calculate_metrics(df, exchange):
    """Calculate various market metrics including VWAP and order book imbalance."""
    # Calculate VWAP
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate order book imbalance
    order_book = exchange.fetch_order_book('BTC/USDT')
    bids_volume = sum(bid[1] for bid in order_book['bids'][:10])  # Top 10 bids
    asks_volume = sum(ask[1] for ask in order_book['asks'][:10])  # Top 10 asks
    df['orderbook_imbalance'] = bids_volume / (bids_volume + asks_volume)
    
    # Calculate bubble sizes
    df['bubble_size'] = df['volume'] * df['orderbook_imbalance']
    df['bubble_size'] = df['bubble_size'] / df['bubble_size'].max()  # Normalize to [0,1]
    
    # Reset index to make 'time' a column for AmplitudeBasedLabeler
    df = df.reset_index()
    
    # Create AmplitudeBasedLabeler instance
    labeler = AmplitudeBasedLabeler(minamp=20, Tinactive=10)
    
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
    return df

# =============================================
# Visualization Components
# =============================================
def setup_plot_layout():
    """Set up the basic plot layout."""
    fig = plt.figure(figsize=CONFIG['visualization']['figure_size'])
    gs = gridspec.GridSpec(1, 2, width_ratios=[6, 1])
    gs.update(wspace=0)
    
    ax_price = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharey=ax_price)
    
    return fig, ax_price, ax_vol

def plot_price_action(ax, df, x_values):
    """Plot price action including price line, VWAP, and bubbles."""
    # Plot main price line
    ax.plot(x_values, df['close'].values,
            color=CONFIG['visualization']['price_color'],
            linestyle='-', alpha=0.8, linewidth=0.5,
            zorder=1, marker='', markersize=0)
    
    # Plot VWAP
    ax.plot(x_values, df['vwap'].values,
            color=CONFIG['visualization']['vwap_color'],
            linestyle='-', alpha=0.5, linewidth=0.5,
            zorder=1, label='VWAP')
    
    # Plot bubbles
    up_mask = df['momentum_label'] > 0
    down_mask = df['momentum_label'] <= 0
    base_size = CONFIG['visualization']['base_bubble_size']
    
    ax.scatter(x_values[up_mask], df.loc[up_mask, 'close'],
              c=CONFIG['visualization']['up_color'],
              s=df.loc[up_mask, 'bubble_size'] * base_size,
              alpha=0.7, zorder=2, edgecolors='none')
    ax.scatter(x_values[down_mask], df.loc[down_mask, 'close'],
              c=CONFIG['visualization']['down_color'],
              s=df.loc[down_mask, 'bubble_size'] * base_size,
              alpha=0.7, zorder=2, edgecolors='none')

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
        
        # Prioritize up/down momentum over neutral
        if volume_hist_up[i] > volume_hist_down[i] and volume_hist_up[i] > 0:
            color = CONFIG['visualization']['volume_colors']['high']
            vol = volume_hist_up[i]
        elif volume_hist_down[i] > volume_hist_up[i] and volume_hist_down[i] > 0:
            color = CONFIG['visualization']['volume_colors']['low']
            vol = volume_hist_down[i]
        elif volume_hist_neutral[i] > 0:
            color = CONFIG['visualization']['volume_colors']['medium']
            vol = volume_hist_neutral[i]
        else:
            continue
            
        normalized_vol = -(vol / max_total * max_width)
        ax.barh(price, normalized_vol, height=height,
               color=color, alpha=1.0,
               edgecolor='black', linewidth=0.5)
    
    # Remove all ticks and labels from volume profile
    ax.set_xticks([])
    ax.set_yticks([])
    
    return max_width

def style_plot(fig, ax_price, ax_vol, df, x_values, max_width):
    """Apply styling to the plot."""
    # Calculate padding
    x_padding = (x_values.max() - x_values.min()) * 0.02
    y_padding = (df['close'].max() - df['close'].min()) * 0.05
    
    # Style price plot
    ax_price.set_xlabel('Time', fontsize=10, color='#666666')
    ax_price.legend(loc='upper left', fontsize=8)
    
    # Add perfectly centered watermark with balanced visibility
    ax_price.text(0.5, 0.5, CONFIG['trading']['symbol'],
                 fontsize=CONFIG['visualization']['watermark']['size'] * 1.5,  # Slightly larger
                 color='#999999',  # Subtle grey
                 alpha=0.25,  # More subtle opacity
                 ha='center', va='center',
                 transform=ax_price.transAxes,
                 zorder=-1)  # Ensure it's behind everything
    
    # Configure time axis
    ax_price.xaxis.set_major_formatter(DateFormatter('%H:%M'))
    ax_price.xaxis.set_major_locator(AutoDateLocator())
    
    # Remove all spines from price plot
    for spine in ax_price.spines.values():
        spine.set_visible(False)
    
    # Style volume profile spines
    ax_vol.spines['left'].set_color('black')
    ax_vol.spines['right'].set_color('black')
    ax_vol.spines['top'].set_color('black')
    ax_vol.spines['bottom'].set_color('black')
    ax_vol.spines['left'].set_linewidth(0.5)
    ax_vol.spines['right'].set_linewidth(0.5)
    ax_vol.spines['top'].set_linewidth(0.5)
    ax_vol.spines['bottom'].set_linewidth(0.5)
    
    # Customize ticks
    ax_price.tick_params(axis='both', colors='#666666', length=3, width=0.5)
    ax_vol.tick_params(axis='both', colors='#666666', length=0, width=0)
    
    # Set plot limits
    ax_price.set_xlim(x_values.min() - x_padding, x_values.max() + x_padding)
    ax_price.set_ylim(df['close'].min() - y_padding, df['close'].max() + y_padding)
    ax_vol.set_xlim(-max_width, 0)
    
    # Final adjustments
    plt.tight_layout()

# =============================================
# Main Execution
# =============================================
def create_visualization(df):
    """Create the complete visualization."""
    fig, ax_price, ax_vol = setup_plot_layout()
    x_values = date2num(df.index.to_pydatetime())
    
    plot_price_action(ax_price, df, x_values)
    max_width = plot_volume_profile(ax_vol, df)
    style_plot(fig, ax_price, ax_vol, df, x_values, max_width)
    
    plt.show()
    return fig

def main():
    print("Fetching BTC/USDT data...")
    exchange = initialize_exchange()
    df = fetch_market_data(exchange)
    
    print("Calculating metrics and labels...")
    df = calculate_metrics(df, exchange)
    
    print("Creating visualization...")
    create_visualization(df)

if __name__ == "__main__":
    configure_matplotlib()
    main() 