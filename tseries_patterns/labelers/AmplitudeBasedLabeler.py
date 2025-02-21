import numpy as np
import pandas as pd

class AmplitudeBasedLabeler:
    def __init__(self, minamp=20, Tinactive=10):
        """
        Initialize the AmplitudeBasedLabeler.
        
        Args:
            minamp (float): Minimum amplitude threshold in basis points
            Tinactive (int): Inactive period in minutes
        """
        self.minamp = minamp / 10000.0  # Convert basis points to decimal
        self.Tinactive = Tinactive
        
    def label(self, df):
        """
        Label price movements based on amplitude and inactive periods.
        
        Args:
            df (pd.DataFrame): DataFrame with 'time' and 'close' columns
            
        Returns:
            pd.DataFrame: DataFrame with labels column added
        """
        # Initialize arrays
        n = len(df)
        labels = np.zeros(n)
        prices = df['close'].values
        times = pd.to_datetime(df['time']).values
        
        # Calculate returns
        returns = np.diff(np.log(prices))
        returns = np.insert(returns, 0, 0)  # Add 0 at the beginning for alignment
        
        # Calculate cumulative returns for amplitude measurement
        cum_returns = np.zeros(n)
        last_signal_idx = 0
        last_signal_time = times[0]
        
        for i in range(1, n):
            # Check if we're in an inactive period
            time_diff = (times[i] - last_signal_time).astype('timedelta64[m]').astype(float)
            
            if time_diff >= self.Tinactive:
                # Reset cumulative returns after inactive period
                cum_returns[i] = returns[i]
                last_signal_idx = i - 1
                last_signal_time = times[i]
            else:
                # Accumulate returns
                cum_returns[i] = cum_returns[i-1] + returns[i]
            
            # Check amplitude threshold
            amplitude = abs(cum_returns[i])
            if amplitude >= self.minamp:
                # Assign label based on direction
                labels[last_signal_idx:i+1] = 1 if cum_returns[i] > 0 else -1
                # Reset cumulative returns and update last signal
                cum_returns[i] = 0
                last_signal_idx = i
                last_signal_time = times[i]
        
        return pd.DataFrame({'label': labels}) 