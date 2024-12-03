import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

class OptionDataFetcher:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
    
    def get_current_price(self) -> float:
        return self.ticker.history(period='1d')['Close'].iloc[-1]
    
    def get_risk_free_rate(self) -> float:
        # In practice, you'd fetch this from a reliable source
        # Using a constant for now
        return 0.05
    
    def get_option_chain(self) -> pd.DataFrame:
        """
        Fetch and format option chain data
        Returns DataFrame with columns: strike, expiration, bid, ask, type
        """
        # Get all available expiration dates
        expirations = self.ticker.options
        
        all_options = []
        for exp in expirations:
            opt = self.ticker.option_chain(exp)
            
            # Process calls
            calls = opt.calls
            calls['type'] = 'call'
            
            # Process puts
            puts = opt.puts
            puts['type'] = 'put'
            
            # Combine and format
            for chain in [calls, puts]:
                chain['expiration'] = exp
                all_options.append(chain[['strike', 'expiration', 'bid', 'ask', 'type']])
        
        return pd.concat(all_options, ignore_index=True)
    
    def prepare_for_iv(self) -> pd.DataFrame:
        """
        Prepare data for IV calculation, filtering out expired options
        """
        S = self.get_current_price()
        r = self.get_risk_free_rate()
        
        # Get option chain
        options_df = self.get_option_chain()
        
        # Calculate time to expiration in years
        today = datetime.now()
        options_df['T'] = pd.to_datetime(options_df['expiration']).apply(
            lambda x: (x - today).days / 365.0
        )
        
        # Add days to expiry column
        options_df['days_to_expiry'] = options_df['T'] * 365
        
        # Filter out expired options and options with zero or negative time to expiration
        options_df = options_df[options_df['T'] > 0.001]
        
        # Calculate mid price
        options_df['price'] = (options_df['bid'] + options_df['ask']) / 2
        
        # Filter out options with zero or very low prices
        options_df = options_df[options_df['price'] > 0.01]
        
        # Add spot price and risk-free rate
        options_df['S'] = S
        options_df['r'] = r
        
        return options_df