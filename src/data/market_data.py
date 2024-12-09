import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict

class OptionDataFetcher:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        
    def prepare_for_iv(self, 
                      min_strike_pct: float = 80.0,
                      max_strike_pct: float = 120.0,
                      min_volume: int = 10) -> pd.DataFrame:  # Added min_volume parameter
        """
        Fetch and prepare option data for IV calculation
        """
        today = pd.Timestamp.now().normalize()
        
        try:
            # Get spot price
            spot_history = self.ticker.history(period='5d')
            if spot_history.empty:
                raise ValueError(f'Failed to retrieve spot price data for {self.symbol}')
            spot_price = spot_history['Close'].iloc[-1]
            
            # Get expiration dates
            expirations = self.ticker.options
            exp_dates = [pd.Timestamp(exp) for exp in expirations 
                        if pd.Timestamp(exp) > today + timedelta(days=7)]
            
            if not exp_dates:
                raise ValueError(f'No available option expiration dates for {self.symbol}')
            
            option_data = []
            
            # Fetch option chains
            for exp_date in exp_dates:
                try:
                    opt_chain = self.ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
                    calls = opt_chain.calls
                    
                    # Filter for valid prices and volume
                    calls = calls[(calls['bid'] > 0) & 
                                (calls['ask'] > 0) & 
                                (calls['volume'].fillna(0) >= min_volume)]  # Added volume filter
                    
                    for _, row in calls.iterrows():
                        strike = row['strike']
                        
                        # Filter based on strike price range
                        if (strike >= spot_price * (min_strike_pct / 100) and 
                            strike <= spot_price * (max_strike_pct / 100)):
                            
                            option_data.append({
                                'expiration': exp_date,
                                'strike': strike,
                                'price': (row['bid'] + row['ask']) / 2,  # midpoint price
                                'type': 'call',
                                'volume': row['volume'] if 'volume' in row else 0
                            })
                            
                except Exception as e:
                    continue
            
            if not option_data:
                raise ValueError('No valid option data available after filtering')
            
            # Create DataFrame
            options_df = pd.DataFrame(option_data)
            
            # Calculate time to expiration
            options_df['T'] = (options_df['expiration'] - today).dt.days / 365
            options_df['days_to_expiry'] = (options_df['expiration'] - today).dt.days
            
            # Add market data
            options_df['S'] = spot_price
            options_df['r'] = 0.05  # Default risk-free rate
            options_df['q'] = self.get_dividend_yield()
            options_df['moneyness'] = options_df['strike'] / spot_price
            
            return options_df
            
        except Exception as e:
            raise ValueError(f'Error preparing data: {str(e)}')
    
    def get_dividend_yield(self) -> float:
        """Get dividend yield from ticker info"""
        try:
            return self.ticker.info.get('dividendYield', 0.0)
        except:
            return 0.0