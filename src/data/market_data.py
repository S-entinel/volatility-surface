"""
Market data fetching and preparation module.

Fetches option chain data from yfinance with comprehensive error handling,
logging, and data quality validation.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class OptionDataFetcher:
    """
    Fetches and prepares option market data for implied volatility calculations.
    
    Features:
    - Comprehensive error handling with specific exception types
    - Detailed logging at each step
    - Data quality metrics tracking
    - Modular method structure for maintainability
    """
    
    def __init__(self, symbol: str):
        """
        Initialize fetcher for a specific ticker symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'SPY', 'AAPL')
        """
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        logger.info(f"Initializing OptionDataFetcher for {symbol}")
        
    def prepare_for_iv(self, 
                      min_strike_pct: float = 80.0,
                      max_strike_pct: float = 120.0,
                      min_volume: int = 10,
                      risk_free_rate: float = 0.045) -> pd.DataFrame:
        """
        Fetch and prepare option data for IV calculation.
        
        Args:
            min_strike_pct: Minimum strike as % of spot (default: 80%)
            max_strike_pct: Maximum strike as % of spot (default: 120%)
            min_volume: Minimum option volume filter (default: 10)
            risk_free_rate: Risk-free rate in decimal form (default: 0.045 = 4.5%)
        
        Returns:
            DataFrame with prepared option data
            
        Raises:
            ValueError: If input parameters are invalid or no data found
            ConnectionError: If unable to connect to data provider
        """
        logger.info(f"Fetching option data for {self.symbol}")
        logger.info(f"Parameters - Strike range: {min_strike_pct}%-{max_strike_pct}%, "
                   f"Min volume: {min_volume}, Risk-free rate: {risk_free_rate:.4f}")
        
        try:
            # Step 1: Fetch spot price
            spot_price = self._fetch_spot_price()
            logger.info(f"Spot price retrieved: ${spot_price:.2f}")
            
            # Step 2: Fetch expiration dates
            exp_dates = self._fetch_expiration_dates()
            logger.info(f"Found {len(exp_dates)} expiration dates")
            
            # Step 3: Fetch option chains
            option_data = self._fetch_option_chains(
                exp_dates=exp_dates,
                spot_price=spot_price,
                min_strike_pct=min_strike_pct,
                max_strike_pct=max_strike_pct,
                min_volume=min_volume
            )
            
            if not option_data:
                raise ValueError('No valid option data available after filtering')
            
            logger.info(f"Successfully fetched {len(option_data)} option contracts")
            
            # Step 4: Prepare final DataFrame
            options_df = self._prepare_dataframe(
                option_data=option_data,
                spot_price=spot_price,
                risk_free_rate=risk_free_rate
            )
            
            # Log data quality metrics
            self._log_data_quality_metrics(options_df)
            
            logger.info(f"Data preparation complete. Final dataset: {len(options_df)} rows")
            return options_df
            
        except (ValueError, ConnectionError, KeyError, AttributeError) as e:
            logger.error(f"Unexpected error in prepare_for_iv: {str(e)}")
            raise
    
    def _fetch_spot_price(self) -> float:
        """
        Fetch current spot price for the ticker.
        
        Returns:
            Current spot price
            
        Raises:
            ValueError: If unable to retrieve spot price
            ConnectionError: If network/API error occurs
        """
        try:
            spot_history = self.ticker.history(period='5d')
            
            if spot_history.empty:
                raise ValueError(f'Failed to retrieve spot price data for {self.symbol}')
            
            spot_price = spot_history['Close'].iloc[-1]
            
            if spot_price <= 0 or pd.isna(spot_price):
                raise ValueError(f'Invalid spot price retrieved: {spot_price}')
            
            return float(spot_price)
            
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Network or API error fetching spot price: {str(e)}")
            raise ConnectionError(f"Failed to connect to market data provider")
    
    def _fetch_expiration_dates(self) -> List[pd.Timestamp]:
        """
        Fetch valid option expiration dates.
        
        Returns:
            List of expiration dates (excluding dates within 7 days)
            
        Raises:
            ValueError: If no valid expiration dates found
            KeyError: If options data structure is unexpected
        """
        try:
            today = pd.Timestamp.now().normalize()
            expirations = self.ticker.options
            
            if not expirations:
                raise ValueError(f'No option expiration dates available for {self.symbol}')
            
            # Filter for dates more than 7 days out
            exp_dates = [
                pd.Timestamp(exp) for exp in expirations 
                if pd.Timestamp(exp) > today + timedelta(days=7)
            ]
            
            if not exp_dates:
                raise ValueError(f'No valid option expiration dates for {self.symbol}')
            
            return exp_dates
            
        except (ValueError, AttributeError) as e:
            raise ValueError(f"Error fetching expiration dates: {str(e)}")
    
    def _fetch_option_chains(self,
                            exp_dates: List[pd.Timestamp],
                            spot_price: float,
                            min_strike_pct: float,
                            max_strike_pct: float,
                            min_volume: int) -> List[Dict]:
        """
        Fetch option chains for all expiration dates.
        
        Args:
            exp_dates: List of expiration dates to fetch
            spot_price: Current spot price for filtering
            min_strike_pct: Minimum strike percentage
            max_strike_pct: Maximum strike percentage
            min_volume: Minimum volume threshold
            
        Returns:
            List of option data dictionaries
        """
        option_data = []
        failed_dates = []
        
        for exp_date in exp_dates:
            try:
                opt_chain = self.ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
                calls = opt_chain.calls
                
                # Filter for valid prices and volume
                calls = calls[
                    (calls['bid'] > 0) & 
                    (calls['ask'] > 0) & 
                    (calls['volume'].fillna(0) >= min_volume)
                ]
                
                # Process each option
                for _, row in calls.iterrows():
                    strike = row['strike']
                    
                    # Filter based on strike price range
                    if (strike >= spot_price * (min_strike_pct / 100) and 
                        strike <= spot_price * (max_strike_pct / 100)):
                        
                        option_data.append({
                            'expiration': exp_date,
                            'strike': strike,
                            'price': (row['bid'] + row['ask']) / 2,  # midpoint
                            'type': 'call',
                            'volume': row['volume'] if 'volume' in row else 0,
                            'days_to_expiry': (exp_date - pd.Timestamp.now().normalize()).days
                        })
                        
            except Exception as e:
                failed_dates.append(exp_date)
                logger.warning(f"Failed to fetch option chain for {exp_date.date()}: {str(e)}")
                continue
        
        if failed_dates:
            logger.warning(f"Failed to fetch {len(failed_dates)} out of {len(exp_dates)} expiration dates")
        
        return option_data
    
    def _prepare_dataframe(self,
                          option_data: List[Dict],
                          spot_price: float,
                          risk_free_rate: float) -> pd.DataFrame:
        """
        Prepare final DataFrame with calculated fields.
        
        Args:
            option_data: List of option dictionaries
            spot_price: Current spot price
            risk_free_rate: Risk-free rate in decimal form
            
        Returns:
            Prepared DataFrame with all necessary fields
        """
        today = pd.Timestamp.now().normalize()
        
        # Create DataFrame
        options_df = pd.DataFrame(option_data)
        
        # Calculate time to expiration in years
        options_df['T'] = (options_df['expiration'] - today).dt.days / 365
        
        # Add market data
        options_df['S'] = spot_price
        options_df['r'] = risk_free_rate
        options_df['q'] = self.get_dividend_yield()
        options_df['moneyness'] = options_df['strike'] / spot_price
        
        return options_df
    
    def _log_data_quality_metrics(self, df: pd.DataFrame) -> None:
        """
        Log data quality metrics for monitoring and debugging.
        
        Args:
            df: Prepared options DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame - no metrics to report")
            return
        
        logger.info(f"Time to expiry range: {df['T'].min():.2f} - {df['T'].max():.2f} years")
        logger.info(f"Strike range: ${df['strike'].min():.2f} - ${df['strike'].max():.2f}")
        logger.info(f"Moneyness range: {df['moneyness'].min():.2f} - {df['moneyness'].max():.2f}")
    
    def get_dividend_yield(self) -> float:
        """
        Get dividend yield from ticker info.
        
        Returns:
            Dividend yield (0.0 if not available)
        """
        try:
            div_yield = self.ticker.info.get('dividendYield', 0.0)
            return div_yield if div_yield is not None else 0.0
        except Exception as e:
            logger.warning(f"Could not retrieve dividend yield: {str(e)}")
            return 0.0