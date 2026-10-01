"""
Market data fetching and preparation module.

Fetches option chain data from yfinance with comprehensive error handling,
logging, and data quality validation. All functions include complete type hints
for better IDE support and type safety.
"""

import math
import numbers
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Any, List, Dict, Mapping, Tuple, Optional
from src.utils.logger import setup_logger
from src.config.config import MarketDataConfig, ModelConfig

logger = setup_logger(__name__)

# Dividend yields above this are treated as bad data (a unit error, not a real yield)
MAX_PLAUSIBLE_DIVIDEND_YIELD = 0.25


def _as_non_negative_float(value: Any) -> Optional[float]:
    """Return ``value`` as a finite, non-negative float, or None if it is not a real number."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0:
        return None
    return number


def resolve_dividend_yield(info: Mapping[str, Any], spot_price: Optional[float] = None) -> float:
    """
    Return the dividend yield as a decimal (0.013 = 1.3%) from Yahoo ``info`` fields.

    Yahoo's ``dividendYield`` is a percentage, so it is checked against ``dividendRate / spot``
    (or ``trailingAnnualDividendYield``) to pick the right units. Returns 0.0 if unavailable.
    """
    raw = _as_non_negative_float(info.get('dividendYield'))
    trailing = _as_non_negative_float(info.get('trailingAnnualDividendYield'))
    rate = _as_non_negative_float(info.get('dividendRate'))
    spot = _as_non_negative_float(spot_price)

    reference: Optional[float] = None
    if rate is not None and spot:
        reference = rate / spot
    elif trailing is not None:
        reference = trailing

    if raw is None:
        result = reference if reference is not None else 0.0
    elif raw == 0:
        result = 0.0
    elif reference is None:
        result = raw / 100.0  # assume percentage (the safe mistake, see above)
    elif reference == 0:
        result = 0.0
    else:
        as_decimal, as_percent = raw, raw / 100.0
        closer = min((as_decimal, as_percent), key=lambda reading: abs(math.log(reading / reference)))
        result = closer if abs(math.log(closer / reference)) <= math.log(3) else reference

    if result > MAX_PLAUSIBLE_DIVIDEND_YIELD:
        logger.warning(f"Implausible dividend yield {result:.4f} rejected; using 0.0")
        return 0.0

    return float(result)


def filter_quotes(chain: pd.DataFrame, min_open_interest: int) -> pd.DataFrame:
    """
    Keep only contracts with a usable two-sided quote and enough open interest.

    A contract is kept when all of these hold:

    - bid > 0 and ask >= bid (a live, uncrossed quote)
    - mid price >= ``MarketDataConfig.MIN_QUOTE_MID_PRICE``
    - (ask - bid) / mid <= ``MarketDataConfig.MAX_RELATIVE_SPREAD``
    - open interest >= ``min_open_interest`` (missing open interest counts as 0)

    Open interest is used instead of daily volume because volume resets every session,
    which would empty the chain before the market opens.

    Args:
        chain: One side (calls or puts) of a Yahoo option chain
        min_open_interest: Minimum open interest

    Returns:
        The rows that pass every filter
    """
    bid = chain['bid']
    ask = chain['ask']
    mid = (bid + ask) / 2
    if 'openInterest' in chain:
        open_interest = chain['openInterest'].fillna(0)
    else:
        open_interest = pd.Series(0, index=chain.index)

    keep = (
        (bid > 0)
        & (ask >= bid)
        & (mid >= MarketDataConfig.MIN_QUOTE_MID_PRICE)
        & ((ask - bid) / mid <= MarketDataConfig.MAX_RELATIVE_SPREAD)
        & (open_interest >= min_open_interest)
    )
    return chain[keep]


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
                      min_strike_pct: float = MarketDataConfig.DEFAULT_MIN_STRIKE_PCT,
                      max_strike_pct: float = MarketDataConfig.DEFAULT_MAX_STRIKE_PCT,
                      min_open_interest: int = MarketDataConfig.DEFAULT_MIN_OPEN_INTEREST,
                      risk_free_rate: float = ModelConfig.DEFAULT_RISK_FREE_RATE) -> pd.DataFrame:
        """
        Fetch and prepare option data for IV calculation.
        
        Args:
            min_strike_pct: Minimum strike as % of spot (default from config)
            max_strike_pct: Maximum strike as % of spot (default from config)
            min_open_interest: Minimum open interest filter (default from config)
            risk_free_rate: Risk-free rate in decimal form (default from config)
        
        Returns:
            DataFrame with prepared option data
            
        Raises:
            ValueError: If input parameters are invalid or no data found
            ConnectionError: If unable to connect to data provider
        """
        logger.info(f"Fetching option data for {self.symbol}")
        logger.info(f"Parameters - Strike range: {min_strike_pct}%-{max_strike_pct}%, "
                   f"Min open interest: {min_open_interest}, Risk-free rate: {risk_free_rate:.4f}")
        
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
                min_open_interest=min_open_interest
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
            List of expiration dates (excluding dates within MIN_DAYS_TO_EXPIRY)
            
        Raises:
            ValueError: If no valid expiration dates found
            KeyError: If options data structure is unexpected
        """
        try:
            today = pd.Timestamp.now().normalize()
            expirations = self.ticker.options
            
            if not expirations:
                raise ValueError(f'No option expiration dates available for {self.symbol}')
            
            # Filter for dates more than MIN_DAYS_TO_EXPIRY days out
            min_days = MarketDataConfig.MIN_DAYS_TO_EXPIRY
            exp_dates = [
                pd.Timestamp(exp) for exp in expirations 
                if pd.Timestamp(exp) > today + timedelta(days=min_days)
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
                            min_open_interest: int) -> List[Dict]:
        """
        Fetch option chains for all expiration dates.
        
        Args:
            exp_dates: List of expiration dates to fetch
            spot_price: Current spot price for filtering
            min_strike_pct: Minimum strike percentage
            max_strike_pct: Maximum strike percentage
            min_open_interest: Minimum open interest threshold
            
        Returns:
            List of option data dictionaries
        """
        option_data = []
        failed_dates = []
        
        for exp_date in exp_dates:
            try:
                opt_chain = self.ticker.option_chain(exp_date.strftime('%Y-%m-%d'))
                calls = opt_chain.calls
                
                # Keep only usable quotes with enough open interest
                calls = filter_quotes(calls, min_open_interest)
                
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
                            'open_interest': row['openInterest'] if 'openInterest' in row else 0,
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
        options_df['q'] = self.get_dividend_yield(spot_price)
        options_df['moneyness'] = options_df['strike'] / spot_price
        
        return options_df
    
    def _log_data_quality_metrics(self, df: pd.DataFrame) -> None:
        """
        Log data quality metrics for monitoring and debugging.
        
        Args:
            df: Prepared options DataFrame
            
        Returns:
            None
        """
        if df.empty:
            logger.warning("Empty DataFrame - no metrics to report")
            return
        
        logger.info(f"Time to expiry range: {df['T'].min():.2f} - {df['T'].max():.2f} years")
        logger.info(f"Strike range: ${df['strike'].min():.2f} - ${df['strike'].max():.2f}")
        logger.info(f"Moneyness range: {df['moneyness'].min():.2f} - {df['moneyness'].max():.2f}")
    
    def get_dividend_yield(self, spot_price: Optional[float] = None) -> float:
        """Return the dividend yield as a decimal, e.g. 0.013 for 1.3% (0.0 if unavailable)."""
        try:
            info = self.ticker.info
            if not isinstance(info, dict):
                logger.warning("Ticker info unavailable; assuming no dividend yield")
                return 0.0
            return resolve_dividend_yield(info, spot_price)
        except Exception as e:
            logger.warning(f"Could not retrieve dividend yield: {str(e)}")
            return 0.0