"""
Implied Volatility calculation module using Black-Scholes model.

Implements Brent's method for numerical solving of implied volatility
with comprehensive input validation and error handling.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Optional
from src.utils.logger import setup_logger
from src.config.config import IVCalculationConfig, ModelConfig

logger = setup_logger(__name__)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
    """
    Calculate Black-Scholes call option price.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield (default: 0)
        
    Returns:
        Call option price
    """
    if T <= 0:
        return max(S - K, 0)  # Intrinsic value at expiration
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
    """
    Calculate Black-Scholes put option price.
    
    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield (default: 0)
        
    Returns:
        Put option price
    """
    if T <= 0:
        return max(K - S, 0)  # Intrinsic value at expiration
    
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price


class IVCalculator:
    """
    Implied Volatility calculator using Brent's root-finding method.
    
    Features:
    - Input validation for all parameters
    - Intrinsic value checking to detect arbitrage violations
    - Statistics tracking for monitoring calculation success rates
    - Detailed error logging for debugging
    """
    
    def __init__(self):
        """Initialize calculator with statistics tracking."""
        self.calculation_count = 0
        self.failed_count = 0
        logger.info("IVCalculator initialized")
    
    def calculate_iv(self, 
                    S: float, 
                    K: float, 
                    T: float, 
                    r: float, 
                    market_price: float,
                    q: float = 0,
                    option_type: str = 'call') -> Optional[float]:
        """
        Calculate implied volatility using Brent's method.
        
        Args:
            S: Spot price (must be > 0)
            K: Strike price (must be > 0)
            T: Time to expiration in years (must be > 0)
            r: Risk-free rate (typically 0 to 1)
            market_price: Market price of the option (must be > 0)
            q: Dividend yield (default: 0, typically 0 to 1)
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility or None if calculation fails
            
        Notes:
            - Returns None for invalid inputs or convergence failures
            - Tracks success/failure statistics internally
        """
        self.calculation_count += 1
        
        # Validate inputs
        validation_error = self._validate_inputs(S, K, T, r, market_price, q, option_type)
        if validation_error:
            logger.debug(f"Input validation failed: {validation_error}")
            self.failed_count += 1
            return None
        
        # Normalize option type
        option_type = option_type.lower()
        
        # Check intrinsic value to catch arbitrage violations
        intrinsic_value = max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        tolerance = IVCalculationConfig.INTRINSIC_VALUE_TOLERANCE
        
        if market_price < intrinsic_value * tolerance:
            logger.debug(f"Market price ({market_price:.4f}) below intrinsic value ({intrinsic_value:.4f})")
            self.failed_count += 1
            return None

        def objective_function(sigma):
            """Objective function: model_price - market_price = 0"""
            if option_type == 'call':
                return bs_call_price(S, K, T, r, sigma, q) - market_price
            else:
                return bs_put_price(S, K, T, r, sigma, q) - market_price

        try:
            # Use Brent's method to find the root with config bounds
            implied_vol = brentq(
                objective_function, 
                IVCalculationConfig.IV_MIN_BOUND, 
                IVCalculationConfig.IV_MAX_BOUND
            )
            return implied_vol
            
        except ValueError as e:
            # Bracketing error - function doesn't cross zero in interval
            logger.debug(f"Brent's method bracketing error: {str(e)} "
                        f"(S={S:.2f}, K={K:.2f}, T={T:.4f}, price={market_price:.4f})")
            self.failed_count += 1
            return None
            
        except RuntimeError as e:
            # Convergence failure
            logger.debug(f"Brent's method convergence error: {str(e)} "
                        f"(S={S:.2f}, K={K:.2f}, T={T:.4f}, price={market_price:.4f})")
            self.failed_count += 1
            return None
    
    def _validate_inputs(self,
                        S: float,
                        K: float,
                        T: float,
                        r: float,
                        market_price: float,
                        q: float,
                        option_type: str) -> Optional[str]:
        """
        Validate all input parameters using config thresholds.
        
        Args:
            S, K, T, r, market_price, q, option_type: Same as calculate_iv
            
        Returns:
            Error message if validation fails, None if all inputs valid
        """
        # Check for positive values using config thresholds
        if S < IVCalculationConfig.MIN_SPOT_PRICE:
            return f"Spot price must be >= {IVCalculationConfig.MIN_SPOT_PRICE} (got {S})"
        if K < IVCalculationConfig.MIN_STRIKE_PRICE:
            return f"Strike price must be >= {IVCalculationConfig.MIN_STRIKE_PRICE} (got {K})"
        if T < IVCalculationConfig.MIN_TIME_TO_EXPIRY:
            return f"Time to expiration must be >= {IVCalculationConfig.MIN_TIME_TO_EXPIRY} (got {T})"
        if market_price < IVCalculationConfig.MIN_MARKET_PRICE:
            return f"Market price must be >= {IVCalculationConfig.MIN_MARKET_PRICE} (got {market_price})"
        
        # Check for reasonable ranges using model config
        if r < ModelConfig.MIN_RISK_FREE_RATE or r > ModelConfig.MAX_RISK_FREE_RATE:
            return f"Risk-free rate should be between {ModelConfig.MIN_RISK_FREE_RATE} and {ModelConfig.MAX_RISK_FREE_RATE} (got {r})"
        if q < ModelConfig.MIN_DIVIDEND_YIELD or q > ModelConfig.MAX_DIVIDEND_YIELD:
            return f"Dividend yield should be between {ModelConfig.MIN_DIVIDEND_YIELD} and {ModelConfig.MAX_DIVIDEND_YIELD} (got {q})"
        
        # Validate option type
        if option_type.lower() not in ['call', 'put']:
            return f"Option type must be 'call' or 'put' (got '{option_type}')"
        
        return None
    
    def get_statistics(self) -> dict:
        """
        Get calculation statistics.
        
        Returns:
            Dictionary with calculation metrics:
            - total: Total calculations attempted
            - successful: Number of successful calculations
            - failed: Number of failed calculations
            - success_rate: Percentage of successful calculations
        """
        success_rate = 0.0
        if self.calculation_count > 0:
            success_rate = ((self.calculation_count - self.failed_count) / 
                          self.calculation_count * 100)
        
        return {
            'total': self.calculation_count,
            'successful': self.calculation_count - self.failed_count,
            'failed': self.failed_count,
            'success_rate': success_rate
        }
    
    def reset_statistics(self) -> None:
        """Reset calculation statistics to zero."""
        self.calculation_count = 0
        self.failed_count = 0
        logger.info("Statistics reset")