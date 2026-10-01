"""
Implied Volatility calculation module using Black-Scholes model.

Implements Brent's method for numerical solving of implied volatility
with comprehensive input validation and error handling.
"""

import numpy as np
from scipy.optimize import brentq
from typing import Optional, Tuple
from src.calculators.black_scholes import BlackScholes, OptionData
from src.utils.logger import setup_logger
from src.config.config import IVCalculationConfig, ModelConfig

logger = setup_logger(__name__)


def bs_call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Black-Scholes call price (thin wrapper around ``BlackScholes.price``).

    Kept as a standalone function for convenience and backwards compatibility.
    At expiry (T <= 0) the intrinsic value is returned.

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
    return BlackScholes.price(OptionData(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type='call'))


def bs_put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0.0) -> float:
    """
    Black-Scholes put price (thin wrapper around ``BlackScholes.price``).

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
    return BlackScholes.price(OptionData(S=S, K=K, T=T, r=r, sigma=sigma, q=q, option_type='put'))


def no_arbitrage_bounds(S: float, K: float, T: float, r: float, q: float,
                        option_type: str) -> Tuple[float, float]:
    """
    Model-free bounds that a European option price must respect.

    Uses *discounted* values (the correct bounds), not raw intrinsic value:

    - Call: max(S*e^(-qT) - K*e^(-rT), 0) <= C <= S*e^(-qT)
    - Put:  max(K*e^(-rT) - S*e^(-qT), 0) <= P <= K*e^(-rT)

    Args:
        S: Spot price
        K: Strike price
        T: Time to expiration in years
        r: Risk-free rate
        q: Dividend yield
        option_type: 'call' or 'put'

    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    discounted_spot = S * np.exp(-q * T)
    discounted_strike = K * np.exp(-r * T)

    if option_type == 'call':
        return max(discounted_spot - discounted_strike, 0.0), discounted_spot
    return max(discounted_strike - discounted_spot, 0.0), discounted_strike


class IVCalculator:
    """
    Implied Volatility calculator using Brent's root-finding method.
    
    Features:
    - Input validation for all parameters
    - No-arbitrage bound checking (discounted lower and upper bounds)
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
        
        # Reject prices outside the no-arbitrage bounds (no implied vol can exist)
        lower_bound, upper_bound = no_arbitrage_bounds(S, K, T, r, q, option_type)
        tolerance = IVCalculationConfig.INTRINSIC_VALUE_TOLERANCE

        if market_price < lower_bound * tolerance:
            logger.debug(f"Market price ({market_price:.4f}) below no-arbitrage lower bound ({lower_bound:.4f})")
            self.failed_count += 1
            return None

        if market_price >= upper_bound:
            logger.debug(f"Market price ({market_price:.4f}) at or above no-arbitrage upper bound ({upper_bound:.4f})")
            self.failed_count += 1
            return None

        def objective_function(sigma):
            """Objective function: model_price - market_price = 0"""
            if option_type == 'call':
                return bs_call_price(S, K, T, r, sigma, q) - market_price
            return bs_put_price(S, K, T, r, sigma, q) - market_price

        try:
            # Use Brent's method to find the root with config bounds
            implied_vol = brentq(
                objective_function, 
                IVCalculationConfig.IV_MIN_BOUND,
                IVCalculationConfig.IV_MAX_BOUND,
                xtol=IVCalculationConfig.IV_CONVERGENCE_TOLERANCE
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