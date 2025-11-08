import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from typing import Optional

def bs_call_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
    """Calculate Black-Scholes call option price"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def bs_put_price(S: float, K: float, T: float, r: float, sigma: float, q: float = 0) -> float:
    """Calculate Black-Scholes put option price"""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
    return put_price

class IVCalculator:
    def __init__(self):
        pass
    
    def calculate_iv(self, 
                    S: float, 
                    K: float, 
                    T: float, 
                    r: float, 
                    market_price: float,
                    q: float = 0,
                    option_type: str = 'call') -> Optional[float]:
        """
        Calculate implied volatility using Brent's method
        
        Args:
            S: Spot price
            K: Strike price
            T: Time to expiration in years
            r: Risk-free rate
            market_price: Market price of the option
            q: Dividend yield
            option_type: 'call' or 'put'
            
        Returns:
            Implied volatility or None if calculation fails
        """
        if T <= 0 or market_price <= 0:
            return None
        
        # Validate option_type
        option_type = option_type.lower()
        if option_type not in ['call', 'put']:
            return None

        def objective_function(sigma):
            if option_type == 'call':
                return bs_call_price(S, K, T, r, sigma, q) - market_price
            else:
                return bs_put_price(S, K, T, r, sigma, q) - market_price

        try:
            implied_vol = brentq(objective_function, 1e-6, 5)
            return implied_vol
        except (ValueError, RuntimeError):
            return None