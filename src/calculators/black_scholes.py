import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
from typing import Literal

@dataclass
class OptionData:
    S: float  # Spot price
    K: float  # Strike price
    T: float  # Time to maturity in years
    r: float  # Risk-free rate
    sigma: float  # Volatility
    option_type: Literal['call', 'put'] = 'call'

class BlackScholes:
    @staticmethod
    def _d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter for Black-Scholes formula"""
        return (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    
    @staticmethod
    def _d2(d1: float, sigma: float, T: float) -> float:
        """Calculate d2 parameter for Black-Scholes formula"""
        return d1 - sigma*np.sqrt(T)
    
    @staticmethod
    def price(option_data: OptionData) -> float:
        """Calculate option price using Black-Scholes formula"""
        d1 = BlackScholes._d1(
            option_data.S, option_data.K, option_data.T, 
            option_data.r, option_data.sigma
        )
        d2 = BlackScholes._d2(d1, option_data.sigma, option_data.T)
        
        if option_data.option_type == 'call':
            return (option_data.S * norm.cdf(d1) - 
                   option_data.K * np.exp(-option_data.r * option_data.T) * norm.cdf(d2))
        else:
            return (option_data.K * np.exp(-option_data.r * option_data.T) * norm.cdf(-d2) - 
                   option_data.S * norm.cdf(-d1))