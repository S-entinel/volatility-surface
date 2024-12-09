import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal

@dataclass
class OptionData:
    S: float  # Spot price
    K: float  # Strike price
    T: float  # Time to maturity in years
    r: float  # Risk-free rate
    sigma: float  # Volatility
    q: float = 0.0  # Dividend yield
    option_type: Literal['call', 'put'] = 'call'

class BlackScholes:
    @staticmethod
    def price(option_data: OptionData) -> float:
        """
        Calculate option price using Black-Scholes formula
        Simplified to match the reference implementation
        """
        d1 = (np.log(option_data.S / option_data.K) + 
              (option_data.r - option_data.q + 0.5 * option_data.sigma ** 2) * option_data.T) / \
             (option_data.sigma * np.sqrt(option_data.T))
        d2 = d1 - option_data.sigma * np.sqrt(option_data.T)
        
        if option_data.option_type == 'call':
            price = (option_data.S * np.exp(-option_data.q * option_data.T) * norm.cdf(d1) - 
                    option_data.K * np.exp(-option_data.r * option_data.T) * norm.cdf(d2))
        else:
            price = (option_data.K * np.exp(-option_data.r * option_data.T) * norm.cdf(-d2) - 
                    option_data.S * np.exp(-option_data.q * option_data.T) * norm.cdf(-d1))
            
        return price