"""
Black-Scholes option pricing model with complete type hints.

Provides option pricing calculations with dividend adjustments.
All functions include comprehensive type annotations for type safety.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Literal

# Type alias for option types
OptionType = Literal['call', 'put']


@dataclass
class OptionData:
    """
    Container for option pricing parameters.
    
    Attributes:
        S: Spot price of the underlying asset
        K: Strike price of the option
        T: Time to maturity in years
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        q: Dividend yield (annualized, default: 0.0)
        option_type: Type of option ('call' or 'put', default: 'call')
    """
    S: float
    K: float
    T: float
    r: float
    sigma: float
    q: float = 0.0
    option_type: OptionType = 'call'


class BlackScholes:
    """
    Black-Scholes option pricing model.
    
    Implements the classic Black-Scholes formula for European options
    with dividend yield adjustments.
    """
    
    @staticmethod
    def price(option_data: OptionData) -> float:
        """
        Calculate option price using Black-Scholes formula.
        
        Args:
            option_data: OptionData instance containing all pricing parameters
            
        Returns:
            Option price as a float
            
        Raises:
            ValueError: If option_type is not 'call' or 'put'
            
        Example:
            >>> data = OptionData(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
            >>> price = BlackScholes.price(data)
            >>> print(f"Option price: ${price:.2f}")
        """
        # Calculate d1 and d2
        d1: float = (
            np.log(option_data.S / option_data.K) + 
            (option_data.r - option_data.q + 0.5 * option_data.sigma ** 2) * option_data.T
        ) / (option_data.sigma * np.sqrt(option_data.T))
        
        d2: float = d1 - option_data.sigma * np.sqrt(option_data.T)
        
        # Calculate price based on option type
        if option_data.option_type == 'call':
            price: float = (
                option_data.S * np.exp(-option_data.q * option_data.T) * norm.cdf(d1) - 
                option_data.K * np.exp(-option_data.r * option_data.T) * norm.cdf(d2)
            )
        elif option_data.option_type == 'put':
            price: float = (
                option_data.K * np.exp(-option_data.r * option_data.T) * norm.cdf(-d2) - 
                option_data.S * np.exp(-option_data.q * option_data.T) * norm.cdf(-d1)
            )
        else:
            raise ValueError(f"Invalid option_type: {option_data.option_type}. Must be 'call' or 'put'")
            
        return price
    
    @staticmethod
    def delta(option_data: OptionData) -> float:
        """
        Calculate option delta (sensitivity to underlying price changes).
        
        Args:
            option_data: OptionData instance
            
        Returns:
            Delta value (between -1 and 1)
        """
        d1: float = (
            np.log(option_data.S / option_data.K) + 
            (option_data.r - option_data.q + 0.5 * option_data.sigma ** 2) * option_data.T
        ) / (option_data.sigma * np.sqrt(option_data.T))
        
        if option_data.option_type == 'call':
            return np.exp(-option_data.q * option_data.T) * norm.cdf(d1)
        else:
            return np.exp(-option_data.q * option_data.T) * (norm.cdf(d1) - 1)
    
    @staticmethod
    def gamma(option_data: OptionData) -> float:
        """
        Calculate option gamma (sensitivity of delta to underlying price changes).
        
        Args:
            option_data: OptionData instance
            
        Returns:
            Gamma value
        """
        d1: float = (
            np.log(option_data.S / option_data.K) + 
            (option_data.r - option_data.q + 0.5 * option_data.sigma ** 2) * option_data.T
        ) / (option_data.sigma * np.sqrt(option_data.T))
        
        return (
            np.exp(-option_data.q * option_data.T) * norm.pdf(d1) / 
            (option_data.S * option_data.sigma * np.sqrt(option_data.T))
        )
    
    @staticmethod
    def vega(option_data: OptionData) -> float:
        """
        Calculate option vega (sensitivity to volatility changes).
        
        Args:
            option_data: OptionData instance
            
        Returns:
            Vega value (change in price per 1% change in volatility)
        """
        d1: float = (
            np.log(option_data.S / option_data.K) + 
            (option_data.r - option_data.q + 0.5 * option_data.sigma ** 2) * option_data.T
        ) / (option_data.sigma * np.sqrt(option_data.T))
        
        return (
            option_data.S * np.exp(-option_data.q * option_data.T) * 
            norm.pdf(d1) * np.sqrt(option_data.T)
        )
    
    @staticmethod
    def theta(option_data: OptionData) -> float:
        """
        Calculate option theta (time decay).
        
        Args:
            option_data: OptionData instance
            
        Returns:
            Theta value (change in price per day)
        """
        d1: float = (
            np.log(option_data.S / option_data.K) + 
            (option_data.r - option_data.q + 0.5 * option_data.sigma ** 2) * option_data.T
        ) / (option_data.sigma * np.sqrt(option_data.T))
        
        d2: float = d1 - option_data.sigma * np.sqrt(option_data.T)
        
        if option_data.option_type == 'call':
            theta: float = (
                -option_data.S * np.exp(-option_data.q * option_data.T) * norm.pdf(d1) * 
                option_data.sigma / (2 * np.sqrt(option_data.T)) -
                option_data.r * option_data.K * np.exp(-option_data.r * option_data.T) * norm.cdf(d2) +
                option_data.q * option_data.S * np.exp(-option_data.q * option_data.T) * norm.cdf(d1)
            )
        else:
            theta: float = (
                -option_data.S * np.exp(-option_data.q * option_data.T) * norm.pdf(d1) * 
                option_data.sigma / (2 * np.sqrt(option_data.T)) +
                option_data.r * option_data.K * np.exp(-option_data.r * option_data.T) * norm.cdf(-d2) -
                option_data.q * option_data.S * np.exp(-option_data.q * option_data.T) * norm.cdf(-d1)
            )
        
        return theta / 365  # Convert to daily theta