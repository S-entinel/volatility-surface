import numpy as np
from scipy.stats import norm
from typing import Optional
from .black_scholes import BlackScholes, OptionData

class IVCalculator:
    def __init__(self, max_iterations: int = 100, precision: float = 1e-5):
        self.max_iterations = max_iterations
        self.precision = precision
    
    def calculate_vega(self, option_data: OptionData) -> float:
        """Calculate option vega (derivative with respect to volatility)"""
        S, K, T, r = option_data.S, option_data.K, option_data.T, option_data.r
        sigma = option_data.sigma
        
        # Add safety checks
        if sigma <= 0 or T <= 0:
            return 0.0
            
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        return vega
    
    def calculate_iv(self, 
                    S: float, 
                    K: float, 
                    T: float, 
                    r: float, 
                    market_price: float,
                    option_type: str = 'call',
                    sigma_init: float = 0.3) -> Optional[float]:
        """
        Calculate implied volatility using Newton-Raphson method
        Returns None if the algorithm doesn't converge
        """
        # Add input validation
        if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
            return None
            
        sigma = sigma_init
        
        for _ in range(self.max_iterations):
            try:
                option_data = OptionData(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type)
                
                # Calculate current price and difference from market price
                price = BlackScholes.price(option_data)
                diff = price - market_price
                
                # Check if we've reached desired precision
                if abs(diff) < self.precision:
                    return sigma
                
                # Calculate vega and update sigma
                vega = self.calculate_vega(option_data)
                if abs(vega) < 1e-10:  # Avoid division by zero
                    return None
                    
                sigma = sigma - diff/vega  # Newton-Raphson update step
                
                # Check if sigma is within reasonable bounds
                if sigma <= 0.001 or sigma > 5:  # 500% vol is a reasonable upper bound
                    return None
                    
            except (ValueError, RuntimeWarning) as e:
                return None
                
        return None  # Did not converge