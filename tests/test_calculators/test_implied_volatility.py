import pytest
from src.calculators.implied_volatility import IVCalculator
from src.calculators.black_scholes import BlackScholes, OptionData

def test_implied_volatility_calculation():
    # First, let's create an option price using a known volatility
    known_sigma = 0.2
    S = 100
    K = 100
    T = 1.0
    r = 0.05
    
    # Calculate option price with known parameters
    option_data = OptionData(S=S, K=K, T=T, r=r, sigma=known_sigma, option_type='call')
    option_price = BlackScholes.price(option_data)
    
    # Now try to recover the volatility from the price
    iv_calculator = IVCalculator()
    calculated_sigma = iv_calculator.calculate_iv(
        S=S, K=K, T=T, r=r, 
        market_price=option_price,
        option_type='call'
    )
    
    # Check if calculated volatility matches our known volatility
    assert abs(calculated_sigma - known_sigma) < 1e-4

def test_invalid_market_price():
    # Test with an invalid (negative) market price
    iv_calculator = IVCalculator()
    result = iv_calculator.calculate_iv(
        S=100, K=100, T=1, r=0.05,
        market_price=-10,  # Invalid negative price
        option_type='call'
    )
    
    assert result is None