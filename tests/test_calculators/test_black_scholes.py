import pytest
from src.calculators.black_scholes import BlackScholes, OptionData

def test_call_option_price():
    option_data = OptionData(
        S=100,    # Spot price
        K=100,    # Strike price
        T=1.0,    # One year to maturity
        r=0.05,   # 5% risk-free rate
        sigma=0.2,  # 20% volatility
        q=0.0,     # No dividend
        option_type='call'
    )
    
    price = BlackScholes.price(option_data)
    expected_price = 10.45  # This is approximately what we expect
    
    assert abs(price - expected_price) < 0.1

def test_call_option_with_dividend():
    option_data = OptionData(
        S=100,    # Spot price
        K=100,    # Strike price
        T=1.0,    # One year to maturity
        r=0.05,   # 5% risk-free rate
        sigma=0.2,  # 20% volatility
        q=0.02,    # 2% dividend yield
        option_type='call'
    )
    
    price = BlackScholes.price(option_data)
    expected_price = 9.23  # Expected price with dividend
    
    assert abs(price - expected_price) < 0.1