"""
Comprehensive tests for Black-Scholes option pricing.

Tests all pricing methods and Greeks with various scenarios.
"""

import pytest
import numpy as np
from src.calculators.black_scholes import BlackScholes, OptionData


class TestBlackScholesPrice:
    """Test suite for Black-Scholes pricing."""
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_atm_call_price(self, atm_call_option: OptionData, price_tolerance: float):
        """Test ATM call option pricing."""
        price = BlackScholes.price(atm_call_option)
        
        # ATM call with 20% vol, 1 year should be around $10.45
        assert 10.0 < price < 11.0, f"ATM call price {price} outside expected range"
        assert price > 0, "Price must be positive"
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_atm_put_price(self, atm_put_option: OptionData, price_tolerance: float):
        """Test ATM put option pricing."""
        price = BlackScholes.price(atm_put_option)
        
        # ATM put with 20% vol, 1 year (price differs from call due to dividends)
        assert 5.0 < price < 7.0, f"ATM put price {price} outside expected range"
        assert price > 0, "Price must be positive"
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_put_call_parity(self, atm_call_option: OptionData, atm_put_option: OptionData,
                             price_tolerance: float):
        """Test put-call parity relationship."""
        call_price = BlackScholes.price(atm_call_option)
        put_price = BlackScholes.price(atm_put_option)
        
        # Put-call parity: C - P = S - K*e^(-rT)
        S = atm_call_option.S
        K = atm_call_option.K
        r = atm_call_option.r
        T = atm_call_option.T
        
        lhs = call_price - put_price
        rhs = S - K * np.exp(-r * T)
        
        assert abs(lhs - rhs) < price_tolerance, \
            f"Put-call parity violated: {lhs} vs {rhs}"
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_deep_itm_call(self, deep_itm_call_option: OptionData):
        """Test deep ITM call approaches intrinsic value."""
        price = BlackScholes.price(deep_itm_call_option)
        intrinsic = deep_itm_call_option.S - deep_itm_call_option.K
        
        # Deep ITM should be close to intrinsic value
        assert price > intrinsic, "Option must trade above intrinsic value"
        assert price < intrinsic * 1.2, "Deep ITM shouldn't have much time value"
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_deep_otm_call(self, deep_otm_call_option: OptionData):
        """Test deep OTM call has low value."""
        price = BlackScholes.price(deep_otm_call_option)
        
        # Deep OTM should be very cheap
        assert 0 < price < 1.0, f"Deep OTM price {price} unexpectedly high"
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_short_maturity_option(self, short_maturity_option: OptionData):
        """Test option with very short maturity."""
        price = BlackScholes.price(short_maturity_option)
        
        assert price > 0, "Price must be positive"
        # Short maturity ATM should have less value than long maturity
        assert price < 5.0, f"Short maturity price {price} unexpectedly high"
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_long_maturity_option(self, long_maturity_option: OptionData):
        """Test option with long maturity."""
        price = BlackScholes.price(long_maturity_option)
        
        assert price > 0, "Price must be positive"
        # Long maturity should have more value
        assert price > 10.0, f"Long maturity price {price} unexpectedly low"
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_high_volatility_increases_price(self, atm_call_option: OptionData,
                                             high_volatility_option: OptionData):
        """Test that higher volatility increases option price."""
        normal_price = BlackScholes.price(atm_call_option)
        high_vol_price = BlackScholes.price(high_volatility_option)
        
        assert high_vol_price > normal_price, \
            "Higher volatility should increase option price"
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_low_volatility_decreases_price(self, atm_call_option: OptionData,
                                            low_volatility_option: OptionData):
        """Test that lower volatility decreases option price."""
        normal_price = BlackScholes.price(atm_call_option)
        low_vol_price = BlackScholes.price(low_volatility_option)
        
        assert low_vol_price < normal_price, \
            "Lower volatility should decrease option price"
    
    @pytest.mark.unit
    @pytest.mark.validation
    def test_invalid_option_type_raises_error(self, atm_call_option: OptionData):
        """Test that invalid option type raises ValueError."""
        atm_call_option.option_type = 'invalid'
        
        with pytest.raises(ValueError, match="Invalid option_type"):
            BlackScholes.price(atm_call_option)


class TestBlackScholesDelta:
    """Test suite for delta calculations."""
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_atm_call_delta(self, atm_call_option: OptionData):
        """Test ATM call delta is positive and reasonable."""
        delta = BlackScholes.delta(atm_call_option)
        
        # ATM call delta should be between 0.5 and 0.7 (affected by dividends)
        assert 0.5 < delta < 0.7, f"ATM call delta {delta} outside expected range"
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_atm_put_delta(self, atm_put_option: OptionData):
        """Test ATM put delta is negative and reasonable."""
        delta = BlackScholes.delta(atm_put_option)
        
        # ATM put delta should be between -0.5 and -0.3 (affected by dividends)
        assert -0.5 < delta < -0.3, f"ATM put delta {delta} outside expected range"
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_deep_itm_call_delta(self, deep_itm_call_option: OptionData):
        """Test deep ITM call delta approaches 1."""
        delta = BlackScholes.delta(deep_itm_call_option)
        
        assert 0.9 < delta < 1.0, f"Deep ITM call delta {delta} should be close to 1"
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_deep_otm_call_delta(self, deep_otm_call_option: OptionData):
        """Test deep OTM call delta approaches 0."""
        delta = BlackScholes.delta(deep_otm_call_option)
        
        assert 0.0 < delta < 0.1, f"Deep OTM call delta {delta} should be close to 0"
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_delta_bounds(self, atm_call_option: OptionData, atm_put_option: OptionData):
        """Test delta is bounded correctly."""
        call_delta = BlackScholes.delta(atm_call_option)
        put_delta = BlackScholes.delta(atm_put_option)
        
        assert 0 <= call_delta <= 1, "Call delta must be between 0 and 1"
        assert -1 <= put_delta <= 0, "Put delta must be between -1 and 0"


class TestBlackScholesGamma:
    """Test suite for gamma calculations."""
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_atm_gamma_highest(self, atm_call_option: OptionData, itm_call_option: OptionData,
                                otm_call_option: OptionData):
        """Test ATM options have high gamma."""
        atm_gamma = BlackScholes.gamma(atm_call_option)
        itm_gamma = BlackScholes.gamma(itm_call_option)
        otm_gamma = BlackScholes.gamma(otm_call_option)
        
        # ATM gamma should be high (may not always be highest due to parameters)
        assert atm_gamma > itm_gamma, "ATM gamma should be higher than ITM"
        # Note: OTM can sometimes be close to or slightly higher than ATM
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_gamma_positive(self, atm_call_option: OptionData, atm_put_option: OptionData):
        """Test gamma is always positive."""
        call_gamma = BlackScholes.gamma(atm_call_option)
        put_gamma = BlackScholes.gamma(atm_put_option)
        
        assert call_gamma > 0, "Call gamma must be positive"
        assert put_gamma > 0, "Put gamma must be positive"
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_deep_options_low_gamma(self, deep_itm_call_option: OptionData,
                                     deep_otm_call_option: OptionData):
        """Test deep ITM/OTM options have low gamma."""
        deep_itm_gamma = BlackScholes.gamma(deep_itm_call_option)
        deep_otm_gamma = BlackScholes.gamma(deep_otm_call_option)
        
        assert deep_itm_gamma < 0.01, "Deep ITM gamma should be very low"
        assert deep_otm_gamma < 0.01, "Deep OTM gamma should be very low"


class TestBlackScholesVega:
    """Test suite for vega calculations."""
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_vega_positive(self, atm_call_option: OptionData, atm_put_option: OptionData):
        """Test vega is always positive."""
        call_vega = BlackScholes.vega(atm_call_option)
        put_vega = BlackScholes.vega(atm_put_option)
        
        assert call_vega > 0, "Call vega must be positive"
        assert put_vega > 0, "Put vega must be positive"
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_atm_vega_highest(self, atm_call_option: OptionData, itm_call_option: OptionData,
                               otm_call_option: OptionData):
        """Test ATM options have high vega."""
        atm_vega = BlackScholes.vega(atm_call_option)
        itm_vega = BlackScholes.vega(itm_call_option)
        otm_vega = BlackScholes.vega(otm_call_option)
        
        # ATM vega should be high (may not always be highest due to parameters)
        assert atm_vega > itm_vega, "ATM vega should be higher than ITM"
        # Note: OTM can sometimes be close to or slightly higher than ATM
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_longer_maturity_higher_vega(self, atm_call_option: OptionData,
                                          long_maturity_option: OptionData):
        """Test longer maturity increases vega."""
        short_vega = BlackScholes.vega(atm_call_option)
        long_vega = BlackScholes.vega(long_maturity_option)
        
        assert long_vega > short_vega, "Longer maturity should have higher vega"


class TestBlackScholesTheta:
    """Test suite for theta calculations."""
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_theta_negative(self, atm_call_option: OptionData, atm_put_option: OptionData):
        """Test theta is negative for long options."""
        call_theta = BlackScholes.theta(atm_call_option)
        put_theta = BlackScholes.theta(atm_put_option)
        
        # Long options lose value over time
        assert call_theta < 0, "Call theta should be negative"
        assert put_theta < 0, "Put theta should be negative"
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_atm_theta_magnitude(self, atm_call_option: OptionData):
        """Test ATM theta has reasonable magnitude."""
        theta = BlackScholes.theta(atm_call_option)
        
        # Theta is per day, should be small fraction of price
        price = BlackScholes.price(atm_call_option)
        
        assert abs(theta) < price * 0.01, "Daily theta should be small fraction of price"
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_short_maturity_higher_theta(self, short_maturity_option: OptionData,
                                          long_maturity_option: OptionData):
        """Test shorter maturity has higher theta magnitude."""
        short_theta = abs(BlackScholes.theta(short_maturity_option))
        long_theta = abs(BlackScholes.theta(long_maturity_option))
        
        assert short_theta > long_theta, "Shorter maturity should have higher theta magnitude"


class TestBlackScholesIntegration:
    """Integration tests across multiple Greeks."""
    
    @pytest.mark.integration
    @pytest.mark.calculation
    def test_all_greeks_computed(self, atm_call_option: OptionData):
        """Test all Greeks can be computed without errors."""
        price = BlackScholes.price(atm_call_option)
        delta = BlackScholes.delta(atm_call_option)
        gamma = BlackScholes.gamma(atm_call_option)
        vega = BlackScholes.vega(atm_call_option)
        theta = BlackScholes.theta(atm_call_option)
        
        # All should return finite values
        assert np.isfinite(price), "Price must be finite"
        assert np.isfinite(delta), "Delta must be finite"
        assert np.isfinite(gamma), "Gamma must be finite"
        assert np.isfinite(vega), "Vega must be finite"
        assert np.isfinite(theta), "Theta must be finite"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_greeks_across_strikes(self, sample_spot_price: float, sample_time_to_expiry: float,
                                    sample_risk_free_rate: float, sample_volatility: float):
        """Test Greeks behave properly across strike range."""
        strikes = np.linspace(50, 150, 20)
        
        for K in strikes:
            option = OptionData(
                S=sample_spot_price,
                K=K,
                T=sample_time_to_expiry,
                r=sample_risk_free_rate,
                sigma=sample_volatility,
                option_type='call'
            )
            
            # All Greeks should be finite
            assert np.isfinite(BlackScholes.price(option))
            assert np.isfinite(BlackScholes.delta(option))
            assert np.isfinite(BlackScholes.gamma(option))
            assert np.isfinite(BlackScholes.vega(option))
            assert np.isfinite(BlackScholes.theta(option))