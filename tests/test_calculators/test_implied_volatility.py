"""
Comprehensive tests for implied volatility calculation.

Tests IV solver convergence, edge cases, and validation.
"""

import pytest
import numpy as np
from src.calculators.implied_volatility import IVCalculator, bs_call_price, bs_put_price
from src.calculators.black_scholes import BlackScholes, OptionData


class TestIVCalculatorBasic:
    """Basic IV calculation tests."""
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_recover_known_volatility(self, iv_calculator: IVCalculator, atm_call_option: OptionData,
                                       iv_tolerance: float):
        """Test recovering known volatility from price."""
        known_sigma = atm_call_option.sigma
        
        # Calculate price with known volatility
        market_price = BlackScholes.price(atm_call_option)
        
        # Recover volatility
        calculated_sigma = iv_calculator.calculate_iv(
            S=atm_call_option.S,
            K=atm_call_option.K,
            T=atm_call_option.T,
            r=atm_call_option.r,
            q=atm_call_option.q,
            market_price=market_price,
            option_type='call'
        )
        
        assert calculated_sigma is not None, "IV calculation failed"
        assert abs(calculated_sigma - known_sigma) < iv_tolerance, \
            f"Failed to recover volatility: {calculated_sigma} vs {known_sigma}"
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_put_option_iv(self, iv_calculator: IVCalculator, atm_put_option: OptionData,
                           iv_tolerance: float):
        """Test IV calculation for put options."""
        known_sigma = atm_put_option.sigma
        market_price = BlackScholes.price(atm_put_option)
        
        calculated_sigma = iv_calculator.calculate_iv(
            S=atm_put_option.S,
            K=atm_put_option.K,
            T=atm_put_option.T,
            r=atm_put_option.r,
            q=atm_put_option.q,
            market_price=market_price,
            option_type='put'
        )
        
        assert calculated_sigma is not None, "IV calculation failed for put"
        assert abs(calculated_sigma - known_sigma) < iv_tolerance


class TestIVCalculatorEdgeCases:
    """Edge case testing for IV calculator."""
    
    @pytest.mark.edge_case
    @pytest.mark.calculation
    def test_deep_itm_option(self, iv_calculator: IVCalculator, deep_itm_call_option: OptionData):
        """Test IV calculation for deep ITM option."""
        market_price = BlackScholes.price(deep_itm_call_option)
        
        iv = iv_calculator.calculate_iv(
            S=deep_itm_call_option.S,
            K=deep_itm_call_option.K,
            T=deep_itm_call_option.T,
            r=deep_itm_call_option.r,
            q=deep_itm_call_option.q,
            market_price=market_price,
            option_type='call'
        )
        
        assert iv is not None, "IV calculation failed for deep ITM"
        assert 0.01 < iv < 2.0, f"IV {iv} outside reasonable range"
    
    @pytest.mark.edge_case
    @pytest.mark.calculation
    def test_deep_otm_option(self, iv_calculator: IVCalculator, deep_otm_call_option: OptionData):
        """Test IV calculation for deep OTM option."""
        market_price = BlackScholes.price(deep_otm_call_option)
        
        iv = iv_calculator.calculate_iv(
            S=deep_otm_call_option.S,
            K=deep_otm_call_option.K,
            T=deep_otm_call_option.T,
            r=deep_otm_call_option.r,
            q=deep_otm_call_option.q,
            market_price=market_price,
            option_type='call'
        )
        
        assert iv is not None, "IV calculation failed for deep OTM"
        assert 0.01 < iv < 2.0, f"IV {iv} outside reasonable range"
    
    @pytest.mark.edge_case
    @pytest.mark.calculation
    def test_high_volatility(self, iv_calculator: IVCalculator, high_volatility_option: OptionData,
                             iv_tolerance: float):
        """Test IV calculation with high volatility."""
        known_sigma = high_volatility_option.sigma
        market_price = BlackScholes.price(high_volatility_option)
        
        calculated_sigma = iv_calculator.calculate_iv(
            S=high_volatility_option.S,
            K=high_volatility_option.K,
            T=high_volatility_option.T,
            r=high_volatility_option.r,
            q=high_volatility_option.q,
            market_price=market_price,
            option_type='call'
        )
        
        assert calculated_sigma is not None, "IV calculation failed for high vol"
        assert abs(calculated_sigma - known_sigma) < iv_tolerance * 10  # Relaxed tolerance
    
    @pytest.mark.edge_case
    @pytest.mark.calculation
    def test_low_volatility(self, iv_calculator: IVCalculator, low_volatility_option: OptionData,
                            iv_tolerance: float):
        """Test IV calculation with low volatility."""
        known_sigma = low_volatility_option.sigma
        market_price = BlackScholes.price(low_volatility_option)
        
        calculated_sigma = iv_calculator.calculate_iv(
            S=low_volatility_option.S,
            K=low_volatility_option.K,
            T=low_volatility_option.T,
            r=low_volatility_option.r,
            q=low_volatility_option.q,
            market_price=market_price,
            option_type='call'
        )
        
        assert calculated_sigma is not None, "IV calculation failed for low vol"
        assert abs(calculated_sigma - known_sigma) < iv_tolerance
    
    @pytest.mark.edge_case
    @pytest.mark.calculation
    def test_short_maturity(self, iv_calculator: IVCalculator, short_maturity_option: OptionData,
                            iv_tolerance: float):
        """Test IV calculation for very short maturity."""
        known_sigma = short_maturity_option.sigma
        market_price = BlackScholes.price(short_maturity_option)
        
        calculated_sigma = iv_calculator.calculate_iv(
            S=short_maturity_option.S,
            K=short_maturity_option.K,
            T=short_maturity_option.T,
            r=short_maturity_option.r,
            q=short_maturity_option.q,
            market_price=market_price,
            option_type='call'
        )
        
        assert calculated_sigma is not None, "IV calculation failed for short maturity"
        assert abs(calculated_sigma - known_sigma) < iv_tolerance * 5  # Relaxed tolerance


class TestIVCalculatorValidation:
    """Input validation tests."""
    
    @pytest.mark.validation
    def test_invalid_spot_price(self, iv_calculator: IVCalculator, invalid_option_params):
        """Test rejection of invalid spot prices."""
        params = invalid_option_params['negative_spot']
        iv = iv_calculator.calculate_iv(**params, q=0, option_type='call')
        assert iv is None, "Should reject negative spot price"
        
        params = invalid_option_params['zero_spot']
        iv = iv_calculator.calculate_iv(**params, q=0, option_type='call')
        assert iv is None, "Should reject zero spot price"
    
    @pytest.mark.validation
    def test_invalid_strike_price(self, iv_calculator: IVCalculator, invalid_option_params):
        """Test rejection of invalid strike prices."""
        params = invalid_option_params['negative_strike']
        iv = iv_calculator.calculate_iv(**params, q=0, option_type='call')
        assert iv is None, "Should reject negative strike"
        
        params = invalid_option_params['zero_strike']
        iv = iv_calculator.calculate_iv(**params, q=0, option_type='call')
        assert iv is None, "Should reject zero strike"
    
    @pytest.mark.validation
    def test_invalid_time_to_expiry(self, iv_calculator: IVCalculator, invalid_option_params):
        """Test rejection of invalid time to expiry."""
        params = invalid_option_params['negative_time']
        iv = iv_calculator.calculate_iv(**params, q=0, option_type='call')
        assert iv is None, "Should reject negative time"
        
        params = invalid_option_params['zero_time']
        iv = iv_calculator.calculate_iv(**params, q=0, option_type='call')
        assert iv is None, "Should reject zero time"
    
    @pytest.mark.validation
    def test_invalid_market_price(self, iv_calculator: IVCalculator, invalid_option_params):
        """Test rejection of invalid market prices."""
        params = invalid_option_params['negative_price']
        iv = iv_calculator.calculate_iv(**params, q=0, option_type='call')
        assert iv is None, "Should reject negative price"
        
        params = invalid_option_params['zero_price']
        iv = iv_calculator.calculate_iv(**params, q=0, option_type='call')
        assert iv is None, "Should reject zero price"
    
    @pytest.mark.validation
    def test_invalid_option_type(self, iv_calculator: IVCalculator):
        """Test rejection of invalid option type."""
        iv = iv_calculator.calculate_iv(
            S=100.0,
            K=100.0,
            T=1.0,
            r=0.05,
            market_price=5.0,
            q=0.0,
            option_type='invalid'
        )
        assert iv is None, "Should reject invalid option type"
    
    @pytest.mark.validation
    def test_price_below_intrinsic_value(self, iv_calculator: IVCalculator):
        """Test rejection of price below intrinsic value."""
        # ITM call with price below intrinsic value (arbitrage)
        iv = iv_calculator.calculate_iv(
            S=100.0,
            K=90.0,
            T=1.0,
            r=0.05,
            market_price=5.0,  # Below intrinsic of 10
            q=0.0,
            option_type='call'
        )
        
        assert iv is None, "Should reject price below intrinsic value"


class TestIVCalculatorStatistics:
    """Test statistics tracking functionality."""
    
    @pytest.mark.unit
    def test_statistics_tracking(self, iv_calculator: IVCalculator, atm_call_option: OptionData):
        """Test that statistics are tracked correctly."""
        # Reset statistics
        iv_calculator.reset_statistics()
        
        # Perform calculations
        market_price = BlackScholes.price(atm_call_option)
        
        # Successful calculation
        iv_calculator.calculate_iv(
            S=atm_call_option.S,
            K=atm_call_option.K,
            T=atm_call_option.T,
            r=atm_call_option.r,
            q=atm_call_option.q,
            market_price=market_price,
            option_type='call'
        )
        
        # Failed calculation (invalid price)
        iv_calculator.calculate_iv(
            S=atm_call_option.S,
            K=atm_call_option.K,
            T=atm_call_option.T,
            r=atm_call_option.r,
            q=atm_call_option.q,
            market_price=-5.0,  # Invalid
            option_type='call'
        )
        
        stats = iv_calculator.get_statistics()
        
        assert stats['total'] == 2, "Should track 2 total calculations"
        assert stats['successful'] == 1, "Should have 1 success"
        assert stats['failed'] == 1, "Should have 1 failure"
        assert stats['success_rate'] == 50.0, "Success rate should be 50%"
    
    @pytest.mark.unit
    def test_statistics_reset(self, iv_calculator: IVCalculator):
        """Test statistics reset functionality."""
        # Do some calculations
        iv_calculator.calculate_iv(
            S=100, K=100, T=1, r=0.05, market_price=10, option_type='call'
        )
        
        # Reset
        iv_calculator.reset_statistics()
        
        stats = iv_calculator.get_statistics()
        assert stats['total'] == 0, "Total should be 0 after reset"
        assert stats['successful'] == 0, "Successful should be 0 after reset"
        assert stats['failed'] == 0, "Failed should be 0 after reset"


class TestBSHelperFunctions:
    """Test Black-Scholes helper functions."""
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_bs_call_price_function(self, atm_call_option: OptionData, price_tolerance: float):
        """Test standalone BS call price function."""
        price = bs_call_price(
            S=atm_call_option.S,
            K=atm_call_option.K,
            T=atm_call_option.T,
            r=atm_call_option.r,
            sigma=atm_call_option.sigma,
            q=atm_call_option.q
        )
        
        expected_price = BlackScholes.price(atm_call_option)
        
        assert abs(price - expected_price) < price_tolerance, \
            "Standalone function should match class method"
    
    @pytest.mark.unit
    @pytest.mark.calculation
    def test_bs_put_price_function(self, atm_put_option: OptionData, price_tolerance: float):
        """Test standalone BS put price function."""
        price = bs_put_price(
            S=atm_put_option.S,
            K=atm_put_option.K,
            T=atm_put_option.T,
            r=atm_put_option.r,
            sigma=atm_put_option.sigma,
            q=atm_put_option.q
        )
        
        expected_price = BlackScholes.price(atm_put_option)
        
        assert abs(price - expected_price) < price_tolerance, \
            "Standalone function should match class method"
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_zero_time_to_expiry(self):
        """Test that zero time returns intrinsic value."""
        call_price = bs_call_price(S=110, K=100, T=0, r=0.05, sigma=0.2)
        put_price = bs_put_price(S=90, K=100, T=0, r=0.05, sigma=0.2)
        
        assert call_price == 10.0, "Call at expiry should be intrinsic"
        assert put_price == 10.0, "Put at expiry should be intrinsic"


class TestIVCalculatorIntegration:
    """Integration tests across multiple scenarios."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_iv_across_strike_range(self, iv_calculator: IVCalculator, sample_spot_price: float,
                                     sample_time_to_expiry: float, sample_risk_free_rate: float):
        """Test IV calculation across range of strikes."""
        strikes = np.linspace(80, 120, 10)
        known_sigma = 0.25
        
        for K in strikes:
            option = OptionData(
                S=sample_spot_price,
                K=K,
                T=sample_time_to_expiry,
                r=sample_risk_free_rate,
                sigma=known_sigma,
                option_type='call'
            )
            
            market_price = BlackScholes.price(option)
            
            calculated_sigma = iv_calculator.calculate_iv(
                S=option.S,
                K=option.K,
                T=option.T,
                r=option.r,
                market_price=market_price,
                option_type='call'
            )
            
            assert calculated_sigma is not None, f"IV calculation failed for strike {K}"
            assert abs(calculated_sigma - known_sigma) < 0.001, \
                f"IV mismatch at strike {K}: {calculated_sigma} vs {known_sigma}"
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_iv_across_maturities(self, iv_calculator: IVCalculator, sample_spot_price: float,
                                   sample_strike_price: float, sample_risk_free_rate: float):
        """Test IV calculation across range of maturities."""
        maturities = np.linspace(0.1, 2.0, 10)
        known_sigma = 0.25
        
        for T in maturities:
            option = OptionData(
                S=sample_spot_price,
                K=sample_strike_price,
                T=T,
                r=sample_risk_free_rate,
                sigma=known_sigma,
                option_type='call'
            )
            
            market_price = BlackScholes.price(option)
            
            calculated_sigma = iv_calculator.calculate_iv(
                S=option.S,
                K=option.K,
                T=option.T,
                r=option.r,
                market_price=market_price,
                option_type='call'
            )
            
            assert calculated_sigma is not None, f"IV calculation failed for maturity {T}"
            assert abs(calculated_sigma - known_sigma) < 0.001, \
                f"IV mismatch at maturity {T}: {calculated_sigma} vs {known_sigma}"