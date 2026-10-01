"""
Tests for market data fetching module.

Uses mocking to test data fetching without actual API calls.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from src.data.market_data import OptionDataFetcher, resolve_dividend_yield, MAX_PLAUSIBLE_DIVIDEND_YIELD


class TestOptionDataFetcherInit:
    """Test OptionDataFetcher initialization."""
    
    @pytest.mark.unit
    def test_init_creates_instance(self):
        """Test fetcher can be instantiated."""
        with patch('src.data.market_data.yf.Ticker'):
            fetcher = OptionDataFetcher('SPY')
            assert fetcher.symbol == 'SPY'
    
    @pytest.mark.unit
    def test_init_stores_symbol(self):
        """Test symbol is stored correctly."""
        with patch('src.data.market_data.yf.Ticker'):
            fetcher = OptionDataFetcher('AAPL')
            assert fetcher.symbol == 'AAPL'
            assert isinstance(fetcher.symbol, str)


class TestFetchSpotPrice:
    """Test _fetch_spot_price method."""
    
    @pytest.mark.unit
    def test_fetch_spot_price_success(self):
        """Test successful spot price fetch."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            # Mock history data
            mock_history = pd.DataFrame({
                'Close': [100.0, 101.0, 102.0]
            })
            mock_ticker.return_value.history.return_value = mock_history
            
            fetcher = OptionDataFetcher('SPY')
            spot_price = fetcher._fetch_spot_price()
            
            assert spot_price == 102.0
            assert isinstance(spot_price, float)
    
    @pytest.mark.unit
    def test_fetch_spot_price_empty_data(self):
        """Test error handling for empty spot data."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            # Mock empty history
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            
            fetcher = OptionDataFetcher('INVALID')
            
            with pytest.raises(ValueError, match='Failed to retrieve spot price'):
                fetcher._fetch_spot_price()
    
    @pytest.mark.unit
    def test_fetch_spot_price_negative_value(self):
        """Test error handling for negative spot price."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            # Mock negative price
            mock_history = pd.DataFrame({
                'Close': [-10.0]
            })
            mock_ticker.return_value.history.return_value = mock_history
            
            fetcher = OptionDataFetcher('SPY')
            
            with pytest.raises(ValueError, match='Invalid spot price'):
                fetcher._fetch_spot_price()


class TestFetchExpirationDates:
    """Test _fetch_expiration_dates method."""
    
    @pytest.mark.unit
    def test_fetch_expiration_dates_success(self):
        """Test successful expiration date fetch."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            # Mock expiration dates
            today = pd.Timestamp.now()
            future_dates = [
                (today + timedelta(days=30)).strftime('%Y-%m-%d'),
                (today + timedelta(days=60)).strftime('%Y-%m-%d'),
                (today + timedelta(days=90)).strftime('%Y-%m-%d'),
            ]
            mock_ticker.return_value.options = future_dates
            
            fetcher = OptionDataFetcher('SPY')
            exp_dates = fetcher._fetch_expiration_dates()
            
            assert len(exp_dates) == 3
            assert all(isinstance(d, pd.Timestamp) for d in exp_dates)
    
    @pytest.mark.unit
    def test_fetch_expiration_filters_near_expiry(self):
        """Test that near-expiry dates are filtered out."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            today = pd.Timestamp.now()
            dates = [
                (today + timedelta(days=1)).strftime('%Y-%m-%d'),  # Too soon
                (today + timedelta(days=5)).strftime('%Y-%m-%d'),  # Too soon
                (today + timedelta(days=30)).strftime('%Y-%m-%d'), # Good
            ]
            mock_ticker.return_value.options = dates
            
            fetcher = OptionDataFetcher('SPY')
            exp_dates = fetcher._fetch_expiration_dates()
            
            # Should only get the 30-day one (>7 days out)
            assert len(exp_dates) == 1
    
    @pytest.mark.unit
    def test_fetch_expiration_no_dates(self):
        """Test error handling when no expiration dates available."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.options = []
            
            fetcher = OptionDataFetcher('SPY')
            
            with pytest.raises(ValueError, match='No option expiration dates'):
                fetcher._fetch_expiration_dates()


class TestGetDividendYield:
    """Test get_dividend_yield method."""
    
    @pytest.mark.unit
    def test_get_dividend_yield_success(self):
        """Test successful dividend yield fetch."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.info = {'dividendYield': 2.0}
            
            fetcher = OptionDataFetcher('SPY')
            div_yield = fetcher.get_dividend_yield()
            
            # With no unit-safe reference, dividendYield is read as a percentage
            assert div_yield == pytest.approx(0.02)
            assert isinstance(div_yield, float)
    
    @pytest.mark.unit
    def test_get_dividend_yield_missing(self):
        """Test dividend yield returns 0 when missing."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.info = {}
            
            fetcher = OptionDataFetcher('SPY')
            div_yield = fetcher.get_dividend_yield()
            
            assert div_yield == 0.0
    
    @pytest.mark.unit
    def test_get_dividend_yield_none(self):
        """Test dividend yield returns 0 when None."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.info = {'dividendYield': None}
            
            fetcher = OptionDataFetcher('SPY')
            div_yield = fetcher.get_dividend_yield()
            
            assert div_yield == 0.0


class TestPrepareDataframe:
    """Test _prepare_dataframe method."""
    
    @pytest.mark.unit
    def test_prepare_dataframe_creates_dataframe(self):
        """Test dataframe creation."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.info = {'dividendYield': 0.01}
            
            fetcher = OptionDataFetcher('SPY')
            
            today = pd.Timestamp.now().normalize()
            option_data = [
                {
                    'expiration': today + timedelta(days=30),
                    'strike': 100.0,
                    'price': 5.0,
                    'type': 'call',
                    'volume': 100,
                    'days_to_expiry': 30
                }
            ]
            
            df = fetcher._prepare_dataframe(
                option_data=option_data,
                spot_price=100.0,
                risk_free_rate=0.045
            )
            
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 1
            assert 'T' in df.columns
            assert 'S' in df.columns
            assert 'r' in df.columns
            assert 'q' in df.columns
            assert 'moneyness' in df.columns
    
    @pytest.mark.unit
    def test_prepare_dataframe_calculates_fields(self):
        """Test calculated fields are correct."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.info = {'dividendYield': 0.01}
            
            fetcher = OptionDataFetcher('SPY')
            
            today = pd.Timestamp.now().normalize()
            option_data = [
                {
                    'expiration': today + timedelta(days=365),  # 1 year
                    'strike': 110.0,
                    'price': 5.0,
                    'type': 'call',
                    'volume': 100,
                    'days_to_expiry': 365
                }
            ]
            
            df = fetcher._prepare_dataframe(
                option_data=option_data,
                spot_price=100.0,
                risk_free_rate=0.045
            )
            
            assert df['T'].iloc[0] == pytest.approx(1.0, rel=0.01)  # 1 year
            assert df['S'].iloc[0] == 100.0
            assert df['r'].iloc[0] == 0.045
            assert df['moneyness'].iloc[0] == 1.1  # 110/100


class TestLogDataQualityMetrics:
    """Test _log_data_quality_metrics method."""
    
    @pytest.mark.unit
    def test_log_quality_metrics_with_data(self):
        """Test logging quality metrics with valid data."""
        with patch('src.data.market_data.yf.Ticker'):
            fetcher = OptionDataFetcher('SPY')
            
            df = pd.DataFrame({
                'T': [0.5, 1.0, 1.5],
                'strike': [95.0, 100.0, 105.0],
                'moneyness': [0.95, 1.0, 1.05]
            })
            
            # Should not raise any exceptions
            fetcher._log_data_quality_metrics(df)
    
    @pytest.mark.unit
    def test_log_quality_metrics_empty_dataframe(self):
        """Test logging with empty dataframe."""
        with patch('src.data.market_data.yf.Ticker'):
            fetcher = OptionDataFetcher('SPY')
            
            df = pd.DataFrame()
            
            # Should handle gracefully
            fetcher._log_data_quality_metrics(df)


class TestPrepareForIVIntegration:
    """Integration tests for prepare_for_iv method."""
    
    @pytest.mark.integration
    def test_prepare_for_iv_full_workflow(self):
        """Test complete workflow with mocked data."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            # Mock spot price
            mock_history = pd.DataFrame({'Close': [100.0]})
            mock_ticker.return_value.history.return_value = mock_history
            
            # Mock expiration dates
            today = pd.Timestamp.now()
            mock_ticker.return_value.options = [
                (today + timedelta(days=30)).strftime('%Y-%m-%d'),
            ]
            
            # Mock option chain
            mock_chain = Mock()
            mock_chain.calls = pd.DataFrame({
                'strike': [95.0, 100.0, 105.0],
                'bid': [7.0, 5.0, 3.0],
                'ask': [7.5, 5.5, 3.5],
                'volume': [100, 200, 150]
            })
            mock_ticker.return_value.option_chain.return_value = mock_chain
            
            # Mock dividend
            mock_ticker.return_value.info = {'dividendYield': 0.01}
            
            fetcher = OptionDataFetcher('SPY')
            result = fetcher.prepare_for_iv()
            
            assert isinstance(result, pd.DataFrame)
            assert len(result) > 0
            assert all(col in result.columns for col in ['strike', 'price', 'S', 'T', 'r', 'q'])
    
    @pytest.mark.integration
    def test_prepare_for_iv_with_filters(self):
        """Test prepare_for_iv with custom filters."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            mock_history = pd.DataFrame({'Close': [100.0]})
            mock_ticker.return_value.history.return_value = mock_history
            
            today = pd.Timestamp.now()
            mock_ticker.return_value.options = [
                (today + timedelta(days=30)).strftime('%Y-%m-%d'),
            ]
            
            mock_chain = Mock()
            mock_chain.calls = pd.DataFrame({
                'strike': [80.0, 90.0, 100.0, 110.0, 120.0],  # Wide range
                'bid': [20.0, 10.0, 5.0, 2.0, 0.5],
                'ask': [20.5, 10.5, 5.5, 2.5, 1.0],
                'volume': [50, 100, 200, 100, 50]
            })
            mock_ticker.return_value.option_chain.return_value = mock_chain
            mock_ticker.return_value.info = {}
            
            fetcher = OptionDataFetcher('SPY')
            result = fetcher.prepare_for_iv(
                min_strike_pct=90.0,  # Filter out 80
                max_strike_pct=110.0,  # Filter out 120
                min_volume=75  # Filter out 50s
            )
            
            # Should only have 90, 100, 110 strikes with volume >= 75
            assert len(result) == 3
            assert result['strike'].min() >= 90.0
            assert result['strike'].max() <= 110.0


class TestErrorHandling:
    """Test error handling in various scenarios."""
    
    @pytest.mark.validation
    def test_connection_error_handling(self):
        """Test handling of connection errors."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.history.side_effect = ConnectionError("Network error")
            
            fetcher = OptionDataFetcher('SPY')
            
            with pytest.raises(ConnectionError):
                fetcher._fetch_spot_price()
    
    @pytest.mark.validation
    def test_invalid_ticker_handling(self):
        """Test handling of invalid ticker."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.history.return_value = pd.DataFrame()
            
            fetcher = OptionDataFetcher('INVALID123')
            
            with pytest.raises(ValueError):
                fetcher._fetch_spot_price()


class TestResolveDividendYield:
    """Unit handling for Yahoo's dividendYield field (decimal vs percentage)."""

    @pytest.mark.unit
    def test_percentage_reading_chosen_when_rate_confirms_it(self):
        """AAPL-like: dividendYield 0.32 means 0.32%, confirmed by rate/price."""
        info = {'dividendYield': 0.32, 'dividendRate': 1.04, 'trailingAnnualDividendYield': 0.0031}
        assert resolve_dividend_yield(info, spot_price=330.0) == pytest.approx(0.0032)

    @pytest.mark.unit
    def test_decimal_reading_chosen_when_rate_confirms_it(self):
        """Older yfinance versions: dividendYield already a decimal."""
        info = {'dividendYield': 0.0032, 'dividendRate': 1.04}
        assert resolve_dividend_yield(info, spot_price=330.0) == pytest.approx(0.0032)

    @pytest.mark.unit
    def test_trailing_yield_used_as_reference_without_rate(self):
        info = {'dividendYield': 1.13, 'trailingAnnualDividendYield': 0.0112}
        assert resolve_dividend_yield(info) == pytest.approx(0.0113)

    @pytest.mark.unit
    def test_reference_wins_when_neither_reading_is_close(self):
        info = {'dividendYield': 7.0, 'dividendRate': 1.0}
        assert resolve_dividend_yield(info, spot_price=100.0) == pytest.approx(0.01)

    @pytest.mark.unit
    def test_trailing_only(self):
        assert resolve_dividend_yield({'trailingAnnualDividendYield': 0.02}) == pytest.approx(0.02)

    @pytest.mark.unit
    def test_ambiguous_value_assumed_percentage(self):
        """Safe default: a wrong guess gives ~0, never a 100x overstatement."""
        assert resolve_dividend_yield({'dividendYield': 0.32}) == pytest.approx(0.0032)

    @pytest.mark.unit
    def test_non_payer(self):
        assert resolve_dividend_yield({'dividendYield': 0.0}) == 0.0
        assert resolve_dividend_yield({'dividendRate': 0.0, 'dividendYield': 0.5}, spot_price=100.0) == 0.0
        assert resolve_dividend_yield({}) == 0.0

    @pytest.mark.unit
    def test_implausible_yield_rejected(self):
        info = {'trailingAnnualDividendYield': 0.9}
        assert resolve_dividend_yield(info) == 0.0
        assert MAX_PLAUSIBLE_DIVIDEND_YIELD < 1.0

    @pytest.mark.unit
    @pytest.mark.validation
    def test_non_numeric_and_negative_values_ignored(self):
        info = {'dividendYield': 'n/a', 'dividendRate': None, 'trailingAnnualDividendYield': -0.01}
        assert resolve_dividend_yield(info, spot_price=100.0) == 0.0
        assert resolve_dividend_yield({'dividendYield': float('nan')}) == 0.0

    @pytest.mark.unit
    def test_get_dividend_yield_uses_spot_price(self):
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.info = {'dividendYield': 0.32, 'dividendRate': 1.04}
            fetcher = OptionDataFetcher('AAPL')
            assert fetcher.get_dividend_yield(spot_price=330.0) == pytest.approx(0.0032)

    @pytest.mark.unit
    def test_get_dividend_yield_handles_non_dict_info(self):
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.info = None
            assert OptionDataFetcher('SPY').get_dividend_yield() == 0.0

    @pytest.mark.unit
    def test_prepare_dataframe_passes_spot_to_resolver(self):
        """The q column should reflect the unit-safe yield, not the raw Yahoo number."""
        with patch('src.data.market_data.yf.Ticker') as mock_ticker:
            mock_ticker.return_value.info = {'dividendYield': 0.32, 'dividendRate': 1.04}
            fetcher = OptionDataFetcher('AAPL')
            today = pd.Timestamp.now().normalize()
            df = fetcher._prepare_dataframe(
                option_data=[{'expiration': today + timedelta(days=30), 'strike': 330.0, 'price': 10.0,
                              'type': 'call', 'volume': 10, 'days_to_expiry': 30}],
                spot_price=330.0, risk_free_rate=0.04)
            assert df['q'].iloc[0] == pytest.approx(0.0032)


class TestResolveDividendYieldRealYahooData:
    """Values captured from yfinance 1.7.0 (dividendYield is reported as a percentage)."""

    @pytest.mark.unit
    def test_aapl_real_values(self):
        info = {'dividendYield': 0.33, 'dividendRate': 1.08, 'trailingAnnualDividendYield': 0.0031876138}
        assert resolve_dividend_yield(info, spot_price=333.02) == pytest.approx(0.0033)

    @pytest.mark.unit
    def test_spy_real_values_no_dividend_rate(self):
        """ETFs have no dividendRate; the trailing yield acts as the unit reference."""
        info = {'dividendYield': 0.98, 'dividendRate': None, 'trailingAnnualDividendYield': 0.0074090553}
        assert resolve_dividend_yield(info, spot_price=762.63) == pytest.approx(0.0098)

    @pytest.mark.unit
    def test_tsla_real_values_non_payer(self):
        info = {'dividendYield': None, 'dividendRate': None, 'trailingAnnualDividendYield': 0.0}
        assert resolve_dividend_yield(info, spot_price=354.81) == 0.0