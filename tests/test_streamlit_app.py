"""End-to-end tests of the Streamlit app, run headlessly with a fake market data feed."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from src.calculators.implied_volatility import bs_call_price, bs_put_price
from src.config.config import MarketDataConfig, UIConfig

APP_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "streamlit_app.py").exists())
APP_PATH = str(APP_ROOT / "streamlit_app.py")
FETCH = 'src.data.market_data.OptionDataFetcher.prepare_for_iv'

SPOT, RATE, DIVIDEND, VOL = 100.0, 0.015, 0.01, 0.20


def make_chain(days=(30, 60, 90, 180), strikes=None) -> pd.DataFrame:
    """A call and a put for every strike and expiry, priced at a flat 20% volatility."""
    strikes = np.arange(85.0, 116.0, 2.5) if strikes is None else strikes
    today = pd.Timestamp.now().normalize()
    rows = []
    for d in days:
        T = d / 365
        for K in strikes:
            for option_type, pricer in (('call', bs_call_price), ('put', bs_put_price)):
                rows.append(dict(
                    expiration=today + pd.Timedelta(days=d), strike=float(K),
                    price=pricer(SPOT, K, T, RATE, VOL, DIVIDEND), type=option_type,
                    days_to_expiry=d, T=T, S=SPOT, r=RATE, q=DIVIDEND, moneyness=K / SPOT))
    chain = pd.DataFrame(rows)
    return chain[chain['price'] >= MarketDataConfig.MIN_QUOTE_MID_PRICE].reset_index(drop=True)


def expected_otm_count(chain: pd.DataFrame) -> int:
    forward = SPOT * np.exp((RATE - DIVIDEND) * chain['T'])
    is_otm = (((chain['type'] == 'put') & (chain['strike'] < forward)) |
              ((chain['type'] == 'call') & (chain['strike'] >= forward)))
    return int(is_otm.sum())


def open_app(chain, steps=()):
    """Load the app with a fake fetch, run each step in turn, and return (app, fetch mock)."""
    with patch(FETCH) as fetch:
        if isinstance(chain, Exception):
            fetch.side_effect = chain
        else:
            fetch.return_value = chain
        at = AppTest.from_file(APP_PATH, default_timeout=60)
        at.run()
        for step in steps:
            step(at)
    return at, fetch


def click(at):
    at.button[0].click().run()


def number_input(at, label):
    return next(n for n in at.number_input if n.label == label)


class TestNormalRun:
    @pytest.mark.integration
    def test_runs_without_errors(self):
        at, _ = open_app(make_chain(), [click])

        assert len(at.exception) == 0
        assert len(at.error) == 0

    @pytest.mark.integration
    def test_only_out_of_the_money_options_are_used(self):
        chain = make_chain()
        at, _ = open_app(chain, [click])
        results = at.session_state['analysis_results']

        assert len(results['valid_options']) == expected_otm_count(chain)
        assert len(results['valid_options']) < len(chain), "the in-the-money side must be discarded"
        assert results['n_puts'] > 0 and results['n_calls'] > 0
        for option in results['valid_options']:
            forward = SPOT * np.exp((RATE - DIVIDEND) * option['T'])
            assert (option['strike'] < forward) if option['type'] == 'put' else (option['strike'] >= forward)

    @pytest.mark.integration
    def test_recovers_the_flat_volatility(self):
        at, _ = open_app(make_chain(), [click])
        results = at.session_state['analysis_results']

        assert np.allclose(results['ivs'], VOL, atol=1e-4)
        assert results['iv_stats']['atm_iv'] == pytest.approx(20.0, abs=0.01)

    @pytest.mark.integration
    def test_skew_is_a_number_and_zero_on_a_flat_surface(self):
        at, _ = open_app(make_chain(), [click])
        skew = at.session_state['analysis_results']['iv_stats']['iv_skew']

        assert skew is not None
        assert skew == pytest.approx(0.0, abs=0.01)

    @pytest.mark.integration
    def test_captions_describe_the_option_mix_and_dividend(self):
        at, _ = open_app(make_chain(), [click])
        captions = [c.value for c in at.caption]

        assert any(c.startswith("Built from out-of-the-money options only") for c in captions)
        assert any(c.startswith("Dividend yield used") and "looked up" in c for c in captions)

    @pytest.mark.integration
    def test_dividend_override_is_used_and_reported(self):
        at, _ = open_app(make_chain(), [lambda a: a.checkbox[0].check().run(), click])
        results = at.session_state['analysis_results']

        assert len(at.exception) == 0
        assert results['dividend_source'] == "manual override"
        assert results['dividend_yield_used'] == pytest.approx(0.013)  # the sidebar default, 1.3%
        assert any("manual override" in c.value for c in at.caption)

    @pytest.mark.integration
    def test_unavailable_smile_slices_are_reported(self):
        """Data starting at 45 days cannot show the default 30-day slice."""
        at, _ = open_app(make_chain(days=(45, 75, 100, 180)), [click])

        assert any("30d" in c.value and "not drawn" in c.value for c in at.caption)


class TestSidebar:
    @pytest.mark.integration
    def test_open_interest_defaults_to_zero(self):
        """Yahoo often reports no open interest before the US open, so the default must not filter."""
        at, _ = open_app(make_chain())

        assert number_input(at, "Min Open Interest").value == 0

    @pytest.mark.integration
    def test_open_interest_setting_reaches_the_fetcher(self):
        at, fetch = open_app(make_chain(), [
            lambda a: number_input(a, "Min Open Interest").set_value(50).run(), click])

        assert fetch.call_args.kwargs['min_open_interest'] == 50

    @pytest.mark.integration
    def test_changing_a_setting_after_success_does_not_refetch(self):
        at, fetch = open_app(make_chain(), [click])
        calls_after_success = fetch.call_count

        with patch(FETCH) as same_fetch:
            same_fetch.return_value = make_chain()
            at.selectbox[0].set_value(UIConfig.AVAILABLE_THEMES[-1]).run()

        assert calls_after_success >= 1
        assert same_fetch.call_count == 0
        assert 'analysis_results' in at.session_state


class TestDataProblems:
    @pytest.mark.integration
    def test_too_few_options_gives_a_clear_error(self):
        at, _ = open_app(make_chain(days=(30,), strikes=np.array([95.0, 105.0])), [click])

        assert len(at.exception) == 0
        assert any("Insufficient options data" in e.value for e in at.error)
        assert 'analysis_results' not in at.session_state

    @pytest.mark.integration
    def test_in_the_money_options_alone_are_rejected(self):
        chain = make_chain()
        itm = chain[((chain['type'] == 'call') & (chain['strike'] < 97)) |
                    ((chain['type'] == 'put') & (chain['strike'] > 103))]
        at, _ = open_app(itm.reset_index(drop=True), [click])

        assert any("Insufficient options data" in e.value for e in at.error)

    @pytest.mark.integration
    def test_empty_result_is_reported(self):
        at, _ = open_app(make_chain().iloc[0:0], [click])

        assert any("No valid options data found" in e.value for e in at.error)

    @pytest.mark.integration
    def test_fetch_failure_is_reported_not_raised(self):
        at, _ = open_app(ValueError("No valid option data available after filtering"), [click])

        assert len(at.exception) == 0
        assert any(e.value.startswith("Analysis Error") for e in at.error)

    @pytest.mark.integration
    def test_unsolvable_prices_give_a_clear_error(self):
        """Prices above any possible option value produce no implied volatilities."""
        chain = make_chain(days=(30, 60, 90))
        chain['price'] = 1e6
        at, _ = open_app(chain, [click])

        assert any("Insufficient valid options" in e.value for e in at.error)


class FakeYahooOvernight:
    """Yahoo outside US hours: no bids or asks, open interest 0, but every contract has a last price."""

    info = {'dividendYield': 1.0, 'dividendRate': None, 'trailingAnnualDividendYield': 0.0099}
    days = (14, 30, 60, 90, 180)

    def __init__(self, symbol):
        today = pd.Timestamp.now().normalize()
        self.today = today
        self.options = [(today + timedelta(days=d)).strftime('%Y-%m-%d') for d in self.days]

    def history(self, period='5d'):
        return pd.DataFrame({'Close': [SPOT]})

    def option_chain(self, date):
        T = (pd.Timestamp(date) - self.today).days / 365
        strikes = np.arange(80.0, 121.0, 2.5)

        def side(pricer):
            return pd.DataFrame(dict(
                strike=strikes, bid=0.0, ask=0.0,
                lastPrice=[pricer(SPOT, K, T, RATE, VOL, DIVIDEND) for K in strikes],
                volume=0.0, openInterest=0.0))

        chain = type('Chain', (), {})()
        chain.calls, chain.puts = side(bs_call_price), side(bs_put_price)
        return chain


class TestOvernightMarket:
    @pytest.mark.integration
    def test_surface_is_built_from_last_prices_when_there_are_no_quotes(self):
        """Runs the real fetcher against a feed with no live quotes and no open interest."""
        with patch('src.data.market_data.yf.Ticker', FakeYahooOvernight):
            at = AppTest.from_file(APP_PATH, default_timeout=60)
            at.run()
            click(at)

        assert len(at.exception) == 0
        assert len(at.error) == 0
        results = at.session_state['analysis_results']
        assert len(results['ivs']) > 50
        assert np.allclose(results['ivs'], VOL, atol=1e-3)
        assert results['n_puts'] > 0 and results['n_calls'] > 0