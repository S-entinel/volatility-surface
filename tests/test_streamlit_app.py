"""End-to-end tests of the Streamlit app, run headlessly with a fake market data feed."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.calculators.implied_volatility import bs_call_price, bs_put_price
from src.config.config import MarketDataConfig, UIConfig

APP_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "streamlit_app.py").exists())
APP_PATH = str(APP_ROOT / "streamlit_app.py")
FETCH = 'src.data.market_data.OptionDataFetcher.prepare_for_iv'

SPOT, RATE, DIVIDEND, VOL = 100.0, 0.015, 0.01, 0.20


@pytest.fixture(autouse=True)
def clear_streamlit_cache():
    """The app caches fetched chains, so every test starts with an empty cache."""
    st.cache_data.clear()
    yield
    st.cache_data.clear()


def make_chain(days=(30, 60, 90, 180), strikes=None, open_interest=None) -> pd.DataFrame:
    """A call and a put for every strike and expiry, priced at a flat 20% volatility."""
    strikes = np.arange(85.0, 116.0, 2.5) if strikes is None else strikes
    open_interest = open_interest or {}
    today = pd.Timestamp.now().normalize()
    rows = []
    for d in days:
        T = d / 365
        for K in strikes:
            for option_type, pricer in (('call', bs_call_price), ('put', bs_put_price)):
                rows.append(dict(
                    expiration=today + pd.Timedelta(days=d), strike=float(K),
                    price=pricer(SPOT, K, T, RATE, VOL, DIVIDEND), type=option_type,
                    open_interest=open_interest.get(d, 100),
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


def number_input(at, label):
    return next(n for n in at.number_input if n.label == label)


def point_count(at) -> int:
    return len(at.session_state['analysis_results']['valid_options'])


class TestNormalRun:
    @pytest.mark.integration
    def test_runs_without_errors(self):
        at, _ = open_app(make_chain())

        assert len(at.exception) == 0
        assert len(at.error) == 0

    @pytest.mark.integration
    def test_only_out_of_the_money_options_are_used(self):
        chain = make_chain()
        at, _ = open_app(chain)
        results = at.session_state['analysis_results']

        assert len(results['valid_options']) == expected_otm_count(chain)
        assert len(results['valid_options']) < len(chain), "the in-the-money side must be discarded"
        assert results['n_puts'] > 0 and results['n_calls'] > 0
        for option in results['valid_options']:
            forward = SPOT * np.exp((RATE - DIVIDEND) * option['T'])
            assert (option['strike'] < forward) if option['type'] == 'put' else (option['strike'] >= forward)

    @pytest.mark.integration
    def test_recovers_the_flat_volatility(self):
        at, _ = open_app(make_chain())
        results = at.session_state['analysis_results']

        assert np.allclose(results['ivs'], VOL, atol=1e-4)
        assert results['iv_stats']['atm_iv'] == pytest.approx(20.0, abs=0.01)

    @pytest.mark.integration
    def test_skew_is_a_number_and_zero_on_a_flat_surface(self):
        at, _ = open_app(make_chain())
        skew = at.session_state['analysis_results']['iv_stats']['iv_skew']

        assert skew is not None
        assert skew == pytest.approx(0.0, abs=0.01)

    @pytest.mark.integration
    def test_captions_describe_the_option_mix_and_dividend(self):
        at, _ = open_app(make_chain())
        captions = [c.value for c in at.caption]

        assert any(c.startswith("Built from out-of-the-money options only") for c in captions)
        assert any(c.startswith("Dividend yield used") and "looked up" in c for c in captions)

    @pytest.mark.integration
    def test_caption_says_when_the_data_was_fetched(self):
        at, _ = open_app(make_chain())

        assert any(c.value.startswith("Data fetched at") for c in at.caption)

    @pytest.mark.integration
    def test_dividend_override_is_used_and_reported(self):
        at, _ = open_app(make_chain(), [lambda a: a.checkbox[0].check().run()])
        results = at.session_state['analysis_results']

        assert len(at.exception) == 0
        assert results['dividend_source'] == "manual override"
        assert results['dividend_yield_used'] == pytest.approx(0.013)  # the sidebar default, 1.3%
        assert any("manual override" in c.value for c in at.caption)

    @pytest.mark.integration
    def test_unavailable_smile_slices_are_reported(self):
        """Data starting at 45 days cannot show the default 30-day slice."""
        at, _ = open_app(make_chain(days=(45, 75, 100, 180)))

        assert any("30d" in c.value and "not drawn" in c.value for c in at.caption)


class TestSidebar:
    @pytest.mark.integration
    def test_open_interest_defaults_to_zero(self):
        """Yahoo often reports no open interest before the US open, so the default must not filter."""
        at, _ = open_app(make_chain())

        assert number_input(at, "Min Open Interest").value == 0

    @pytest.mark.integration
    def test_fetches_the_widest_window_with_no_open_interest_limit(self):
        _, fetch = open_app(make_chain())

        assert fetch.call_count == 1
        assert fetch.call_args.kwargs['min_strike_pct'] == UIConfig.MIN_STRIKE_PCT_LOWER
        assert fetch.call_args.kwargs['max_strike_pct'] == UIConfig.MAX_STRIKE_PCT_UPPER
        assert fetch.call_args.kwargs['min_open_interest'] == 0

    @pytest.mark.integration
    def test_changing_settings_does_not_refetch(self):
        at, fetch = open_app(make_chain(), [
            lambda a: a.selectbox[0].set_value(UIConfig.AVAILABLE_THEMES[-1]).run(),
            lambda a: a.selectbox[2].set_value(UIConfig.AVAILABLE_Y_AXIS_TYPES[-1]).run(),
            lambda a: number_input(a, "Risk-Free Rate (%)").set_value(3.0).run(),
        ])

        assert fetch.call_count == 1
        assert len(at.exception) == 0 and 'analysis_results' in at.session_state

    @pytest.mark.integration
    def test_open_interest_filters_the_cached_chain(self):
        chain = make_chain(open_interest={30: 0, 60: 0, 90: 100, 180: 100})
        at, fetch = open_app(chain)
        before = point_count(at)

        number_input(at, "Min Open Interest").set_value(50).run()

        assert before == expected_otm_count(chain)
        assert point_count(at) == expected_otm_count(chain[chain['open_interest'] >= 50])
        assert point_count(at) < before
        assert fetch.call_count == 1

    @pytest.mark.integration
    def test_strike_range_filters_the_cached_chain(self):
        chain = make_chain()
        at, fetch = open_app(chain)

        number_input(at, "Min Strike %").set_value(95.0).run()

        assert point_count(at) == expected_otm_count(chain[chain['strike'] >= 95.0])
        assert fetch.call_count == 1

    @pytest.mark.integration
    def test_refresh_data_fetches_again(self):
        at, fetch = open_app(make_chain(), [lambda a: a.button[0].click().run()])

        assert fetch.call_count == 2
        assert 'analysis_results' in at.session_state

    @pytest.mark.integration
    def test_a_new_ticker_fetches_again(self):
        at, fetch = open_app(make_chain(), [lambda a: a.text_input[0].set_value("QQQ").run()])

        assert fetch.call_count == 2
        assert at.session_state['analysis_results']['ticker'] == "QQQ"


class TestDataProblems:
    @pytest.mark.integration
    def test_too_few_options_gives_a_clear_error(self):
        at, _ = open_app(make_chain(days=(30,), strikes=np.array([95.0, 105.0])))

        assert len(at.exception) == 0
        assert any("Insufficient options data" in e.value for e in at.error)
        assert 'analysis_results' not in at.session_state

    @pytest.mark.integration
    def test_in_the_money_options_alone_are_rejected(self):
        chain = make_chain()
        itm = chain[((chain['type'] == 'call') & (chain['strike'] < 97)) |
                    ((chain['type'] == 'put') & (chain['strike'] > 103))]
        at, _ = open_app(itm.reset_index(drop=True))

        assert any("Insufficient options data" in e.value for e in at.error)

    @pytest.mark.integration
    def test_empty_result_is_reported(self):
        at, _ = open_app(make_chain().iloc[0:0])

        assert any("No valid options data found" in e.value for e in at.error)

    @pytest.mark.integration
    def test_fetch_failure_is_reported_not_raised(self):
        at, _ = open_app(ValueError("No valid option data available after filtering"))

        assert len(at.exception) == 0
        assert any(e.value.startswith("Analysis Error") for e in at.error)

    @pytest.mark.integration
    def test_unsolvable_prices_give_a_clear_error(self):
        """Prices above any possible option value produce no implied volatilities."""
        chain = make_chain(days=(30, 60, 90))
        chain['price'] = 1e6
        at, _ = open_app(chain)

        assert any("Insufficient valid options" in e.value for e in at.error)

    @pytest.mark.integration
    def test_a_failed_fetch_is_not_repeated_when_a_setting_changes(self):
        at, fetch = open_app(ValueError("Yahoo is down"), [
            lambda a: a.selectbox[0].set_value(UIConfig.AVAILABLE_THEMES[-1]).run(),
            lambda a: number_input(a, "Min Open Interest").set_value(5).run(),
        ])

        assert fetch.call_count == 1
        assert any("Yahoo is down" in e.value for e in at.error)

    @pytest.mark.integration
    def test_refresh_data_retries_a_failed_fetch(self):
        with patch(FETCH) as fetch:
            fetch.side_effect = ValueError("Yahoo is down")
            at = AppTest.from_file(APP_PATH, default_timeout=60)
            at.run()
            assert fetch.call_count == 1 and len(at.error) == 1

            fetch.side_effect = None
            fetch.return_value = make_chain()
            at.button[0].click().run()

        assert fetch.call_count == 2
        assert len(at.error) == 0
        assert 'analysis_results' in at.session_state

    @pytest.mark.integration
    def test_a_different_ticker_is_not_blocked_by_an_earlier_failure(self):
        with patch(FETCH) as fetch:
            fetch.side_effect = ValueError("Yahoo is down")
            at = AppTest.from_file(APP_PATH, default_timeout=60)
            at.run()

            fetch.side_effect = None
            fetch.return_value = make_chain()
            at.text_input[0].set_value("QQQ").run()

        assert len(at.error) == 0
        assert at.session_state['analysis_results']['ticker'] == "QQQ"


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

        assert len(at.exception) == 0
        assert len(at.error) == 0
        results = at.session_state['analysis_results']
        assert len(results['ivs']) > 50
        assert np.allclose(results['ivs'], VOL, atol=1e-3)
        assert results['n_puts'] > 0 and results['n_calls'] > 0