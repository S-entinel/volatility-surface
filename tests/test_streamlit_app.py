"""End-to-end tests of the Streamlit app, run headlessly with a fake market data feed."""

from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src.analysis import build_heatmap_figure, key_figures
from src.calculators.implied_volatility import bs_call_price, bs_put_price
from src.config.config import MarketDataConfig, ModelConfig, UIConfig

APP_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "streamlit_app.py").exists())
APP_PATH = str(APP_ROOT / "streamlit_app.py")
FETCH = 'src.data.market_data.OptionDataFetcher.prepare_for_iv'

SPOT, RATE, DIVIDEND, VOL = 100.0, ModelConfig.DEFAULT_RISK_FREE_RATE, 0.01, 0.20


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


def widget(elements, label):
    return next(e for e in elements if e.label == label)


def number_input(at, label):
    return widget(at.number_input, label)


def strike_range(at, low, high):
    widget(at.slider, "Strike range (% of spot)").set_range(low, high).run()


def y_axis(at, option):
    widget(at.selectbox, "Strike axis").set_value(option).run()


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
        assert key_figures(results)['atm_near'] == pytest.approx(20.0, abs=0.05)

    @pytest.mark.integration
    def test_risk_reversal_is_a_number_and_zero_on_a_flat_surface(self):
        at, _ = open_app(make_chain())
        risk_reversal = key_figures(at.session_state['analysis_results'])['rr_near']

        assert risk_reversal is not None
        assert risk_reversal == pytest.approx(0.0, abs=0.05)

    @pytest.mark.integration
    def test_header_line_describes_the_surface(self):
        chain = make_chain()
        at, _ = open_app(chain)
        results = at.session_state['analysis_results']
        expiries = len({o['days_to_expiry'] for o in results['valid_options']})
        header = next(m.value for m in at.markdown if "IV points" in m.value)

        assert header.startswith("**SPY** · spot $100.00")
        assert f"{len(results['ivs']):,} IV points ({results['n_puts']:,} puts / {results['n_calls']:,} calls)" in header
        assert f"across {expiries} expiries" in header

    @pytest.mark.integration
    def test_dividend_override_is_used_and_reported(self):
        at, _ = open_app(make_chain(), [lambda a: a.checkbox[0].check().run()])
        results = at.session_state['analysis_results']

        assert len(at.exception) == 0
        assert results['dividend_source'] == "manual override"
        assert results['dividend_yield_used'] == pytest.approx(0.01)  # the sidebar default, 1%

    @pytest.mark.integration
    def test_the_chart_is_a_single_surface_with_no_smile_slice_lines(self):
        at, _ = open_app(make_chain())
        fig = at.session_state['analysis_results']['fig']

        assert [trace.type for trace in fig.data] == ['surface']
        assert not any("not drawn" in c.value for c in at.caption)


class TestLayout:
    @pytest.mark.integration
    def test_four_key_figures(self):
        at, _ = open_app(make_chain())

        assert [m.label for m in at.metric] == ["30d ATM IV", "30d Implied Move (±$5.73)", "30d Risk Reversal (25Δ)",
                                               "Term Structure (90d - 30d)"]
        assert at.metric[0].value == "20.0%"
        assert at.metric[1].value == "±5.7%"  # 20% × sqrt(30/365) of a $100 spot is $5.73
        assert [m.value for m in at.metric[2:]] == ["+0.0 pts", "+0.0 pts"]  # a flat surface has no skew or slope

    @pytest.mark.integration
    def test_tabs(self):
        at, _ = open_app(make_chain())

        assert [t.label for t in at.tabs] == ["Surface", "Smile & Skew", "Data & Export"]

    @pytest.mark.integration
    def test_table_has_one_row_per_point(self):
        at, _ = open_app(make_chain())
        table = at.dataframe[0].value
        results = at.session_state['analysis_results']

        assert list(table.columns) == ['Strike', 'Days_to_Expiry', 'Option_Type', 'Price', 'Price_Source',
                                       'Open_Interest', 'IV', 'Moneyness']
        assert len(table) == len(results['ivs'])
        assert table['IV'].between(19.99, 20.01).all()
        assert set(table['Option_Type']) == {'call', 'put'}

    @pytest.mark.integration
    def test_surface_always_uses_the_viridis_colour_scale(self):
        at, _ = open_app(make_chain())
        surface = at.session_state['analysis_results']['fig'].data[0]

        assert surface.type == 'surface'
        assert surface.colorscale[0][1] == '#440154'  # the dark purple at the bottom of Viridis


class TestSmileTab:
    @pytest.mark.integration
    def test_defaults_to_the_expiry_nearest_30_days(self):
        at, _ = open_app(make_chain(days=(14, 45, 90, 180)))

        assert widget(at.selectbox, "Expiry").value == 45

    @pytest.mark.integration
    def test_the_svi_fit_is_shown_even_when_the_surface_is_interpolated(self):
        at, _ = open_app(make_chain())

        assert widget(at.radio, "Surface method").value == "Interpolated"
        assert any("The white line is an SVI fit" in c.value for c in at.caption)

    @pytest.mark.integration
    def test_another_expiry_can_be_picked(self):
        at, _ = open_app(make_chain(), [lambda a: widget(a.selectbox, "Expiry").set_value(90).run()])

        assert len(at.exception) == 0
        assert widget(at.selectbox, "Expiry").value == 90


class TestHeatmapView:
    @pytest.mark.integration
    def test_the_surface_can_be_switched_to_a_heatmap(self):
        at, _ = open_app(make_chain(), [lambda a: widget(a.radio, "View").set_value("Heatmap").run()])

        assert list(widget(at.radio, "View").options) == ["3D surface", "Heatmap"]
        assert widget(at.radio, "View").value == "Heatmap"
        assert len(at.exception) == 0


class TestDataPoints:
    @pytest.mark.integration
    def test_data_points_can_be_shown_on_the_heatmap(self):
        at, _ = open_app(make_chain(), [lambda a: widget(a.radio, "View").set_value("Heatmap").run(),
                                        lambda a: a.toggle[0].set_value(True).run()])

        assert at.toggle[0].value is True
        assert len(at.exception) == 0
        assert any("Dots are real options" in c.value for c in at.caption)


class TestSviSurface:
    @pytest.mark.integration
    def test_svi_fit_can_be_chosen(self):
        at, _ = open_app(make_chain(), [lambda a: widget(a.radio, "Surface method").set_value("SVI fit").run()])
        svi = at.session_state['analysis_results']['svi']

        assert len(at.exception) == 0
        assert len(svi['slices']) >= 2 and svi['mesh'] is not None
        assert any(c.value.startswith("SVI fit:") for c in at.caption)

    @pytest.mark.integration
    def test_svi_is_off_by_default(self):
        at, _ = open_app(make_chain())

        assert widget(at.radio, "Surface method").value == "Interpolated"
        assert 'svi' not in at.session_state['analysis_results']


class TestMoneynessAxis:
    @pytest.mark.integration
    def test_moneyness_is_the_log_of_strike_over_forward(self):
        at, _ = open_app(make_chain(), [lambda a: y_axis(a, "Moneyness")])
        results = at.session_state['analysis_results']
        surface_y = np.asarray(results['fig'].data[0].y, dtype=float)
        forward_line = build_heatmap_figure(results, "Moneyness").data[-1]

        assert len(at.exception) == 0
        assert np.nanmax(np.abs(surface_y)) < 1.0  # logs of strike over forward are small; dollar strikes are about 100
        assert np.allclose(forward_line.x, 0.0)    # the forward is at zero on every expiry


class TestSidebar:
    @pytest.mark.integration
    def test_there_is_no_colour_or_theme_choice(self):
        at, _ = open_app(make_chain())

        assert [s.label for s in at.selectbox if s.label != "Expiry"] == ["Strike axis"]

    @pytest.mark.integration
    def test_strike_range_is_one_slider_with_the_default_window(self):
        at, _ = open_app(make_chain())
        slider = widget(at.slider, "Strike range (% of spot)")

        assert slider.value == (75, 125)
        assert (slider.min, slider.max) == (UIConfig.MIN_STRIKE_PCT_LOWER, UIConfig.MAX_STRIKE_PCT_UPPER)

    @pytest.mark.integration
    def test_rate_and_dividend_controls_are_under_advanced(self):
        at, _ = open_app(make_chain())

        assert [e.label for e in at.expander] == ["Advanced settings"]
        assert widget(at.expander[0].number_input, "Risk-free rate (% per year)").value == pytest.approx(4.2)
        assert widget(at.expander[0].number_input, "Dividend yield (% per year)").value == pytest.approx(1.0)

    @pytest.mark.integration
    def test_open_interest_defaults_to_zero(self):
        """Yahoo often reports no open interest before the US open, so the default must not filter."""
        at, _ = open_app(make_chain())

        assert number_input(at, "Minimum open interest (contracts)").value == 0

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
            lambda a: y_axis(a, UIConfig.AVAILABLE_Y_AXIS_TYPES[-1]),
            lambda a: number_input(a, "Risk-free rate (% per year)").set_value(3.0).run(),
            lambda a: strike_range(a, 90, 110),
        ])

        assert fetch.call_count == 1
        assert len(at.exception) == 0 and 'analysis_results' in at.session_state

    @pytest.mark.integration
    def test_open_interest_filters_the_cached_chain(self):
        chain = make_chain(open_interest={30: 0, 60: 0, 90: 100, 180: 100})
        at, fetch = open_app(chain)
        before = point_count(at)

        number_input(at, "Minimum open interest (contracts)").set_value(50).run()

        assert before == expected_otm_count(chain)
        assert point_count(at) == expected_otm_count(chain[chain['open_interest'] >= 50])
        assert point_count(at) < before
        assert fetch.call_count == 1

    @pytest.mark.integration
    def test_strike_range_filters_the_cached_chain(self):
        chain = make_chain()
        at, fetch = open_app(chain)

        strike_range(at, 95, 125)

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
            lambda a: y_axis(a, UIConfig.AVAILABLE_Y_AXIS_TYPES[-1]),
            lambda a: number_input(a, "Minimum open interest (contracts)").set_value(5).run(),
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

    @staticmethod
    def side(pricer, strikes, maturity):
        return pd.DataFrame(dict(
            strike=strikes, bid=0.0, ask=0.0,
            lastPrice=[pricer(SPOT, K, maturity, RATE, VOL, DIVIDEND) for K in strikes],
            volume=0.0, openInterest=0.0))

    def option_chain(self, date):
        T = (pd.Timestamp(date) - self.today).days / 365
        strikes = np.arange(80.0, 121.0, 2.5)

        chain = type('Chain', (), {})()
        chain.calls = self.side(bs_call_price, strikes, T)
        chain.puts = self.side(bs_put_price, strikes, T)
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