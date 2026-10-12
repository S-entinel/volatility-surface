"""Tests of the cached-chain functions in the analysis module, run with no Streamlit involved."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analysis import AnalysisError, AnalysisParams, analyse_chain, fetch_full_chain, filter_chain, run_analysis
from src.calculators.implied_volatility import bs_call_price, bs_put_price
from src.config.config import MarketDataConfig, ModelConfig, UIConfig

SPOT, RATE, DIVIDEND, VOL = 100.0, 0.015, 0.01, 0.20


def make_chain(days=(30, 60, 90, 180), open_interest=None) -> pd.DataFrame:
    """A call and a put for every strike and expiry, priced at a flat 20% volatility."""
    open_interest = open_interest or {}
    today = pd.Timestamp.now().normalize()
    rows = []
    for d in days:
        T = d / 365
        for K in np.arange(70.0, 131.0, 2.5):
            for option_type, pricer in (('call', bs_call_price), ('put', bs_put_price)):
                rows.append(dict(
                    expiration=today + pd.Timedelta(days=d), strike=float(K),
                    price=pricer(SPOT, K, T, RATE, VOL, DIVIDEND), type=option_type,
                    open_interest=open_interest.get(d, 100),
                    days_to_expiry=d, T=T, S=SPOT, r=RATE, q=DIVIDEND, moneyness=K / SPOT))
    chain = pd.DataFrame(rows)
    return chain[chain['price'] >= MarketDataConfig.MIN_QUOTE_MID_PRICE].reset_index(drop=True)


def make_params(**overrides) -> AnalysisParams:
    settings = dict(
        ticker='TEST', min_strike_pct=75.0, max_strike_pct=125.0, min_open_interest=0,
        risk_free_rate=RATE * 100, override_dividend=False, dividend_yield=1.3,
        theme=UIConfig.AVAILABLE_THEMES[0], colormap=UIConfig.AVAILABLE_COLORMAPS[0],
        y_axis_type=UIConfig.AVAILABLE_Y_AXIS_TYPES[0])
    settings.update(overrides)
    return AnalysisParams(**settings)


class TestFilterChain:
    @pytest.mark.unit
    def test_strike_range_is_a_percentage_of_spot_and_inclusive(self):
        kept = filter_chain(make_chain(), make_params(min_strike_pct=90.0, max_strike_pct=110.0))

        assert kept['strike'].min() == 90.0 and kept['strike'].max() == 110.0

    @pytest.mark.unit
    def test_minimum_open_interest(self):
        chain = make_chain(open_interest={30: 5, 60: 10, 90: 50, 180: 500})

        assert set(filter_chain(chain, make_params(min_open_interest=10))['days_to_expiry']) == {60, 90, 180}
        assert set(filter_chain(chain, make_params(min_open_interest=51))['days_to_expiry']) == {180}

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_missing_open_interest_counts_as_zero(self):
        chain = make_chain()
        chain.loc[chain.index[:3], 'open_interest'] = np.nan

        everything = dict(min_strike_pct=50.0, max_strike_pct=200.0)

        assert len(filter_chain(chain, make_params(min_open_interest=0, **everything))) == len(chain)
        assert len(filter_chain(chain, make_params(min_open_interest=1, **everything))) == len(chain) - 3

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_chain_without_an_open_interest_column(self):
        chain = make_chain().drop(columns='open_interest')

        assert len(filter_chain(chain, make_params(min_open_interest=0))) == len(chain[chain['strike'].between(75, 125)])
        assert filter_chain(chain, make_params(min_open_interest=1)).empty

    @pytest.mark.unit
    def test_does_not_modify_the_input(self):
        chain = make_chain()
        before = chain.copy()
        filter_chain(chain, make_params(min_strike_pct=90.0, min_open_interest=10))

        pd.testing.assert_frame_equal(chain, before)

    @pytest.mark.unit
    def test_can_return_nothing(self):
        assert filter_chain(make_chain(), make_params(min_strike_pct=99.0, max_strike_pct=99.5)).empty


class TestFetchFullChain:
    @pytest.mark.unit
    def test_asks_for_the_widest_window_and_no_open_interest_limit(self):
        with patch('src.analysis.OptionDataFetcher') as fetcher:
            fetcher.return_value.prepare_for_iv.return_value = make_chain()
            fetch_full_chain('SPY')

        fetcher.assert_called_once_with('SPY')
        assert fetcher.return_value.prepare_for_iv.call_args.kwargs == dict(
            min_strike_pct=UIConfig.MIN_STRIKE_PCT_LOWER, max_strike_pct=UIConfig.MAX_STRIKE_PCT_UPPER,
            min_open_interest=0, risk_free_rate=ModelConfig.DEFAULT_RISK_FREE_RATE)

    @pytest.mark.unit
    def test_empty_chain_is_an_error(self):
        with patch('src.analysis.OptionDataFetcher') as fetcher:
            fetcher.return_value.prepare_for_iv.return_value = make_chain().iloc[0:0]
            with pytest.raises(AnalysisError, match="No valid options data found"):
                fetch_full_chain('SPY')


class TestAnalyseChain:
    @pytest.mark.unit
    def test_filtering_a_wide_chain_matches_a_pre_filtered_chain(self):
        """Filtering inside analyse_chain gives the same surface as fetching a narrower chain."""
        wide = make_chain(open_interest={30: 0, 60: 0, 90: 100, 180: 100})
        params = make_params(min_strike_pct=85.0, max_strike_pct=115.0, min_open_interest=50)

        from_wide = analyse_chain(wide, params)
        from_narrow = analyse_chain(filter_chain(wide, params), params)

        assert from_wide['ivs'] == from_narrow['ivs']
        assert from_wide['iv_stats'] == from_narrow['iv_stats']
        assert (from_wide['n_puts'], from_wide['n_calls']) == (from_narrow['n_puts'], from_narrow['n_calls'])

    @pytest.mark.unit
    def test_narrower_settings_use_fewer_points(self):
        chain = make_chain()
        full = analyse_chain(chain, make_params())
        narrow = analyse_chain(chain, make_params(min_strike_pct=95.0, max_strike_pct=105.0))

        assert len(narrow['ivs']) < len(full['ivs'])

    @pytest.mark.unit
    def test_nothing_left_after_filtering(self):
        with pytest.raises(AnalysisError, match="No valid options data found"):
            analyse_chain(make_chain(), make_params(min_open_interest=10_000))

    @pytest.mark.unit
    def test_too_few_options_after_filtering(self):
        with pytest.raises(AnalysisError, match="Insufficient options data"):
            analyse_chain(make_chain(), make_params(min_strike_pct=99.0, max_strike_pct=101.0))

    @pytest.mark.unit
    def test_status_messages_exclude_the_fetch_message(self):
        messages = []
        analyse_chain(make_chain(), make_params(), on_status=lambda kind, text: messages.append(kind))

        assert messages == ['success', 'info']

    @pytest.mark.unit
    def test_run_analysis_still_reports_the_fetch_message_first(self):
        messages = []
        with patch('src.data.market_data.OptionDataFetcher.prepare_for_iv', return_value=make_chain()):
            run_analysis(make_params(), on_status=lambda kind, text: messages.append((kind, text)))

        assert messages[0] == ('info', "Fetching market data...")
        assert [kind for kind, _ in messages] == ['info', 'success', 'info']