"""Tests of the analysis module, run directly with no Streamlit involved."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    AnalysisError, AnalysisParams, analyse_chain, build_surface_figure, calculate_ivs, key_figures, smile_stats
)
from src.calculators.implied_volatility import bs_call_price, bs_put_price
from src.config.config import MarketDataConfig, UIConfig

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


def make_params(**overrides) -> AnalysisParams:
    settings = dict(
        ticker='TEST', min_strike_pct=75.0, max_strike_pct=125.0, min_open_interest=0,
        risk_free_rate=RATE * 100, override_dividend=False, dividend_yield=1.3,
        theme=UIConfig.AVAILABLE_THEMES[0], colormap='Viridis',
        y_axis_type=UIConfig.AVAILABLE_Y_AXIS_TYPES[0])
    settings.update(overrides)
    return AnalysisParams(**settings)


def analyse(chain, **kwargs):
    return analyse_chain(chain, make_params(), **kwargs)


class TestAnalysisParams:
    @pytest.mark.unit
    def test_rates_are_converted_to_decimals(self):
        assert make_params(risk_free_rate=4.5).risk_free_decimal == pytest.approx(0.045)

    @pytest.mark.unit
    def test_dividend_override_only_when_ticked(self):
        assert make_params(override_dividend=False, dividend_yield=2.0).dividend_override is None
        assert make_params(override_dividend=True, dividend_yield=2.0).dividend_override == pytest.approx(0.02)


class TestAnalyseChain:
    @pytest.mark.unit
    def test_uses_only_out_of_the_money_options(self):
        chain = make_chain()
        results = analyse(chain)

        forward = SPOT * np.exp((RATE - DIVIDEND) * chain['T'])
        expected = int((((chain['type'] == 'put') & (chain['strike'] < forward)) |
                        ((chain['type'] == 'call') & (chain['strike'] >= forward))).sum())
        assert len(results['valid_options']) == expected
        assert results['n_puts'] > 0 and results['n_calls'] > 0
        assert results['n_puts'] + results['n_calls'] == len(results['ivs'])

    @pytest.mark.unit
    def test_recovers_a_flat_volatility(self):
        results = analyse(make_chain())

        assert np.allclose(results['ivs'], VOL, atol=1e-4)
        assert key_figures(results)['atm_near'] == pytest.approx(20.0, abs=0.05)
        assert results['spot_price'] == SPOT and results['ticker'] == 'TEST'

    @pytest.mark.unit
    def test_result_keys(self):
        assert sorted(analyse(make_chain())) == sorted([
            'fig', 'spot_price', 'ticker', 'ivs', 'valid_options', 'dividend_yield_used',
            'dividend_source', 'n_puts', 'n_calls'])

    @pytest.mark.unit
    def test_dividend_override_is_used(self):
        results = analyse_chain(make_chain(), make_params(override_dividend=True, dividend_yield=3.0))

        assert results['dividend_source'] == "manual override"
        assert results['dividend_yield_used'] == pytest.approx(0.03)

    @pytest.mark.unit
    def test_status_messages_come_in_order(self):
        messages = []
        analyse(make_chain(), on_status=lambda kind, text: messages.append((kind, text)))

        assert [kind for kind, _ in messages] == ['success', 'info']
        assert messages[0][1].startswith("Market data fetched - TEST @ $100.00")
        assert messages[1][1] == "Computing implied volatilities..."

    @pytest.mark.unit
    def test_progress_is_reported_for_every_option(self):
        calls = []
        results = analyse(make_chain(), on_progress=lambda done, total: calls.append((done, total)))

        total = len(results['valid_options'])
        assert calls == [(i, total) for i in range(1, total + 1)]

    @pytest.mark.unit
    def test_works_without_callbacks(self):
        assert len(analyse(make_chain())['ivs']) > 10


class TestAnalyseChainErrors:
    @pytest.mark.unit
    def test_empty_fetch_result(self):
        with pytest.raises(AnalysisError, match="No valid options data found"):
            analyse_chain(make_chain().iloc[0:0], make_params())

    @pytest.mark.unit
    def test_too_few_out_of_the_money_options(self):
        chain = make_chain(days=(30,), strikes=np.array([95.0, 105.0]))
        with pytest.raises(AnalysisError, match="Insufficient options data. Found 2 usable"):
            analyse_chain(chain, make_params())

    @pytest.mark.unit
    def test_unsolvable_prices(self):
        chain = make_chain(days=(30, 60, 90))
        chain['price'] = 1e6
        with pytest.raises(AnalysisError, match="Insufficient valid options"):
            analyse_chain(chain, make_params())


class TestCalculateIvs:
    @pytest.mark.unit
    def test_empty_frame_gives_nothing_and_no_progress(self):
        calls = []
        ivs, options = calculate_ivs(make_chain().iloc[0:0], RATE, on_progress=lambda *a: calls.append(a))

        assert ivs == [] and options == [] and calls == []

    @pytest.mark.unit
    def test_progress_still_reported_when_options_fail(self):
        chain = make_chain(days=(30,))
        chain['price'] = 1e6
        calls = []
        ivs, options = calculate_ivs(chain, RATE, on_progress=lambda done, total: calls.append(done))

        assert ivs == []
        assert calls == list(range(1, len(chain) + 1))

    @pytest.mark.unit
    def test_dividend_override_replaces_the_looked_up_yield(self):
        chain = make_chain(days=(90,))
        default_ivs, default_options = calculate_ivs(chain, RATE)
        override_ivs, override_options = calculate_ivs(chain, RATE, dividend_yield_override=0.03)

        by_contract = {(o['strike'], o['type']): iv for o, iv in zip(default_options, default_ivs)}
        shared = [(iv, by_contract[(o['strike'], o['type'])])
                  for o, iv in zip(override_options, override_ivs) if (o['strike'], o['type']) in by_contract]

        assert np.allclose(default_ivs, VOL, atol=1e-4)
        assert len(shared) > 5
        assert any(abs(new - old) > 1e-3 for new, old in shared)


def flat_options(days, strikes):
    return [{'strike': float(K), 'T': d / 365} for d in days for K in strikes]


class TestBuildSurfaceFigure:
    @staticmethod
    def surface(days=(45, 75, 100, 180), **overrides):
        options = flat_options(days, np.arange(85.0, 116.0, 5.0))
        return build_surface_figure(options, [VOL] * len(options), SPOT, make_params(**overrides))

    @pytest.mark.unit
    def test_figure_has_a_surface(self):
        fig = self.surface()

        assert fig.data[0].type == 'surface'

    @pytest.mark.unit
    def test_moneyness_axis_is_the_log_of_strike_over_forward(self):
        options = [{**option, 'forward': 100.0 * (1 + option['T'])}
                   for option in flat_options((45, 75, 100, 180), np.arange(85.0, 116.0, 5.0))]
        fig = build_surface_figure(options, [VOL] * len(options), SPOT, make_params(y_axis_type="Moneyness"))
        y = np.asarray(fig.data[0].y, dtype=float)
        logs = [np.log(option['strike'] / option['forward']) for option in options]

        assert y.min() == pytest.approx(min(logs)) and y.max() == pytest.approx(max(logs))


class TestSmileStats:
    @pytest.mark.unit
    def test_flat_surface_has_a_flat_atm_and_no_risk_reversal(self):
        stats = smile_stats(analyse(make_chain()), 60)

        assert stats['atm_iv'] == pytest.approx(20.0, abs=0.05)
        assert stats['rr_25'] == pytest.approx(0.0, abs=0.05)