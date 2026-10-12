"""
Tests for out-of-the-money option selection (puts below the forward, calls at or above it).
"""

import numpy as np
import pandas as pd
import pytest

from src.data.option_selection import select_otm_options, forward_price


def _options(strikes, types, S=100.0, T=1.0) -> pd.DataFrame:
    return pd.DataFrame({'S': S, 'T': T, 'strike': strikes, 'type': types})


class TestForwardPrice:
    @pytest.mark.unit
    def test_matches_formula(self):
        assert forward_price(100.0, 0.05, 0.02, 2.0) == pytest.approx(100 * np.exp((0.05 - 0.02) * 2.0))

    @pytest.mark.unit
    def test_no_carry_means_forward_equals_spot(self):
        assert forward_price(100.0, 0.03, 0.03, 1.0) == pytest.approx(100.0)


class TestSelectOtmOptions:
    @pytest.mark.unit
    def test_keeps_puts_below_and_calls_above_forward(self):
        df = _options([90, 95, 105, 110] * 2, ['put'] * 4 + ['call'] * 4)
        kept = select_otm_options(df, 0.0, 0.0)  # forward = 100

        assert set(kept.loc[kept['type'] == 'put', 'strike']) == {90, 95}
        assert set(kept.loc[kept['type'] == 'call', 'strike']) == {105, 110}

    @pytest.mark.unit
    def test_strike_equal_to_forward_is_a_call(self):
        df = _options([100, 100], ['put', 'call'])
        kept = select_otm_options(df, 0.0, 0.0)

        assert list(kept['type']) == ['call']

    @pytest.mark.unit
    def test_forward_uses_rates_and_dividends(self):
        """With r = 5% the forward is about 105.13, so a 105 strike is still below it."""
        df = _options([105, 105], ['put', 'call'], T=1.0)

        with_carry = select_otm_options(df, 0.05, 0.0)
        assert list(with_carry['type']) == ['put']

        no_carry = select_otm_options(df, 0.0, 0.0)
        assert list(no_carry['type']) == ['call']

    @pytest.mark.unit
    def test_forward_is_computed_per_expiry(self):
        df = pd.DataFrame({
            'S': 100.0,
            'T': [0.1, 2.0, 0.1, 2.0],
            'strike': [102.0, 102.0, 102.0, 102.0],
            'type': ['put', 'put', 'call', 'call'],
        })
        kept = select_otm_options(df, 0.05, 0.0)

        # Short expiry: forward is about 100.5, below 102, so the call is OTM.
        # Long expiry: forward is about 110.5, above 102, so the put is OTM.
        assert sorted(zip(kept['T'], kept['type'])) == [(0.1, 'call'), (2.0, 'put')]

    @pytest.mark.unit
    def test_adds_forward_column_and_keeps_others(self):
        df = _options([90, 110], ['put', 'call'])
        df['price'] = [1.0, 2.0]
        kept = select_otm_options(df, 0.0, 0.0)

        assert 'forward' in kept.columns and 'price' in kept.columns
        assert (kept['forward'] == 100.0).all()

    @pytest.mark.unit
    def test_does_not_modify_input(self):
        df = _options([90, 110], ['put', 'call'])
        before = df.copy()
        select_otm_options(df, 0.0, 0.0)

        pd.testing.assert_frame_equal(df, before)

    @pytest.mark.unit
    @pytest.mark.validation
    def test_missing_column_raises(self):
        with pytest.raises(ValueError, match="missing required columns"):
            select_otm_options(pd.DataFrame({'S': [100.0], 'strike': [100.0]}), 0.0, 0.0)

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_empty_frame_returns_empty(self):
        df = pd.DataFrame({'S': [], 'T': [], 'strike': [], 'type': []})
        assert select_otm_options(df, 0.0, 0.0).empty