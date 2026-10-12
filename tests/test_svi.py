"""Tests of the SVI maths: parameter recovery and the two arbitrage checks."""

import numpy as np
import pytest

from src.svi import SviSlice, butterfly_density, calendar_check, fit_slice, total_variance


def slice_with(params, days=30) -> SviSlice:
    return SviSlice(days, days / 365, 100.0, params, 50, 0.0, -0.2, 0.2, 0.5)


class TestFit:
    @pytest.mark.unit
    def test_recovers_known_parameters_from_noisy_quotes(self):
        true, maturity = (0.02, 0.15, -0.6, 0.03, 0.12), 0.25
        rng = np.random.default_rng(1)
        k = np.sort(rng.uniform(-0.25, 0.2, 80))
        iv = np.sqrt(total_variance(k, true) / maturity) + rng.normal(0, 0.0005, 80)

        fitted = fit_slice(k, iv, maturity, days=91)

        grid = np.linspace(-0.25, 0.2, 100)
        curve_error = np.abs(np.sqrt(total_variance(grid, fitted.params) / maturity)
                             - np.sqrt(total_variance(grid, true) / maturity))
        assert curve_error.max() < 0.002          # within 0.2 vol points everywhere
        assert fitted.butterfly_ok

    @pytest.mark.unit
    def test_too_few_points_are_not_fitted(self):
        assert fit_slice(np.linspace(-0.1, 0.1, 5), np.full(5, 0.2), 0.25) is None


class TestArbitrageChecks:
    @pytest.mark.unit
    def test_butterfly_density_is_negative_for_a_kinked_wing(self):
        assert butterfly_density(np.linspace(-0.3, 0.3, 200), (0.0, 2.0, 0.95, 0.0, 0.01)).min() < 0

    @pytest.mark.unit
    def test_calendar_check_flags_variance_that_falls_with_time(self):
        crossing = calendar_check([slice_with((0.02, 0.1, -0.5, 0.0, 0.1), 30),
                                   slice_with((0.005, 0.1, -0.5, 0.0, 0.1), 60)])
        ordered = calendar_check([slice_with((0.02, 0.1, -0.5, 0.0, 0.1), 30),
                                  slice_with((0.05, 0.1, -0.5, 0.0, 0.1), 60)])

        assert crossing['violations'] == 1 and crossing['pairs'] == [(30, 60)]
        assert ordered['violations'] == 0