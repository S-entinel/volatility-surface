"""
Shared pytest fixtures for volatility surface analyser tests.

Provides reusable test data and configurations across all test modules.
"""

import pytest
import numpy as np
from typing import Dict, Any

from src.calculators.black_scholes import OptionData, BlackScholes
from src.calculators.implied_volatility import IVCalculator
from src.config.config import ModelConfig


# ============================================================================
# Basic Test Data Fixtures
# ============================================================================

@pytest.fixture
def sample_spot_price() -> float:
    """Standard spot price for testing."""
    return 100.0


@pytest.fixture
def sample_strike_price() -> float:
    """Standard strike price for testing."""
    return 100.0


@pytest.fixture
def sample_time_to_expiry() -> float:
    """Standard time to expiration (1 year)."""
    return 1.0


@pytest.fixture
def sample_risk_free_rate() -> float:
    """Standard risk-free rate from config."""
    return ModelConfig.DEFAULT_RISK_FREE_RATE


@pytest.fixture
def sample_dividend_yield() -> float:
    """Standard dividend yield from config."""
    return ModelConfig.DEFAULT_DIVIDEND_YIELD


@pytest.fixture
def sample_volatility() -> float:
    """Standard volatility (20%)."""
    return 0.20


# ============================================================================
# OptionData Fixtures
# ============================================================================

@pytest.fixture
def atm_call_option(sample_spot_price: float, sample_strike_price: float,
                    sample_time_to_expiry: float, sample_risk_free_rate: float,
                    sample_volatility: float) -> OptionData:
    """At-the-money call option."""
    return OptionData(
        S=sample_spot_price,
        K=sample_strike_price,
        T=sample_time_to_expiry,
        r=sample_risk_free_rate,
        sigma=sample_volatility,
        q=0.0,
        option_type='call'
    )


@pytest.fixture
def atm_put_option(sample_spot_price: float, sample_strike_price: float,
                   sample_time_to_expiry: float, sample_risk_free_rate: float,
                   sample_volatility: float) -> OptionData:
    """At-the-money put option."""
    return OptionData(
        S=sample_spot_price,
        K=sample_strike_price,
        T=sample_time_to_expiry,
        r=sample_risk_free_rate,
        sigma=sample_volatility,
        q=0.0,
        option_type='put'
    )


@pytest.fixture
def itm_call_option(sample_spot_price: float, sample_time_to_expiry: float,
                    sample_risk_free_rate: float, sample_volatility: float) -> OptionData:
    """In-the-money call option (K=90)."""
    return OptionData(
        S=sample_spot_price,
        K=90.0,
        T=sample_time_to_expiry,
        r=sample_risk_free_rate,
        sigma=sample_volatility,
        q=0.0,
        option_type='call'
    )


@pytest.fixture
def otm_call_option(sample_spot_price: float, sample_time_to_expiry: float,
                    sample_risk_free_rate: float, sample_volatility: float) -> OptionData:
    """Out-of-the-money call option (K=110)."""
    return OptionData(
        S=sample_spot_price,
        K=110.0,
        T=sample_time_to_expiry,
        r=sample_risk_free_rate,
        sigma=sample_volatility,
        q=0.0,
        option_type='call'
    )


@pytest.fixture
def deep_itm_call_option(sample_spot_price: float, sample_time_to_expiry: float,
                         sample_risk_free_rate: float, sample_volatility: float) -> OptionData:
    """Deep in-the-money call option (K=50)."""
    return OptionData(
        S=sample_spot_price,
        K=50.0,
        T=sample_time_to_expiry,
        r=sample_risk_free_rate,
        sigma=sample_volatility,
        q=0.0,
        option_type='call'
    )


@pytest.fixture
def deep_otm_call_option(sample_spot_price: float, sample_time_to_expiry: float,
                         sample_risk_free_rate: float, sample_volatility: float) -> OptionData:
    """Deep out-of-the-money call option (K=150)."""
    return OptionData(
        S=sample_spot_price,
        K=150.0,
        T=sample_time_to_expiry,
        r=sample_risk_free_rate,
        sigma=sample_volatility,
        q=0.0,
        option_type='call'
    )


@pytest.fixture
def short_maturity_option(sample_spot_price: float, sample_strike_price: float,
                          sample_risk_free_rate: float, sample_volatility: float) -> OptionData:
    """Option with very short maturity (1 week)."""
    return OptionData(
        S=sample_spot_price,
        K=sample_strike_price,
        T=7/365,  # 1 week
        r=sample_risk_free_rate,
        sigma=sample_volatility,
        q=0.0,
        option_type='call'
    )


@pytest.fixture
def long_maturity_option(sample_spot_price: float, sample_strike_price: float,
                         sample_risk_free_rate: float, sample_volatility: float) -> OptionData:
    """Option with long maturity (2 years)."""
    return OptionData(
        S=sample_spot_price,
        K=sample_strike_price,
        T=2.0,
        r=sample_risk_free_rate,
        sigma=sample_volatility,
        q=0.0,
        option_type='call'
    )


@pytest.fixture
def high_volatility_option(sample_spot_price: float, sample_strike_price: float,
                           sample_time_to_expiry: float, sample_risk_free_rate: float) -> OptionData:
    """Option with high volatility (80%)."""
    return OptionData(
        S=sample_spot_price,
        K=sample_strike_price,
        T=sample_time_to_expiry,
        r=sample_risk_free_rate,
        sigma=0.80,
        q=0.0,
        option_type='call'
    )


@pytest.fixture
def low_volatility_option(sample_spot_price: float, sample_strike_price: float,
                          sample_time_to_expiry: float, sample_risk_free_rate: float) -> OptionData:
    """Option with low volatility (5%)."""
    return OptionData(
        S=sample_spot_price,
        K=sample_strike_price,
        T=sample_time_to_expiry,
        r=sample_risk_free_rate,
        sigma=0.05,
        q=0.0,
        option_type='call'
    )


# ============================================================================
# Calculator Fixtures
# ============================================================================

@pytest.fixture
def iv_calculator() -> IVCalculator:
    """Fresh IV calculator instance."""
    return IVCalculator()


@pytest.fixture
def black_scholes() -> BlackScholes:
    """Black-Scholes calculator instance."""
    return BlackScholes()


# ============================================================================
# Surface Data Fixtures
# ============================================================================

@pytest.fixture
def sample_surface_data() -> Dict[str, Any]:
    """Sample volatility surface data."""
    n_strikes = 10
    n_expiries = 5
    
    strikes = np.linspace(80, 120, n_strikes)
    expiries = np.linspace(0.1, 2.0, n_expiries)
    
    # Create meshgrid
    strike_mesh, expiry_mesh = np.meshgrid(strikes, expiries)
    
    # Generate sample IV surface (with smile shape)
    ivs = []
    for expiry in expiries:
        for strike in strikes:
            moneyness = strike / 100.0
            # IV smile: higher vol for OTM options
            base_vol = 0.20
            smile_effect = 0.05 * (moneyness - 1.0) ** 2
            term_effect = 0.02 * np.sqrt(expiry)
            iv = base_vol + smile_effect - term_effect
            ivs.append(max(iv, 0.05))  # Floor at 5%
    
    return {
        'strikes': strike_mesh.flatten(),
        'expiries': expiry_mesh.flatten(),
        'ivs': np.array(ivs),
        'spot_price': 100.0
    }


# ============================================================================
# Validation Test Data
# ============================================================================

@pytest.fixture
def invalid_option_params() -> Dict[str, Dict[str, Any]]:
    """Collection of invalid option parameters for validation testing."""
    return {
        'negative_spot': {'S': -100.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'market_price': 5.0},
        'zero_spot': {'S': 0.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'market_price': 5.0},
        'negative_strike': {'S': 100.0, 'K': -100.0, 'T': 1.0, 'r': 0.05, 'market_price': 5.0},
        'zero_strike': {'S': 100.0, 'K': 0.0, 'T': 1.0, 'r': 0.05, 'market_price': 5.0},
        'negative_time': {'S': 100.0, 'K': 100.0, 'T': -1.0, 'r': 0.05, 'market_price': 5.0},
        'zero_time': {'S': 100.0, 'K': 100.0, 'T': 0.0, 'r': 0.05, 'market_price': 5.0},
        'negative_price': {'S': 100.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'market_price': -5.0},
        'zero_price': {'S': 100.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'market_price': 0.0},
        'invalid_rate': {'S': 100.0, 'K': 100.0, 'T': 1.0, 'r': 2.0, 'market_price': 5.0},
        'invalid_option_type': {'S': 100.0, 'K': 100.0, 'T': 1.0, 'r': 0.05, 'market_price': 5.0, 'option_type': 'invalid'},
    }


# ============================================================================
# Tolerance Fixtures
# ============================================================================

@pytest.fixture
def price_tolerance() -> float:
    """Acceptable tolerance for price comparisons."""
    return 0.01  # $0.01


@pytest.fixture
def iv_tolerance() -> float:
    """Acceptable tolerance for IV comparisons."""
    return 0.0001  # 0.01%