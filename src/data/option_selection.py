"""Select out-of-the-money options: puts below the forward price, calls at or above it."""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ('S', 'T', 'strike', 'type')


def forward_price(spot: np.ndarray, risk_free_rate: float, dividend_yield: float,
                  time_to_expiry: np.ndarray) -> np.ndarray:
    """
    Forward price F = S * exp((r - q) * T).

    Args:
        spot: Spot price(s)
        risk_free_rate: Risk-free rate in decimal form
        dividend_yield: Dividend yield in decimal form
        time_to_expiry: Time(s) to expiry in years

    Returns:
        Forward price(s)
    """
    return np.asarray(spot) * np.exp((risk_free_rate - dividend_yield) * np.asarray(time_to_expiry))


def select_otm_options(options_df: pd.DataFrame, risk_free_rate: float,
                       dividend_yield: float) -> pd.DataFrame:
    """
    Keep puts below the forward price and calls at or above it.

    The forward is computed per row from that row's own expiry. A strike exactly at the
    forward is treated as a call.

    Args:
        options_df: Options with columns ``S``, ``T``, ``strike`` and ``type`` ('call'/'put')
        risk_free_rate: Risk-free rate in decimal form
        dividend_yield: Dividend yield in decimal form (use the same value as for the IVs)

    Returns:
        A copy of the matching rows with an extra ``forward`` column

    Raises:
        ValueError: If a required column is missing
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in options_df.columns]
    if missing:
        raise ValueError(f"options_df is missing required columns: {missing}")

    selected = options_df.copy()
    selected['forward'] = forward_price(selected['S'], risk_free_rate, dividend_yield, selected['T'])

    is_otm_put = (selected['type'] == 'put') & (selected['strike'] < selected['forward'])
    is_otm_call = (selected['type'] == 'call') & (selected['strike'] >= selected['forward'])

    return selected[is_otm_put | is_otm_call]