"""Fetch option data, select out-of-the-money options, solve implied volatilities and build the surface."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from src.data.market_data import OptionDataFetcher
from src.data.option_selection import select_otm_options
from src.calculators.implied_volatility import IVCalculator
from src.visualization.surface_plot import SurfacePlotter, SurfaceData
from src.config.config import MarketDataConfig, ModelConfig, StatisticsConfig, UIConfig

StatusCallback = Callable[[str, str], None]    # (kind: 'info' or 'success', message)
ProgressCallback = Callable[[int, int], None]  # (options done, options total)


class AnalysisError(Exception):
    """A problem to show the user (for example too little data), not a bug."""


@dataclass(frozen=True)
class AnalysisParams:
    """Everything chosen in the sidebar, exactly as the widgets report it (rates in percent)."""
    ticker: str
    min_strike_pct: float
    max_strike_pct: float
    min_open_interest: int
    risk_free_rate: float
    override_dividend: bool
    dividend_yield: float
    theme: str
    colormap: str
    y_axis_type: str
    show_slices: bool = True

    @property
    def risk_free_decimal(self) -> float:
        return self.risk_free_rate / 100

    @property
    def dividend_override(self) -> Optional[float]:
        """The dividend yield to use instead of the looked-up one, as a decimal, or None."""
        return self.dividend_yield / 100 if self.override_dividend else None


def fetch_options(params: AnalysisParams) -> pd.DataFrame:
    """Fetch calls and puts that pass the quote filters; raise AnalysisError if there are none."""
    fetcher = OptionDataFetcher(params.ticker)
    options_df = fetcher.prepare_for_iv(
        min_strike_pct=params.min_strike_pct,
        max_strike_pct=params.max_strike_pct,
        min_open_interest=int(params.min_open_interest),
        risk_free_rate=params.risk_free_decimal
    )
    
    if options_df.empty:
        raise AnalysisError("No valid options data found.")
    
    return options_df


def fetch_full_chain(ticker: str) -> pd.DataFrame:
    """Fetch every usable contract in the widest strike window, with no open interest limit."""
    options_df = OptionDataFetcher(ticker).prepare_for_iv(
        min_strike_pct=UIConfig.MIN_STRIKE_PCT_LOWER,
        max_strike_pct=UIConfig.MAX_STRIKE_PCT_UPPER,
        min_open_interest=MarketDataConfig.MIN_OPEN_INTEREST_THRESHOLD,
        risk_free_rate=ModelConfig.DEFAULT_RISK_FREE_RATE
    )
    
    if options_df.empty:
        raise AnalysisError("No valid options data found.")
    
    return options_df


def filter_chain(chain: pd.DataFrame, params: AnalysisParams) -> pd.DataFrame:
    """Keep the contracts inside the chosen strike range with enough open interest."""
    lowest = chain['S'] * params.min_strike_pct / 100
    highest = chain['S'] * params.max_strike_pct / 100
    open_interest = chain['open_interest'].fillna(0) if 'open_interest' in chain else pd.Series(0, index=chain.index)
    
    keep = (chain['strike'] >= lowest) & (chain['strike'] <= highest) & \
           (open_interest >= int(params.min_open_interest))
    return chain[keep]


def calculate_ivs(options_df: pd.DataFrame, risk_free_rate: float,
                 dividend_yield_override: Optional[float] = None,
                 on_progress: Optional[ProgressCallback] = None) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Calculate IVs for every option in the DataFrame.
    
    Args:
        options_df: DataFrame containing option data
        risk_free_rate: Risk-free rate in decimal form
        dividend_yield_override: Dividend yield in decimal form to use for every option.
            If None (the default), each option's own ``q`` column is used, which holds the
            yield looked up for the ticker.
        on_progress: Optional callback, called as ``on_progress(done, total)`` after each option
        
    Returns:
        Tuple of (list of IVs, list of valid option dictionaries)
        
    Example:
        >>> ivs, valid_options = calculate_ivs(df, 0.045)          # use fetched dividend yield
        >>> ivs, valid_options = calculate_ivs(df, 0.045, 0.013)   # manual override
    """
    iv_calc = IVCalculator()
    ivs = []
    valid_options = []
    total_options = len(options_df)
    
    # Handle edge case
    if total_options == 0:
        return ivs, valid_options
    
    for idx, (_, row) in enumerate(options_df.iterrows()):
        try:
            iv = iv_calc.calculate_iv(
                S=row['S'],
                K=row['strike'],
                T=row['T'],
                r=risk_free_rate,  # Already in decimal form
                q=dividend_yield_override if dividend_yield_override is not None else row['q'],
                market_price=row['price'],
                option_type=row['type']
            )
            
            if iv is not None and np.isfinite(iv):  # Added NaN/Inf check
                ivs.append(iv)
                valid_options.append(row)
        except Exception:
            pass
        
        if on_progress is not None:
            on_progress(idx + 1, total_options)
    
    return ivs, valid_options


def calculate_iv_statistics(valid_options: List[Dict[str, Any]], ivs: List[float], 
                           spot_price: float) -> Dict[str, Optional[float]]:
    """
    Calculate key IV statistics for quant research.
    
    Args:
        valid_options: List of valid option dictionaries
        ivs: List of implied volatilities
        spot_price: Current spot price
        
    Returns:
        Dictionary containing:
            - atm_iv: At-the-money implied volatility (%)
            - iv_skew: IV skew (OTM put - OTM call, %)
            - term_structure: Term structure (long - short ATM IV, %)
            - iv_min: Minimum IV across surface (%)
            - iv_max: Maximum IV across surface (%)
            - avg_iv: Average IV across surface (%)
            - iv_std: Standard deviation of IV (%)
            
    Example:
        >>> stats = calculate_iv_statistics(valid_opts, ivs, 100.0)
        >>> print(f"ATM IV: {stats['atm_iv']:.1f}%")
    """
    if len(valid_options) == 0 or len(ivs) == 0:
        return {
            'atm_iv': 0.0,
            'iv_skew': None,
            'term_structure': 0.0,
            'iv_min': 0.0,
            'iv_max': 0.0,
            'avg_iv': 0.0,
            'iv_std': 0.0
        }
    
    df = pd.DataFrame({
        'strike': [opt['strike'] for opt in valid_options],
        'T': [opt['T'] for opt in valid_options],
        'iv': ivs,
        'moneyness': [opt['strike']/spot_price for opt in valid_options]
    })
    
    # ATM IV (using closest to 1.0 moneyness)
    atm_idx = abs(df['moneyness'] - 1.0).idxmin()
    atm_iv = df.loc[atm_idx, 'iv']
    
    # IV Skew - using config values
    df_nearest = df[df['T'] == df['T'].min()]
    if len(df_nearest) > 0:
        otm_puts = df_nearest[df_nearest['moneyness'] < StatisticsConfig.OTM_PUT_MONEYNESS]['iv']
        otm_calls = df_nearest[df_nearest['moneyness'] > StatisticsConfig.OTM_CALL_MONEYNESS]['iv']
        
        otm_put = otm_puts.mean() if len(otm_puts) > 0 else None
        otm_call = otm_calls.mean() if len(otm_calls) > 0 else None
        
        skew = otm_put - otm_call if (otm_put is not None and otm_call is not None) else None
    else:
        skew = None
    
    # Term Structure
    df['atm_dist'] = abs(df['moneyness'] - 1.0)
    short_term = df[df['T'] == df['T'].min()]
    long_term = df[df['T'] == df['T'].max()]
    
    if len(short_term) > 0 and len(long_term) > 0:
        short_atm = short_term.loc[short_term['atm_dist'].idxmin(), 'iv']
        long_atm = long_term.loc[long_term['atm_dist'].idxmin(), 'iv']
        term_structure = long_atm - short_atm
    else:
        term_structure = 0.0
    
    # Surface statistics
    iv_array = np.array(ivs)
    iv_min = np.min(iv_array)
    iv_max = np.max(iv_array)
    avg_iv = np.mean(iv_array)
    iv_std = np.std(iv_array)
    
    return {
        'atm_iv': atm_iv * StatisticsConfig.IV_DISPLAY_MULTIPLIER,
        'iv_skew': skew * StatisticsConfig.IV_DISPLAY_MULTIPLIER if skew is not None else None,
        'term_structure': term_structure * StatisticsConfig.IV_DISPLAY_MULTIPLIER,
        'iv_min': iv_min * StatisticsConfig.IV_DISPLAY_MULTIPLIER,
        'iv_max': iv_max * StatisticsConfig.IV_DISPLAY_MULTIPLIER,
        'avg_iv': avg_iv * StatisticsConfig.IV_DISPLAY_MULTIPLIER,
        'iv_std': iv_std * StatisticsConfig.IV_DISPLAY_MULTIPLIER
    }


def build_surface_figure(valid_options: List[Any], ivs: List[float], spot_price: float,
                         params: AnalysisParams) -> Tuple[go.Figure, List[int]]:
    """Build the 3D surface with smile slices; also return the slices that could not be drawn."""
    strikes = np.array([opt['strike']/spot_price for opt in valid_options]) if params.y_axis_type == "Moneyness" \
             else np.array([opt['strike'] for opt in valid_options])
    
    surface_data = SurfaceData(
        strikes=strikes,
        expiries=np.array([opt['T'] for opt in valid_options]),
        ivs=np.array(ivs),
        spot_price=spot_price,
        y_axis_type=params.y_axis_type.split()[0]
    )
    
    plotter = SurfacePlotter(surface_data)
    # Pass ticker to create_surface_plot
    fig = plotter.create_surface_plot(theme=params.theme.lower(), colormap=params.colormap,
                                      ticker=params.ticker)
    if params.show_slices:
        fig = plotter.add_smile_slices(fig, theme=params.theme.lower())
    return fig, list(plotter.skipped_slice_days)


def expiry_labels(results: Dict[str, Any]) -> Dict[int, str]:
    """The expiries in the results as {days to expiry: label}, shortest first."""
    labels = {}
    for opt in results['valid_options']:
        days = int(opt['days_to_expiry'])
        if days not in labels:
            expiry = opt.get('expiration')
            labels[days] = f"{days} days ({pd.Timestamp(expiry):%d %b %Y})" if expiry is not None else f"{days} days"
    return dict(sorted(labels.items()))


def _expiry_points(results: Dict[str, Any], expiry_days: int) -> Tuple[List[Any], List[float]]:
    """The options and implied volatilities (decimals) for one expiry."""
    pairs = [(opt, iv) for opt, iv in zip(results['valid_options'], results['ivs'])
             if int(opt['days_to_expiry']) == expiry_days]
    return [opt for opt, _ in pairs], [iv for _, iv in pairs]


def smile_stats(results: Dict[str, Any], expiry_days: int) -> Dict[str, Any]:
    """ATM volatility and skew for a single expiry, defined as in the key figures."""
    options, ivs = _expiry_points(results, expiry_days)
    return calculate_iv_statistics(options, ivs, results['spot_price'])


def build_smile_figure(results: Dict[str, Any], expiry_days: int, y_axis_type: str) -> go.Figure:
    """Implied volatility against strike for one expiry, with puts and calls in different colours."""
    options, ivs = _expiry_points(results, expiry_days)
    spot = results['spot_price']
    use_moneyness = y_axis_type == "Moneyness"
    font = 'IBM Plex Mono, monospace'
    grid = 'rgb(38, 38, 38)'
    multiplier = StatisticsConfig.IV_DISPLAY_MULTIPLIER
    
    def x_value(opt) -> float:
        return float(opt['strike']) / spot if use_moneyness else float(opt['strike'])
    
    ordered = sorted(zip(options, ivs), key=lambda pair: x_value(pair[0]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[x_value(opt) for opt, _ in ordered], y=[iv * multiplier for _, iv in ordered],
        mode='lines', line=dict(color='rgb(70, 70, 70)', width=1), hoverinfo='skip', showlegend=False
    ))
    
    for option_type, name, colour in (('put', 'Puts', '#35b779'), ('call', 'Calls', '#fde725')):
        for source, suffix, symbol in (('mid', '', 'circle'), ('last', ' (last price)', 'circle-open')):
            points = [(opt, iv) for opt, iv in ordered
                      if opt['type'] == option_type and (opt.get('price_source') or 'mid') == source]
            if not points:
                continue
            fig.add_trace(go.Scatter(
                x=[x_value(opt) for opt, _ in points],
                y=[iv * multiplier for _, iv in points],
                mode='markers',
                name=name + suffix,
                marker=dict(color=colour, size=9, symbol=symbol, line=dict(width=2, color=colour)),
                customdata=[[float(opt['strike']), float(opt['price']), opt.get('open_interest')] for opt, _ in points],
                hovertemplate=(
                    f"<b>{name}</b><br>Strike: $%{{customdata[0]:,.2f}}<br>"
                    "Implied volatility: %{y:.1f}%<br>Price: $%{customdata[1]:.2f}"
                    "<br>Open interest: %{customdata[2]}<extra></extra>"
                )
            ))
    
    if options and 'forward' in options[0]:
        forward = float(options[0]['forward'])
        fig.add_vline(x=forward / spot if use_moneyness else forward, line_dash='dot',
                      line_color='rgb(120, 120, 120)', annotation_text="forward",
                      annotation_font=dict(color='rgb(160, 160, 160)', size=11, family=font))
    
    fig.update_layout(
        paper_bgcolor='rgb(0, 0, 0)', plot_bgcolor='rgb(0, 0, 0)',
        font=dict(family=font, color='white'),
        xaxis=dict(title='Moneyness (strike / spot)' if use_moneyness else 'Strike price ($)',
                   tickformat='.2f' if use_moneyness else ',.0f', tickprefix='' if use_moneyness else '$',
                   gridcolor=grid, zeroline=False),
        yaxis=dict(title='Implied volatility (%)', ticksuffix='%', gridcolor=grid, zeroline=False),
        legend=dict(orientation='h', x=0, y=1.1),
        margin=dict(l=60, r=20, t=40, b=60),
        height=480,
        hovermode='closest'
    )
    return fig


def analyse_chain(chain: pd.DataFrame, params: AnalysisParams, on_status: Optional[StatusCallback] = None,
                  on_progress: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    """Filter a fetched chain, select out-of-the-money options, solve the IVs and build the figure."""
    def report(kind: str, message: str) -> None:
        if on_status is not None:
            on_status(kind, message)
    
    options_df = filter_chain(chain, params)
    if options_df.empty:
        raise AnalysisError("No valid options data found.")
    
    spot_price = options_df['S'].iloc[0]
    fetched_dividend = float(options_df['q'].iloc[0])
    dividend_override = params.dividend_override
    used_dividend = dividend_override if dividend_override is not None else fetched_dividend
    dividend_source = "manual override" if dividend_override is not None else "looked up"
    
    # Out-of-the-money options only: puts below the forward price, calls at or above it
    options_df = select_otm_options(options_df, params.risk_free_decimal, used_dividend)
    
    # Check against minimum threshold from config
    if len(options_df) < MarketDataConfig.MIN_VALID_OPTIONS:
        raise AnalysisError(
            f"Insufficient options data. Found {len(options_df)} usable out-of-the-money options, "
            f"need at least {MarketDataConfig.MIN_VALID_OPTIONS}."
        )
    report('success',
           f"Market data fetched - {params.ticker} @ ${spot_price:.2f} | "
           f"dividend yield {used_dividend * 100:.2f}% ({dividend_source})")
    
    report('info', "Computing implied volatilities...")
    ivs, valid_options = calculate_ivs(options_df, params.risk_free_decimal, dividend_override, on_progress)
    
    if len(ivs) < MarketDataConfig.MIN_VALID_OPTIONS:
        raise AnalysisError(f"Insufficient valid options for analysis. Found only {len(ivs)} valid IVs.")
    
    fig, skipped_slice_days = build_surface_figure(valid_options, ivs, spot_price, params)
    
    # Calculate IV statistics
    iv_stats = calculate_iv_statistics(valid_options, ivs, spot_price)
    
    return {
        'fig': fig,
        'spot_price': spot_price,
        'ticker': params.ticker,
        'ivs': ivs,
        'valid_options': valid_options,
        'iv_stats': iv_stats,
        'dividend_yield_used': used_dividend,
        'dividend_source': dividend_source,
        'skipped_slice_days': skipped_slice_days,
        'n_puts': sum(1 for opt in valid_options if opt['type'] == 'put'),
        'n_calls': sum(1 for opt in valid_options if opt['type'] == 'call')
    }


def run_analysis(params: AnalysisParams, on_status: Optional[StatusCallback] = None,
                 on_progress: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    """Fetch the chain for the chosen ticker and analyse it."""
    if on_status is not None:
        on_status('info', "Fetching market data...")
    
    return analyse_chain(fetch_options(params), params, on_status, on_progress)