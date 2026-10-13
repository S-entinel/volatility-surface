"""Fetch option data, select out-of-the-money options, solve implied volatilities and build the surface."""

import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple
from src.data.market_data import OptionDataFetcher
from src.data.option_selection import select_otm_options
from src.svi import (
    BlendedSmile, SviSlice, blended_smile, calendar_check, delta_log_moneyness, fit_slice, surface_iv,
    surface_iv_log_moneyness, total_variance
)
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
    surface_method: str = 'Interpolated'

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


def fit_surface_svi(valid_options: List[Any], ivs: List[float], spot_price: float) -> Dict[str, Any]:
    """Fit an SVI curve to every expiry, check for arbitrage, and evaluate the fitted surface on a grid."""
    groups: Dict[float, List[int]] = {}
    for index, opt in enumerate(valid_options):
        groups.setdefault(round(float(opt['T']), 10), []).append(index)
    
    slices = []
    for maturity, indices in sorted(groups.items()):
        forward = float(valid_options[indices[0]]['forward'])
        fitted = fit_slice(
            k=np.log(np.array([float(valid_options[i]['strike']) for i in indices]) / forward),
            iv=np.array([ivs[i] for i in indices]),
            maturity=maturity, days=int(valid_options[indices[0]]['days_to_expiry']), forward=forward
        )
        if fitted is not None:
            slices.append(fitted)
    
    fit = {'slices': slices, 'n_expiries': len(groups), 'calendar': calendar_check(slices), 'mesh': None}
    if len(slices) >= 2:
        strikes = np.linspace(min(float(o['strike']) for o in valid_options),
                              max(float(o['strike']) for o in valid_options), 80)
        times = np.array([s.maturity for s in slices])
        maturities = np.unique(np.concatenate([np.linspace(times.min(), times.max(), 50), times]))
        fit['mesh'] = {'strikes': strikes, 'maturities': maturities, 'iv': surface_iv(slices, strikes, maturities)}
    return fit


def build_surface_figure(valid_options: List[Any], ivs: List[float], spot_price: float,
                         params: AnalysisParams, svi: Optional[Dict[str, Any]] = None) -> Tuple[go.Figure, List[int]]:
    """Build the 3D surface with smile slices; also return the slices that could not be drawn."""
    strikes = np.array([_axis_value(opt, params.y_axis_type) for opt in valid_options])
    
    surface_data = SurfaceData(
        strikes=strikes,
        expiries=np.array([opt['T'] for opt in valid_options]),
        ivs=np.array(ivs),
        spot_price=spot_price,
        y_axis_type=params.y_axis_type.split()[0]
    )
    
    mesh = None
    if svi is not None and svi['mesh'] is not None:
        columns, maturities, iv = _svi_grid(svi, params.y_axis_type)
        strike_mesh, expiry_mesh = np.meshgrid(columns, maturities)
        mesh = (strike_mesh, expiry_mesh, np.ma.array(iv, mask=np.isnan(iv)))
    
    plotter = SurfacePlotter(surface_data, mesh)
    # Pass ticker to create_surface_plot
    fig = plotter.create_surface_plot(theme=params.theme.lower(), colormap=params.colormap,
                                      ticker=params.ticker)
    if params.show_slices:
        fig = plotter.add_smile_slices(fig, theme=params.theme.lower())
    return fig, list(plotter.skipped_slice_days)


AXIS_STRIKE = "Strike Price ($)"
AXIS_MONEYNESS = "Moneyness"  # the log of strike over forward, so 0 is at the money on every expiry
AXIS_STYLES = {
    AXIS_STRIKE: {'title': 'Strike price ($)', 'tickformat': ',.0f', 'tickprefix': '$', 'hover': 'Strike: $%{x:,.0f}'},
    AXIS_MONEYNESS: {'title': 'Moneyness: ln(strike / forward)', 'tickformat': '.2f', 'tickprefix': '',
                     'hover': 'Moneyness: %{x:.3f}'},
}

KEY_FIGURE_NEAR_DAYS = 30
KEY_FIGURE_FAR_DAYS = 90


def _smile_at(results: Dict[str, Any], days: List[int], target_days: int) -> Optional[BlendedSmile]:
    """The smile at a target maturity, blended from the fits of the expiries either side of it."""
    below = [d for d in days if d <= target_days]
    above = [d for d in days if d >= target_days]
    if not below or not above:
        return None
    earlier, later = smile_fit(results, max(below)), smile_fit(results, min(above))
    if earlier is None or later is None:
        return None
    return blended_smile(earlier, later, target_days / 365)


def _atm_volatility(smile: Optional[BlendedSmile], target_days: int) -> Optional[float]:
    """Volatility at the forward price (log-moneyness 0), if the smile covers it."""
    if smile is None or not smile.k_low <= 0.0 <= smile.k_high:
        return None
    return float(np.sqrt(smile.variance(np.array([0.0]))[0] / (target_days / 365)))


def _risk_reversal(smile: BlendedSmile, maturity: float, dividend_yield: float) -> Optional[float]:
    """25-delta call volatility minus 25-delta put volatility, in volatility points, if the smile covers both."""
    k_call = delta_log_moneyness(smile, maturity, dividend_yield, 0.25)
    k_put = delta_log_moneyness(smile, maturity, dividend_yield, -0.25)
    if k_call is None or k_put is None:
        return None
    iv_call = float(np.sqrt(smile.variance(np.array([k_call]))[0] / maturity))
    iv_put = float(np.sqrt(smile.variance(np.array([k_put]))[0] / maturity))
    return (iv_call - iv_put) * StatisticsConfig.IV_DISPLAY_MULTIPLIER


def key_figures(results: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Fixed-maturity summaries of the surface, None where the data cannot give one.

    atm_near: at-the-money volatility at 30 days, in percent. move_pct and move_usd: the one-standard-deviation
    move that volatility implies by 30 days, as a percentage of spot and in dollars. rr_near: 25-delta call
    volatility minus 25-delta put volatility at 30 days, in volatility points. term: 90-day minus 30-day
    at-the-money volatility, in volatility points. Each uses SVI fits of the two expiries either side of the maturity.
    """
    figures: Dict[str, Optional[float]] = {
        'atm_near': None, 'move_pct': None, 'move_usd': None, 'rr_near': None, 'term': None
    }
    days = sorted({int(opt['days_to_expiry']) for opt in results['valid_options']})
    multiplier = StatisticsConfig.IV_DISPLAY_MULTIPLIER

    near = _smile_at(results, days, KEY_FIGURE_NEAR_DAYS)
    atm_near = _atm_volatility(near, KEY_FIGURE_NEAR_DAYS)
    if atm_near is not None:
        figures['atm_near'] = atm_near * multiplier
        one_sigma = atm_near * np.sqrt(KEY_FIGURE_NEAR_DAYS / 365)
        figures['move_pct'] = one_sigma * multiplier
        figures['move_usd'] = one_sigma * results['spot_price']
    if near is not None and atm_near is not None:
        figures['rr_near'] = _risk_reversal(near, KEY_FIGURE_NEAR_DAYS / 365, results['dividend_yield_used'])
    atm_far = _atm_volatility(_smile_at(results, days, KEY_FIGURE_FAR_DAYS), KEY_FIGURE_FAR_DAYS)
    if atm_near is not None and atm_far is not None:
        figures['term'] = (atm_far - atm_near) * multiplier
    return figures


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


def smile_stats(results: Dict[str, Any], expiry_days: int) -> Dict[str, Optional[float]]:
    """ATM volatility (percent) and 25-delta risk reversal (volatility points) for one expiry, None where unavailable."""
    fit = smile_fit(results, expiry_days)
    if fit is None:
        return {'atm_iv': None, 'rr_25': None}
    smile = blended_smile(fit, fit, fit.maturity)
    atm = _atm_volatility(smile, expiry_days)
    return {
        'atm_iv': None if atm is None else atm * StatisticsConfig.IV_DISPLAY_MULTIPLIER,
        'rr_25': _risk_reversal(smile, fit.maturity, results['dividend_yield_used']),
    }


def _axis_style(y_axis_type: str) -> Dict[str, str]:
    """Title, hover text and tick format for the chosen strike axis."""
    return AXIS_STYLES.get(y_axis_type, AXIS_STYLES[AXIS_STRIKE])


def _axis_value(opt: Any, y_axis_type: str, key: str = 'strike') -> float:
    """A strike (or forward) price of an option on the chosen strike axis: dollars, or ln(price / forward)."""
    price = float(opt[key])
    if y_axis_type == AXIS_MONEYNESS:
        return float(np.log(price / float(opt['forward'])))
    return price


def _curve_axis_values(forward: float, k: np.ndarray, y_axis_type: str) -> np.ndarray:
    """Positions on the chosen strike axis for log-moneyness values k of a smile with the given forward."""
    return k if y_axis_type == AXIS_MONEYNESS else forward * np.exp(k)


def _svi_grid(svi: Dict[str, Any], y_axis_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The SVI surface as columns on the chosen strike axis, maturities and implied volatilities."""
    mesh = svi['mesh']
    if y_axis_type != AXIS_MONEYNESS:
        return mesh['strikes'], mesh['maturities'], mesh['iv']
    slices = svi['slices']
    k = np.linspace(min(s.k_min for s in slices), max(s.k_max for s in slices), len(mesh['strikes']))
    return k, mesh['maturities'], surface_iv_log_moneyness(slices, k, mesh['maturities'])


def smile_fit(results: Dict[str, Any], expiry_days: int) -> Optional[SviSlice]:
    """The SVI fit for one expiry: the surface fit if there is one, otherwise fitted now for this expiry only."""
    existing = next((s for s in (results.get('svi') or {}).get('slices', []) if s.days == expiry_days), None)
    if existing is not None:
        return existing
    options, ivs = _expiry_points(results, expiry_days)
    if not options or 'forward' not in options[0]:
        return None
    forward = float(options[0]['forward'])
    return fit_slice(
        k=np.log(np.array([float(opt['strike']) for opt in options]) / forward), iv=np.array(ivs),
        maturity=float(options[0]['T']), days=expiry_days, forward=forward
    )


def build_smile_figure(results: Dict[str, Any], expiry_days: int, y_axis_type: str) -> go.Figure:
    """Implied volatility against strike for one expiry, with puts and calls in different colours."""
    options, ivs = _expiry_points(results, expiry_days)
    style = _axis_style(y_axis_type)
    font = 'IBM Plex Mono, monospace'
    grid = 'rgb(38, 38, 38)'
    multiplier = StatisticsConfig.IV_DISPLAY_MULTIPLIER
    
    ordered = sorted(zip(options, ivs), key=lambda pair: _axis_value(pair[0], y_axis_type))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[_axis_value(opt, y_axis_type) for opt, _ in ordered], y=[iv * multiplier for _, iv in ordered],
        mode='lines', line=dict(color='rgb(70, 70, 70)', width=1), hoverinfo='skip', showlegend=False
    ))
    
    for option_type, name, colour in (('put', 'Puts', '#35b779'), ('call', 'Calls', '#fde725')):
        for source, suffix, symbol in (('mid', '', 'circle'), ('last', ' (last price)', 'circle-open')):
            points = [(opt, iv) for opt, iv in ordered
                      if opt['type'] == option_type and (opt.get('price_source') or 'mid') == source]
            if not points:
                continue
            fig.add_trace(go.Scatter(
                x=[_axis_value(opt, y_axis_type) for opt, _ in points],
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
    
    fit = smile_fit(results, expiry_days)
    if fit is not None:
        k = np.linspace(fit.k_min, fit.k_max, 120)
        fig.add_trace(go.Scatter(
            x=_curve_axis_values(fit.forward, k, y_axis_type),
            y=np.sqrt(total_variance(k, fit.params) / fit.maturity) * multiplier,
            mode='lines', name='SVI fit', line=dict(color='white', width=2), hoverinfo='skip'
        ))
    
    if options and 'forward' in options[0]:
        fig.add_vline(x=_axis_value(options[0], y_axis_type, key='forward'), line_dash='dot',
                      line_color='rgb(120, 120, 120)', annotation_text="forward",
                      annotation_font=dict(color='rgb(160, 160, 160)', size=11, family=font))
    
    fig.update_layout(
        paper_bgcolor='rgb(0, 0, 0)', plot_bgcolor='rgb(0, 0, 0)',
        font=dict(family=font, color='white'),
        xaxis=dict(title=style['title'], tickformat=style['tickformat'], tickprefix=style['tickprefix'],
                   gridcolor=grid, zeroline=False),
        yaxis=dict(title='Implied volatility (%)', ticksuffix='%', gridcolor=grid, zeroline=False),
        legend=dict(orientation='h', x=0, y=1.1),
        margin=dict(l=60, r=20, t=40, b=60),
        height=480,
        hovermode='closest'
    )
    return fig


def build_heatmap_figure(results: Dict[str, Any], y_axis_type: str, show_points: bool = False) -> go.Figure:
    """The surface as a 2D heatmap of implied volatility over strike and expiry."""
    options, ivs = results['valid_options'], results['ivs']
    style = _axis_style(y_axis_type)
    font = 'IBM Plex Mono, monospace'
    multiplier = StatisticsConfig.IV_DISPLAY_MULTIPLIER
    
    svi_mesh = (results.get('svi') or {}).get('mesh')
    point_x = np.array([_axis_value(opt, y_axis_type) for opt in options])
    point_root = np.sqrt(np.array([opt['T'] for opt in options], dtype=float))
    
    # Own grid for the heatmap: rows evenly spaced in the square root of time, which gives the crowded
    # short expiries more rows, and interpolation on axes scaled to the same range
    size = 80
    if svi_mesh is not None:
        x, maturities, svi_iv = _svi_grid(results['svi'], y_axis_type)
        roots = np.sqrt(maturities)
        z = svi_iv * multiplier
    else:
        x = np.linspace(point_x.min(), point_x.max(), size)
        roots = np.unique(np.concatenate([np.linspace(point_root.min(), point_root.max(), size),
                                          np.unique(point_root)]))
        roots = roots[np.concatenate([[True], np.diff(roots) > 1e-9])]
        x_span = (point_x.max() - point_x.min()) or 1.0
        root_span = (point_root.max() - point_root.min()) or 1.0
        grid_x, grid_root = np.meshgrid(x, roots)
        z = griddata(
            np.column_stack(((point_root - point_root.min()) / root_span, (point_x - point_x.min()) / x_span)),
            np.array(ivs), ((grid_root - point_root.min()) / root_span, (grid_x - point_x.min()) / x_span),
            method='linear'
        ) * multiplier
        maturities = roots ** 2
    
    # Time is drawn on a square root scale so the crowded short expiries get room; ticks are in years
    tick_years = [t for t in (0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 3) if maturities[0] <= t <= maturities[-1]]
    
    hover_x = style['hover']
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=x, y=roots, z=z, colorscale='Viridis', zsmooth='best',
        customdata=np.repeat(maturities[:, None], len(x), axis=1),
        hovertemplate='Expiry: %{customdata:.2f} years<br>' + hover_x + '<br>Implied volatility: %{z:.1f}%<extra></extra>',
        colorbar=dict(title=dict(text='Implied volatility', side='top', font=dict(size=12, family=font)),
                      thickness=14, len=0.8, tickformat='.0f', ticksuffix='%',
                      tickfont=dict(size=11, family=font))
    ))
    
    if show_points:
        for source, name, symbol in (('mid', 'Live quotes', 'circle'), ('last', 'Last traded prices', 'circle-open')):
            points = [(opt, iv) for opt, iv in zip(options, ivs) if (opt.get('price_source') or 'mid') == source]
            if not points:
                continue
            fig.add_trace(go.Scatter(
                x=[_axis_value(opt, y_axis_type) for opt, _ in points], y=[np.sqrt(opt['T']) for opt, _ in points],
                mode='markers', name=name,
                marker=dict(size=3, symbol=symbol, color='rgba(255, 255, 255, 0.55)', line=dict(width=1)),
                customdata=[[opt['type'], float(opt['strike']), int(opt['days_to_expiry']), iv * multiplier]
                            for opt, iv in points],
                hovertemplate=('%{customdata[0]} at $%{customdata[1]:,.2f}<br>%{customdata[2]} days to expiry'
                               '<br>Implied volatility: %{customdata[3]:.1f}%<extra></extra>')
            ))
    
    if options and 'forward' in options[0]:
        forwards = {float(opt['T']): _axis_value(opt, y_axis_type, key='forward') for opt in options}
        times = sorted(forwards)
        fig.add_trace(go.Scatter(
            x=[forwards[t] for t in times], y=np.sqrt(times), mode='lines', hoverinfo='skip',
            line=dict(color='rgba(255, 255, 255, 0.6)', width=1, dash='dot'), showlegend=False
        ))
    
    fig.update_layout(
        paper_bgcolor='rgb(0, 0, 0)', plot_bgcolor='rgb(0, 0, 0)',
        font=dict(family=font, color='white'),
        xaxis=dict(title=style['title'], tickformat=style['tickformat'], tickprefix=style['tickprefix'],
                   showgrid=False, zeroline=False),
        yaxis=dict(title='Time to expiry (years)', tickmode='array', tickvals=list(np.sqrt(tick_years)),
                   ticktext=[f"{t:g}" for t in tick_years], showgrid=False, zeroline=False),
        legend=dict(orientation='h', x=0, y=1.08),
        showlegend=show_points,
        margin=dict(l=70, r=20, t=40, b=60),
        height=560,
        hovermode='closest'
    )
    return fig


def figure_html(fig: go.Figure) -> str:
    """A standalone page with the interactive chart. It fills the browser window; plotly.js loads from a CDN."""
    page = go.Figure(fig).update_layout(width=None, height=None, autosize=True)
    html = page.to_html(include_plotlyjs="cdn", default_width="100%", default_height="100vh")
    return html.replace("<body>", '<body style="margin: 0; background: #000000;">', 1)


def _report(on_status: Optional[StatusCallback], kind: str, message: str) -> None:
    """Pass a status message to the callback, if there is one."""
    if on_status is not None:
        on_status(kind, message)


def analyse_chain(chain: pd.DataFrame, params: AnalysisParams, on_status: Optional[StatusCallback] = None,
                  on_progress: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    """Filter a fetched chain, select out-of-the-money options, solve the IVs and build the figure."""
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
    _report(on_status, 'success',
           f"Market data fetched - {params.ticker} @ ${spot_price:.2f} | "
           f"dividend yield {used_dividend * 100:.2f}% ({dividend_source})")
    
    _report(on_status, 'info', "Computing implied volatilities...")
    ivs, valid_options = calculate_ivs(options_df, params.risk_free_decimal, dividend_override, on_progress)
    
    if len(ivs) < MarketDataConfig.MIN_VALID_OPTIONS:
        raise AnalysisError(f"Insufficient valid options for analysis. Found only {len(ivs)} valid IVs.")
    
    svi = fit_surface_svi(valid_options, ivs, spot_price) if params.surface_method == "SVI fit" else None
    fig, skipped_slice_days = build_surface_figure(valid_options, ivs, spot_price, params, svi)
    
    results = {
        'fig': fig,
        'spot_price': spot_price,
        'ticker': params.ticker,
        'ivs': ivs,
        'valid_options': valid_options,
        'dividend_yield_used': used_dividend,
        'dividend_source': dividend_source,
        'skipped_slice_days': skipped_slice_days,
        'n_puts': sum(1 for opt in valid_options if opt['type'] == 'put'),
        'n_calls': sum(1 for opt in valid_options if opt['type'] == 'call')
    }
    if svi is not None:
        results['svi'] = svi
    return results


def run_analysis(params: AnalysisParams, on_status: Optional[StatusCallback] = None,
                 on_progress: Optional[ProgressCallback] = None) -> Dict[str, Any]:
    """Fetch the chain for the chosen ticker and analyse it."""
    if on_status is not None:
        on_status('info', "Fetching market data...")
    
    return analyse_chain(fetch_options(params), params, on_status, on_progress)