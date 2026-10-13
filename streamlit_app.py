"""Streamlit user interface for the implied volatility surface analyser."""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
from src.analysis import (
    AnalysisError, AnalysisParams, analyse_chain, build_heatmap_figure, build_smile_figure, expiry_labels,
    fetch_full_chain, figure_html, key_figures, smile_fit, smile_stats
)
from src.config.config import (
    UIConfig, 
    MarketDataConfig, 
    ModelConfig
)


# The chart is always dark with the Viridis colour scale
CHART_THEME = UIConfig.AVAILABLE_THEMES[0]
COLOUR_SCALE = 'Viridis'

STRIKE_AXIS_NAMES = {'Strike Price ($)': 'Strike price ($)', 'Moneyness': 'Moneyness (strike / spot)'}


def apply_custom_style() -> None:
    """Apply the black and white styling; the surface chart is the only colour on the page."""
    st.markdown("""
        <style>
        .block-container {
            padding-top: 2rem;
        }
        .stApp {
            font-family: "IBM Plex Mono", monospace;
        }
        div[data-testid="stMetric"], div[data-testid="metric-container"] {
            background-color: #111111;
            border: 1px solid #333333;
            padding: 1rem;
            border-radius: 2px;
        }
        .stAlert {
            background-color: #111111;
            border: 1px solid #333333;
            border-radius: 2px;
        }
        .stButton > button {
            width: 100%;
            font-family: "IBM Plex Mono", monospace;
            font-weight: 500;
            border-radius: 2px;
        }
        button[data-baseweb="tab"] {
            font-family: "IBM Plex Mono", monospace;
        }
        div[data-baseweb="tab-highlight"] {
            background-color: #FFFFFF;
        }
        section[data-testid="stSidebar"] h3 {
            font-size: 0.72rem;
            font-weight: 600;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            color: #9A9A9A;
            padding-bottom: 0.25rem;
        }
        section[data-testid="stSidebar"] hr {
            border-color: #262626;
            margin: 1.1rem 0;
        }
        section[data-testid="stSidebar"] label p {
            font-size: 0.85rem;
        }
        </style>
    """, unsafe_allow_html=True)


def show_status(kind: str, message: str) -> None:
    """Show an analysis status message as a success or info box."""
    if kind == 'success':
        st.success(message)
    else:
        st.info(message)


class ProgressDisplay:
    """Shows the implied volatility progress bar, and removes it when the last option is done."""
    
    def __init__(self) -> None:
        self.bar = None
        self.text = None
    
    def __call__(self, done: int, total: int) -> None:
        if self.bar is None:
            self.bar = st.progress(0)
            self.text = st.empty()
        self.text.text(f"Calculating IV for option {done} of {total}")
        self.bar.progress(done / total)
        if done == total:
            self.bar.empty()
            self.text.empty()


@st.cache_data(ttl=UIConfig.CACHE_TTL_SECONDS, show_spinner=False)
def cached_chain(ticker: str) -> Tuple[pd.DataFrame, datetime]:
    """The full option chain for a ticker and when it was fetched, reused between reruns."""
    return fetch_full_chain(ticker), datetime.now()


def render_sidebar() -> Tuple[AnalysisParams, bool]:
    """Draw the sidebar; return the chosen parameters and whether Refresh data was clicked."""
    with st.sidebar:
        st.subheader("Market")
        ticker = st.text_input(
            "Ticker",
            value=UIConfig.DEFAULT_TICKER,
            placeholder="e.g. SPY",
            help="A US-listed stock or ETF that has options, such as SPY, QQQ or AAPL."
        ).strip().upper()
        refresh = st.button(
            "Refresh data",
            use_container_width=True,
            help="Fetch fresh prices instead of reusing the data from the last few minutes."
        )
        
        st.divider()
        st.subheader("Filters")
        strike_low, strike_high = st.slider(
            "Strike range (% of spot)",
            min_value=int(UIConfig.MIN_STRIKE_PCT_LOWER),
            max_value=int(UIConfig.MAX_STRIKE_PCT_UPPER),
            value=(int(MarketDataConfig.DEFAULT_MIN_STRIKE_PCT), int(MarketDataConfig.DEFAULT_MAX_STRIKE_PCT)),
            help="Only contracts with strikes inside this window are used. 100% is the current price."
        )
        min_open_interest = st.number_input(
            "Minimum open interest (contracts)",
            min_value=MarketDataConfig.MIN_OPEN_INTEREST_THRESHOLD,
            value=MarketDataConfig.DEFAULT_MIN_OPEN_INTEREST,
            help="Contracts with fewer open positions than this are ignored. "
                 "Yahoo often reports no open interest before the US market opens, so 0 keeps everything."
        )
        
        st.divider()
        st.subheader("Chart")
        y_axis_type = st.selectbox(
            "Strike axis",
            options=UIConfig.AVAILABLE_Y_AXIS_TYPES,
            format_func=lambda option: STRIKE_AXIS_NAMES.get(option, option),
            help="Show strikes as dollar prices, or as a multiple of the current price."
        )
        
        surface_method = st.radio(
            "Surface method",
            options=["Interpolated", "SVI fit"],
            help="Interpolated joins your quotes with straight lines. SVI fits a smooth, standard curve to each "
                 "expiry and checks it for arbitrage."
        )
        
        st.divider()
        with st.expander("Advanced settings"):
            risk_free_rate = st.number_input(
                "Risk-free rate (% per year)",
                min_value=UIConfig.RISK_FREE_RATE_MIN,
                max_value=UIConfig.RISK_FREE_RATE_MAX,
                value=ModelConfig.DEFAULT_RISK_FREE_RATE * 100,  # Convert to percentage
                step=UIConfig.RISK_FREE_RATE_STEP,
                format="%.1f",
                help="Used to discount option prices and to place the forward price (e.g. 4.5 for 4.5%)."
            )
            
            override_dividend = st.checkbox(
                "Override dividend yield",
                value=False,
                help="By default the dividend yield is looked up for the ticker. "
                     "Tick this box to enter your own value instead."
            )
            
            dividend_yield = st.number_input(
                "Dividend yield (% per year)",
                min_value=UIConfig.DIVIDEND_YIELD_MIN,
                max_value=UIConfig.DIVIDEND_YIELD_MAX,
                value=ModelConfig.DEFAULT_DIVIDEND_YIELD * 100,  # Convert to percentage
                step=UIConfig.DIVIDEND_YIELD_STEP,
                format="%.1f",
                disabled=not override_dividend,
                help="Only used when the override box above is ticked (e.g. 1.0 for 1%)."
            )
        
        st.caption(
            "Out-of-the-money options only. Prices are bid-ask midpoints, "
            "or the last traded price when there is no live quote."
        )
    
    params = AnalysisParams(
        ticker=ticker,
        min_strike_pct=float(strike_low),
        max_strike_pct=float(strike_high),
        min_open_interest=min_open_interest,
        risk_free_rate=risk_free_rate,
        override_dividend=override_dividend,
        dividend_yield=dividend_yield,
        theme=CHART_THEME,
        colormap=COLOUR_SCALE,
        y_axis_type=y_axis_type,
        show_slices=False,
        surface_method=surface_method
    )
    return params, refresh


def results_table(results: Dict[str, Any]) -> pd.DataFrame:
    """One row per implied volatility point, as shown in the Data & Export tab and the CSV."""
    options = results['valid_options']
    return pd.DataFrame({
        'Strike': [opt['strike'] for opt in options],
        'Days_to_Expiry': [opt['days_to_expiry'] for opt in options],
        'Option_Type': [opt['type'] for opt in options],
        'Price': [round(float(opt['price']), 4) for opt in options],
        'Price_Source': [opt.get('price_source') for opt in options],
        'Open_Interest': [opt.get('open_interest') for opt in options],
        'IV': [round(iv * 100, 4) for iv in results['ivs']],
        'Moneyness': [round(opt['strike'] / results['spot_price'], 4) for opt in options]
    })


# Only the smile panel reruns when an expiry is picked, so choosing one is instant
fragment = getattr(st, "fragment", lambda func: func)


@fragment
def render_smile_tab(results: Dict[str, Any], y_axis_type: str) -> None:
    """Pick an expiry and show its implied volatility smile."""
    labels = expiry_labels(results)
    days = list(labels)
    chosen = st.selectbox(
        "Expiry",
        options=days,
        index=days.index(min(days, key=lambda d: abs(d - 30))),
        format_func=lambda d: labels[d]
    )
    
    stats = smile_stats(results, chosen)
    skew = f"{stats['iv_skew']:.1f}%" if stats['iv_skew'] is not None else "N/A"
    fit = smile_fit(results, chosen)
    fit_text = ""
    if fit is not None:
        fit_text = f" The white line is an SVI fit, off by {fit.rmse * 100:.2f} vol points on average."
        if not fit.butterfly_ok:
            fit_text += " It allows butterfly arbitrage."
    st.caption(
        f"At this expiry: ATM IV {stats['atm_iv']:.1f}% · skew {skew} (out-of-the-money put minus call). "
        f"Open circles are last traded prices, not live quotes.{fit_text}"
    )
    st.plotly_chart(build_smile_figure(results, chosen, y_axis_type), use_container_width=True)


def render_svi_summary(svi: Dict[str, Any]) -> None:
    """One line on how well SVI fitted and whether it allows arbitrage, with the per-expiry detail below it."""
    slices = svi['slices']
    if svi['mesh'] is None:
        st.warning("Too few expiries could be fitted with SVI, so the interpolated surface is shown instead.")
        return
    
    butterfly = [s.days for s in slices if not s.butterfly_ok]
    calendar_pairs = svi['calendar']['violations']
    butterfly_text = "none" if not butterfly else f"{len(butterfly)} expiries"
    calendar_text = "none" if not calendar_pairs else f"{calendar_pairs} pairs of expiries"
    typical_error = np.median([s.rmse for s in slices]) * 100
    st.caption(
        f"SVI fit: {len(slices)} of {svi['n_expiries']} expiries fitted, typical error {typical_error:.2f} vol points "
        f"· butterfly arbitrage: {butterfly_text} · calendar arbitrage: {calendar_text}"
    )
    with st.expander("SVI fit details"):
        st.dataframe(pd.DataFrame({
            'Days': [s.days for s in slices],
            'Points': [s.n_points for s in slices],
            'Error (vol points)': [round(s.rmse * 100, 2) for s in slices],
            'Butterfly': ['ok' if s.butterfly_ok else 'arbitrage' for s in slices],
            'a': [round(s.params[0], 5) for s in slices],
            'b': [round(s.params[1], 4) for s in slices],
            'rho': [round(s.params[2], 3) for s in slices],
            'm': [round(s.params[3], 4) for s in slices],
            'sigma': [round(s.params[4], 4) for s in slices],
        }), use_container_width=True, hide_index=True)


@fragment
def render_surface_tab(results: Dict[str, Any], y_axis_type: str) -> None:
    """The implied volatility surface, as a 3D chart or a 2D heatmap."""
    svi = results.get('svi')
    if svi is not None:
        render_svi_summary(svi)
    
    view_column, points_column = st.columns([1, 1])
    view = view_column.radio("View", ["3D surface", "Heatmap"], horizontal=True, key="surface_view",
                             label_visibility="collapsed")
    if view == "Heatmap":
        show_points = points_column.toggle("Show data points", value=False, key="show_points")
        st.plotly_chart(build_heatmap_figure(results, y_axis_type, show_points), use_container_width=True)
        note = "The dotted line is the forward price. Time is drawn on a square root scale so short expiries are easier to see."
        if show_points:
            note += " Dots are real options. Open circles are last traded prices, not live quotes."
        st.caption(note)
    else:
        st.plotly_chart(results['fig'], use_container_width=True)


def format_points(value: Optional[float]) -> str:
    """A volatility difference in points with its sign, or N/A."""
    return "N/A" if value is None else f"{round(value, 1) + 0.0:+.1f} pts"


def render_results(results: Dict[str, Any], y_axis_type: str) -> None:
    """Draw the header, the key figures, and the Surface and Data & Export tabs."""
    expiries = len({opt['days_to_expiry'] for opt in results['valid_options']})
    st.markdown(
        f"**{results['ticker']}** · spot ${results['spot_price']:,.2f} · {len(results['ivs']):,} IV points "
        f"({results['n_puts']:,} puts / {results['n_calls']:,} calls) across {expiries} expiries"
    )
    figures = key_figures(results)
    move_label = ("30d Implied Move (1σ)" if figures['move_usd'] is None
                  else f"30d Implied Move (±${figures['move_usd']:,.2f})")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("30d ATM IV", "N/A" if figures['atm_near'] is None else f"{figures['atm_near']:.1f}%",
                help="Implied volatility at the forward price for a 30-day option, read from a smooth SVI curve "
                     "fitted through the out-of-the-money puts and calls and blended between the two nearest expiries.")
    col2.metric(move_label, "N/A" if figures['move_pct'] is None else f"±{figures['move_pct']:.1f}%",
                help="The one-standard-deviation move the options imply by 30 days, as a percentage of the current "
                     "price and in dollars (in the title): spot × 30-day ATM volatility × √(30/365). If the "
                     "volatility is right, the price finishes inside this range about two-thirds of the time.")
    col3.metric("30d Risk Reversal (25Δ)", format_points(figures['rr_near']),
                help="Volatility of a 25-delta call minus a 25-delta put, at 30 days. Negative means downside "
                     "puts cost more, which is normal for stocks and indices. More negative is a steeper skew.")
    col4.metric("Term Structure (90d - 30d)", format_points(figures['term']),
                help="At-the-money volatility at 90 days minus at 30 days. Positive means calmer near term and "
                     "more uncertainty later. Negative means near-term stress.")
    
    surface_tab, smile_tab, data_tab = st.tabs(["Surface", "Smile & Skew", "Data & Export"])
    
    with surface_tab:
        render_surface_tab(results, y_axis_type)
    
    with smile_tab:
        render_smile_tab(results, y_axis_type)
    
    with data_tab:
        table = results_table(results)
        st.dataframe(table, use_container_width=True, hide_index=True)
        today = datetime.now().strftime('%Y%m%d')
        heatmap = st.session_state.get('surface_view') == "Heatmap"
        chart = (build_heatmap_figure(results, y_axis_type, st.session_state.get('show_points', False))
                 if heatmap else results['fig'])
        csv_column, html_column, _ = st.columns([1, 1, 2])
        csv_column.download_button(
            "Export data (CSV)",
            data=table.to_csv(index=False).encode('utf-8'),
            file_name=f"{results['ticker']}_iv_analysis_{today}.csv",
            mime='text/csv',
            use_container_width=True
        )
        html_column.download_button(
            "Export chart (HTML)",
            data=figure_html(chart).encode('utf-8'),
            file_name=f"{results['ticker']}_iv_{'heatmap' if heatmap else 'surface'}_{today}.html",
            mime='text/html',
            help="An interactive copy of the chart you are viewing that opens in any browser. It needs an internet connection to load.",
            use_container_width=True
        )


def main() -> None:
    """Streamlit entry point: page setup, sidebar, analysis and results."""
    st.set_page_config(
        page_title="IV Surface Analysis",
        page_icon="■",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_style()
    
    st.title("Implied Volatility Surface Analysis")
    
    params, refresh = render_sidebar()
    
    if refresh:
        cached_chain.clear()
        st.session_state.pop('fetch_error', None)
    
    # A failed fetch is not repeated on every widget change; Refresh data retries it
    saved_error = st.session_state.get('fetch_error')
    if saved_error and saved_error['ticker'] == params.ticker:
        st.error(saved_error['text'])
        return
    
    try:
        with st.spinner("Fetching market data..."):
            chain, _ = cached_chain(params.ticker)
    except Exception as e:
        text = str(e) if isinstance(e, AnalysisError) else f"Analysis Error: {str(e)}"
        st.session_state.fetch_error = {'ticker': params.ticker, 'text': text}
        st.error(text)
        return
    
    st.session_state.pop('analysis_results', None)
    status = st.empty()
    
    try:
        with status.container():
            try:
                st.session_state.analysis_results = analyse_chain(chain, params, show_status, ProgressDisplay())
            except AnalysisError as e:
                st.error(str(e))
                return
        
        status.empty()
        
    except Exception as e:
        st.error(f"Analysis Error: {str(e)}")
        return
    
    render_results(st.session_state.analysis_results, params.y_axis_type)


if __name__ == "__main__":
    main()