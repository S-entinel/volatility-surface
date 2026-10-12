"""Streamlit user interface for the implied volatility surface analyser."""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
from src.analysis import AnalysisError, AnalysisParams, analyse_chain, fetch_full_chain
from src.config.config import (
    UIConfig, 
    MarketDataConfig, 
    ModelConfig
)


def apply_custom_style() -> None:
    """
    Apply custom CSS styling to the Streamlit app.
    
    Returns:
        None
    """
    st.markdown("""
        <style>
        .main > div {
            padding-top: 2rem;
        }
        .stApp {
            font-family: "IBM Plex Mono", monospace;
        }
        .stAlert {
            background-color: rgba(28, 131, 225, 0.1);
            padding: 1rem;
            border-radius: 4px;
        }
        div[data-testid="metric-container"] {
            background-color: rgba(28, 131, 225, 0.1);
            border: 1px solid rgba(28, 131, 225, 0.1);
            padding: 1rem;
            border-radius: 4px;
            margin: 0.5rem 0;
        }
        .stButton > button {
            width: 100%;
            font-family: "IBM Plex Mono", monospace;
            font-weight: 500;
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
        with st.expander("Market Data Configuration", expanded=True):
            ticker = st.text_input(
                "Ticker Symbol",
                value=UIConfig.DEFAULT_TICKER
            ).upper()
            
            st.markdown("#### Strike Price Range")
            col1, col2 = st.columns(2)
            with col1:
                min_strike_pct = st.number_input(
                    "Min Strike %",
                    min_value=UIConfig.MIN_STRIKE_PCT_LOWER,
                    max_value=UIConfig.MIN_STRIKE_PCT_UPPER,
                    value=MarketDataConfig.DEFAULT_MIN_STRIKE_PCT,
                    format="%.1f"
                )
            with col2:
                max_strike_pct = st.number_input(
                    "Max Strike %",
                    min_value=min_strike_pct + 1,
                    max_value=UIConfig.MAX_STRIKE_PCT_UPPER,
                    value=MarketDataConfig.DEFAULT_MAX_STRIKE_PCT,
                    format="%.1f"
                )
                
            min_open_interest = st.number_input(
                "Min Open Interest",
                min_value=MarketDataConfig.MIN_OPEN_INTEREST_THRESHOLD,
                value=MarketDataConfig.DEFAULT_MIN_OPEN_INTEREST,
                help="Contracts with fewer open positions than this are ignored. "
                     "Yahoo often reports no open interest before the US market opens, so 0 keeps everything."
            )
        
        with st.expander("Model Parameters", expanded=True):
            risk_free_rate = st.number_input(
                "Risk-Free Rate (%)",
                min_value=UIConfig.RISK_FREE_RATE_MIN,
                max_value=UIConfig.RISK_FREE_RATE_MAX,
                value=ModelConfig.DEFAULT_RISK_FREE_RATE * 100,  # Convert to percentage
                step=UIConfig.RISK_FREE_RATE_STEP,
                format="%.1f",
                help="Annual risk-free rate (e.g., 4.5 for 4.5%)"
            )

            override_dividend = st.checkbox(
                "Override dividend yield",
                value=False,
                help="By default the dividend yield is looked up for the ticker. "
                     "Tick this box to enter your own value instead."
            )

            dividend_yield = st.number_input(
                "Dividend Yield (%)",
                min_value=UIConfig.DIVIDEND_YIELD_MIN,
                max_value=UIConfig.DIVIDEND_YIELD_MAX,
                value=ModelConfig.DEFAULT_DIVIDEND_YIELD * 100,  # Convert to percentage
                step=UIConfig.DIVIDEND_YIELD_STEP,
                format="%.1f",
                disabled=not override_dividend,
                help="Annual dividend yield (e.g., 1.3 for 1.3%). Only used when the override box is ticked."
            )
        
        with st.expander("Visualisation Settings", expanded=True):
            theme = st.selectbox(
                "Theme",
                options=UIConfig.AVAILABLE_THEMES
            )
            
            colormap = st.selectbox(
                "Colour Scheme",
                options=UIConfig.AVAILABLE_COLORMAPS
            )
            
            y_axis_type = st.selectbox(
                "Y-Axis Scale",
                options=UIConfig.AVAILABLE_Y_AXIS_TYPES
            )
        
        refresh = st.button(
            "Refresh data",
            use_container_width=True,
            help="Fetch fresh prices instead of reusing the data from the last few minutes."
        )

    params = AnalysisParams(
        ticker=ticker,
        min_strike_pct=min_strike_pct,
        max_strike_pct=max_strike_pct,
        min_open_interest=min_open_interest,
        risk_free_rate=risk_free_rate,
        override_dividend=override_dividend,
        dividend_yield=dividend_yield,
        theme=theme,
        colormap=colormap,
        y_axis_type=y_axis_type
    )
    return params, refresh


def render_results(results: Dict[str, Any], fetched_at: Optional[datetime] = None) -> None:
    """Draw the chart, metrics, captions and the CSV export button."""
    # Main visualisation
    st.plotly_chart(results['fig'], use_container_width=True)

    if results.get('skipped_slice_days'):
        skipped_text = ", ".join(f"{days}d" for days in results['skipped_slice_days'])
        st.caption(
            f"Smile slice(s) not drawn because they fall outside the available expiries: {skipped_text}."
        )
    
    # Metrics in clean table format below visualisation
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Spot Price", f"${results['spot_price']:.2f}")
        st.metric("ATM IV", f"{results['iv_stats']['atm_iv']:.1f}%")
    
    with col2:
        st.metric("Valid Options", f"{len(results['ivs'])}")
        if results['iv_stats']['iv_skew'] is not None:
            st.metric("IV Skew", f"{results['iv_stats']['iv_skew']:.1f}%")
        else:
            st.metric("IV Skew", "N/A")
    
    with col3:
        st.metric("Term Structure", f"{results['iv_stats']['term_structure']:.1f}%")
        st.metric("IV Range", f"{results['iv_stats']['iv_min']:.1f}% - {results['iv_stats']['iv_max']:.1f}%")
    
    with col4:
        st.metric("Avg IV", f"{results['iv_stats']['avg_iv']:.1f}%")
        st.metric("IV Std Dev", f"{results['iv_stats']['iv_std']:.1f}%")
    
    st.caption(
        f"Built from out-of-the-money options only: {results['n_puts']} puts below the forward price "
        f"and {results['n_calls']} calls at or above it."
    )
    st.caption(
        f"Dividend yield used: {results['dividend_yield_used'] * 100:.2f}% ({results['dividend_source']})"
    )
    if fetched_at is not None:
        st.caption(
            f"Data fetched at {fetched_at:%H:%M:%S}, reused for up to {UIConfig.CACHE_TTL_SECONDS // 60} "
            f"minutes. Use Refresh data for the latest prices."
        )
    
    # Export section
    st.markdown("---")
    col_export1, col_export2, col_export3 = st.columns([1, 1, 2])
    
    with col_export1:
        if len(results['valid_options']) > 0:
            df_download = pd.DataFrame({
                'Strike': [opt['strike'] for opt in results['valid_options']],
                'Days_to_Expiry': [opt['days_to_expiry'] for opt in results['valid_options']],
                'IV': [iv * 100 for iv in results['ivs']],
                'Option_Type': [opt['type'] for opt in results['valid_options']],
                'Moneyness': [opt['strike']/results['spot_price'] for opt in results['valid_options']]
            })
            
            st.download_button(
                "Export Data (CSV)",
                data=df_download.to_csv(index=False).encode('utf-8'),
                file_name=f"{results['ticker']}_iv_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
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
            chain, fetched_at = cached_chain(params.ticker)
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
    
    render_results(st.session_state.analysis_results, fetched_at)


if __name__ == "__main__":
    main()