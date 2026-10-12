"""
Streamlit application for Implied Volatility Surface Analysis.

Main application with complete type hints for all functions and proper
return type annotations.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime
import time
from typing import Tuple, List, Dict, Optional, Any
from src.data.market_data import OptionDataFetcher
from src.data.option_selection import select_otm_options
from src.calculators.implied_volatility import IVCalculator
from src.visualization.surface_plot import SurfacePlotter, SurfaceData
from src.config.config import (
    UIConfig, 
    MarketDataConfig, 
    ModelConfig, 
    StatisticsConfig
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

def calculate_ivs(options_df: pd.DataFrame, risk_free_rate: float,
                 dividend_yield_override: Optional[float] = None) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Calculate IVs with proper progress tracking.
    
    Args:
        options_df: DataFrame containing option data
        risk_free_rate: Risk-free rate in decimal form
        dividend_yield_override: Dividend yield in decimal form to use for every option.
            If None (the default), each option's own ``q`` column is used, which holds the
            yield looked up for the ticker.
        
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
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for idx, (_, row) in enumerate(options_df.iterrows()):
        # Fixed progress calculation
        current_progress = (idx + 1) / total_options
        progress_text.text(f"Calculating IV for option {idx + 1} of {total_options}")
        progress_bar.progress(current_progress)
        
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
            continue
    
    progress_bar.empty()
    progress_text.empty()
    
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

    @property
    def risk_free_decimal(self) -> float:
        return self.risk_free_rate / 100

    @property
    def dividend_override(self) -> Optional[float]:
        """The dividend yield to use instead of the looked-up one, as a decimal, or None."""
        return self.dividend_yield / 100 if self.override_dividend else None

    def session_summary(self) -> Dict[str, Any]:
        """The record kept in ``st.session_state.current_params``."""
        return {
            'ticker': self.ticker,
            'min_strike_pct': self.min_strike_pct,
            'max_strike_pct': self.max_strike_pct,
            'min_open_interest': self.min_open_interest,
            'risk_free_rate': self.risk_free_rate,
            'dividend_override_pct': self.dividend_yield if self.override_dividend else None,
            'theme': self.theme,
            'colormap': self.colormap,
            'y_axis_type': self.y_axis_type
        }


def render_sidebar() -> Tuple[AnalysisParams, bool]:
    """Draw the sidebar; return the chosen parameters and whether the button was clicked."""
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
        
        generate = st.button("Generate Analysis", use_container_width=True)

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
    return params, generate


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
    fig = plotter.add_smile_slices(fig, theme=params.theme.lower())
    return fig, list(plotter.skipped_slice_days)


def run_analysis(params: AnalysisParams) -> Dict[str, Any]:
    """Fetch, select out-of-the-money options, solve the IVs and build the figure."""
    st.info("Fetching market data...")
    
    options_df = fetch_options(params)
    
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
    st.success(
        f"Market data fetched - {params.ticker} @ ${spot_price:.2f} | "
        f"dividend yield {used_dividend * 100:.2f}% ({dividend_source})"
    )
    
    st.info("Computing implied volatilities...")
    ivs, valid_options = calculate_ivs(options_df, params.risk_free_decimal, dividend_override)
    
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


def render_results(results: Dict[str, Any]) -> None:
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
    
    params, generate = render_sidebar()

    # Remember that the button was clicked
    if generate:
        st.session_state.should_generate = True
        st.session_state.current_params = params.session_summary()
    
    # Run when the button was clicked, and also on first load while there are no saved results
    if st.session_state.get('should_generate', False) or 'analysis_results' not in st.session_state:
        
        status = st.empty()
        
        try:
            with status.container():
                try:
                    st.session_state.analysis_results = run_analysis(params)
                except AnalysisError as e:
                    st.error(str(e))
                    return
                st.session_state.should_generate = False
            
            status.empty()
            
        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")
            return
    
    # Display results if available (either just generated or from previous generation)
    if 'analysis_results' in st.session_state:
        render_results(st.session_state.analysis_results)


if __name__ == "__main__":
    main()