"""
Streamlit application for Implied Volatility Surface Analysis.

Main application with complete type hints for all functions and proper
return type annotations.
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time
from typing import Tuple, List, Dict, Optional, Any
from src.data.market_data import OptionDataFetcher
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
                 dividend_yield: float) -> Tuple[List[float], List[Dict[str, Any]]]:
    """
    Calculate IVs with proper progress tracking.
    
    Args:
        options_df: DataFrame containing option data
        risk_free_rate: Risk-free rate in decimal form
        dividend_yield: Dividend yield in decimal form
        
    Returns:
        Tuple of (list of IVs, list of valid option dictionaries)
        
    Example:
        >>> ivs, valid_options = calculate_ivs(df, 0.045, 0.013)
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
                q=dividend_yield,   # Already in decimal form
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

def main() -> None:
    """
    Main Streamlit application entry point.
    
    Sets up the UI, handles user interactions, and orchestrates
    data fetching, IV calculation, and visualization.
    
    Returns:
        None
    """
    st.set_page_config(
        page_title="IV Surface Analysis",
        page_icon="■",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_style()
    
    st.title("Implied Volatility Surface Analysis")
    
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
                
            min_volume = st.number_input(
                "Min Option Volume",
                min_value=MarketDataConfig.MIN_VOLUME_THRESHOLD,
                value=MarketDataConfig.DEFAULT_MIN_VOLUME
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

            dividend_yield = st.number_input(
                "Dividend Yield (%)",
                min_value=UIConfig.DIVIDEND_YIELD_MIN,
                max_value=UIConfig.DIVIDEND_YIELD_MAX,
                value=ModelConfig.DEFAULT_DIVIDEND_YIELD * 100,  # Convert to percentage
                step=UIConfig.DIVIDEND_YIELD_STEP,
                format="%.1f",
                help="Annual dividend yield (e.g., 1.3 for 1.3%)"
            )
        
        with st.expander("Visualization Settings", expanded=True):
            theme = st.selectbox(
                "Theme",
                options=UIConfig.AVAILABLE_THEMES
            )
            
            colormap = st.selectbox(
                "Color Scheme",
                options=UIConfig.AVAILABLE_COLORMAPS
            )
            
            y_axis_type = st.selectbox(
                "Y-Axis Scale",
                options=UIConfig.AVAILABLE_Y_AXIS_TYPES
            )
        
        generate = st.button("Generate Analysis", use_container_width=True)

    # Auto-generate on page load or when button clicked
    if generate or 'generated' not in st.session_state:
        st.session_state.generated = True
        
        status = st.empty()
        
        try:
            with status.container():
                st.info("Fetching market data...")
                
                # Convert percentages to decimals
                risk_free_decimal = risk_free_rate / 100
                dividend_decimal = dividend_yield / 100
                
                fetcher = OptionDataFetcher(ticker)
                options_df = fetcher.prepare_for_iv(
                    min_strike_pct=min_strike_pct,
                    max_strike_pct=max_strike_pct,
                    min_volume=min_volume,
                    risk_free_rate=risk_free_decimal  # Pass the correct rate
                )
                
                if options_df.empty:
                    st.error("No valid options data found.")
                    return
                
                # Check against minimum threshold from config
                if len(options_df) < MarketDataConfig.MIN_VALID_OPTIONS:
                    st.error(f"Insufficient options data. Found {len(options_df)}, need at least {MarketDataConfig.MIN_VALID_OPTIONS}.")
                    return
                
                spot_price = options_df['S'].iloc[0]
                st.success(f"Market data fetched - {ticker} @ ${spot_price:.2f}")
                
                st.info("Computing implied volatilities...")
                ivs, valid_options = calculate_ivs(options_df, risk_free_decimal, dividend_decimal)
                
                if len(ivs) < MarketDataConfig.MIN_VALID_OPTIONS:
                    st.error(f"Insufficient valid options for analysis. Found only {len(ivs)} valid IVs.")
                    return
                
                strikes = np.array([opt['strike']/spot_price for opt in valid_options]) if y_axis_type == "Moneyness" \
                         else np.array([opt['strike'] for opt in valid_options])
                
                surface_data = SurfaceData(
                    strikes=strikes,
                    expiries=np.array([opt['T'] for opt in valid_options]),
                    ivs=np.array(ivs),
                    spot_price=spot_price,
                    y_axis_type=y_axis_type.split()[0]
                )
                
                plotter = SurfacePlotter(surface_data)
                fig = plotter.create_surface_plot(theme=theme.lower(), colormap=colormap)
                fig = plotter.add_smile_slices(fig, theme=theme.lower())
            
            status.empty()
            
            # Calculate IV statistics
            iv_stats = calculate_iv_statistics(valid_options, ivs, spot_price)
            
            # Main visualization
            st.plotly_chart(fig, use_container_width=True)
            
            # Metrics in clean table format below visualization
            st.markdown("---")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Spot Price", f"${spot_price:.2f}")
                st.metric("ATM IV", f"{iv_stats['atm_iv']:.1f}%")
            
            with col2:
                st.metric("Valid Options", f"{len(ivs)}")
                if iv_stats['iv_skew'] is not None:
                    st.metric("IV Skew", f"{iv_stats['iv_skew']:.1f}%")
                else:
                    st.metric("IV Skew", "N/A")
            
            with col3:
                st.metric("Term Structure", f"{iv_stats['term_structure']:.1f}%")
                st.metric("IV Range", f"{iv_stats['iv_min']:.1f}% - {iv_stats['iv_max']:.1f}%")
            
            with col4:
                st.metric("Avg IV", f"{iv_stats['avg_iv']:.1f}%")
                st.metric("IV Std Dev", f"{iv_stats['iv_std']:.1f}%")
            
            # Export section
            st.markdown("---")
            col_export1, col_export2, col_export3 = st.columns([1, 1, 2])
            
            with col_export1:
                if len(valid_options) > 0:
                    df_download = pd.DataFrame({
                        'Strike': [opt['strike'] for opt in valid_options],
                        'Days_to_Expiry': [opt['days_to_expiry'] for opt in valid_options],
                        'IV': [iv * 100 for iv in ivs],
                        'Option_Type': [opt['type'] for opt in valid_options],
                        'Moneyness': [opt['strike']/spot_price for opt in valid_options]
                    })
                    
                    st.download_button(
                        "Export Data (CSV)",
                        data=df_download.to_csv(index=False).encode('utf-8'),
                        file_name=f'{ticker}_iv_analysis_{datetime.now().strftime("%Y%m%d")}.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
            
        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")
            

if __name__ == "__main__":
    main()