import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time
from src.data.market_data import OptionDataFetcher
from src.calculators.implied_volatility import IVCalculator
from src.visualization.surface_plot import SurfacePlotter, SurfaceData

# Custom CSS styling for professional look
def apply_custom_style():
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

def calculate_ivs(options_df, risk_free_rate, dividend_yield):
    """Calculate IVs with proper progress tracking"""
    iv_calc = IVCalculator()
    ivs = []
    valid_options = []
    total_options = len(options_df)
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for idx, row in enumerate(options_df.iterrows()):
        current_progress = min(idx / (total_options - 1), 1.0) if total_options > 1 else 1.0
        progress_text.text(f"Calculating IV for option {idx + 1} of {total_options}")
        progress_bar.progress(current_progress)
        
        try:
            iv = iv_calc.calculate_iv(
                S=row[1]['S'],
                K=row[1]['strike'],
                T=row[1]['T'],
                r=risk_free_rate/100,  # Convert from percentage
                q=dividend_yield/100,   # Convert from percentage
                market_price=row[1]['price'],
                option_type=row[1]['type']
            )
            
            if iv is not None:
                ivs.append(iv)
                valid_options.append(row[1])
        except Exception:
            continue
    
    progress_bar.empty()
    progress_text.empty()
    
    return ivs, valid_options

def calculate_iv_statistics(valid_options, ivs, spot_price):
    """Calculate key IV statistics for quant research"""
    df = pd.DataFrame({
        'strike': [opt['strike'] for opt in valid_options],
        'T': [opt['T'] for opt in valid_options],
        'iv': ivs,
        'moneyness': [opt['strike']/spot_price for opt in valid_options]
    })
    
    # ATM IV (using closest to 1.0 moneyness)
    atm_idx = abs(df['moneyness'] - 1.0).idxmin()
    atm_iv = df.loc[atm_idx, 'iv']
    
    # IV Skew
    df_nearest = df[df['T'] == df['T'].min()]
    otm_put = df_nearest[df_nearest['moneyness'] < 0.95]['iv'].mean()
    otm_call = df_nearest[df_nearest['moneyness'] > 1.05]['iv'].mean()
    skew = otm_put - otm_call if (not np.isnan(otm_put) and not np.isnan(otm_call)) else None
    
    # Term Structure
    df['atm_dist'] = abs(df['moneyness'] - 1.0)
    short_term = df[df['T'] == df['T'].min()]
    long_term = df[df['T'] == df['T'].max()]
    
    short_atm = short_term.loc[short_term['atm_dist'].idxmin(), 'iv']
    long_atm = long_term.loc[long_term['atm_dist'].idxmin(), 'iv']
    term_structure = long_atm - short_atm
    
    return {
        'atm_iv': atm_iv * 100,
        'iv_skew': skew * 100 if skew is not None else None,
        'term_structure': term_structure * 100
    }

def main():
    st.set_page_config(
        page_title="IV Surface Analysis",
        page_icon="IV",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_style()
    
    st.title("Implied Volatility Surface Analysis")
    
    with st.sidebar:
        with st.expander("Market Data Configuration", expanded=True):
            ticker = st.text_input(
                "Ticker Symbol",
                value="SPY"
            ).upper()
            
            st.markdown("#### Strike Price Range")
            col1, col2 = st.columns(2)
            with col1:
                min_strike_pct = st.number_input(
                    "Min Strike %",
                    min_value=50.0,
                    max_value=99.0,
                    value=70.0,
                    format="%.1f"
                )
            with col2:
                max_strike_pct = st.number_input(
                    "Max Strike %",
                    min_value=min_strike_pct + 1,
                    max_value=200.0,
                    value=130.0,
                    format="%.1f"
                )
                
            min_volume = st.number_input(
                "Min Option Volume",
                min_value=0,
                value=10
            )
        
        with st.expander("Model Parameters", expanded=True):
            risk_free_rate = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=25.0,
                value=1.5,
                step=0.1,
                format="%.1f",
                help="Annual risk-free rate (e.g., 1.5 for 1.5%)"
            )

            dividend_yield = st.number_input(
                "Dividend Yield (%)",
                min_value=0.0,
                max_value=25.0,
                value=1.3,
                step=0.1,
                format="%.1f",
                help="Annual dividend yield (e.g., 1.3 for 1.3%)"
            )
        
        with st.expander("Visualization Settings", expanded=True):
            theme = st.selectbox(
                "Theme",
                options=["Dark", "Light"]
            )
            
            colormap = st.selectbox(
                "Color Scheme",
                options=["Hot", "Viridis", "Plasma", "Blues", "Rainbow", "Greyscale"]
            )
            
            y_axis_type = st.selectbox(
                "Y-Axis Scale",
                options=["Strike Price ($)", "Moneyness"]
            )
        
        generate = st.button("Generate Analysis", use_container_width=True)

    if generate:
        status = st.empty()
        
        try:
            with status.container():
                st.info("Fetching market data...")
                
                fetcher = OptionDataFetcher(ticker)
                options_df = fetcher.prepare_for_iv(
                    min_strike_pct=min_strike_pct,
                    max_strike_pct=max_strike_pct,
                    min_volume=min_volume
                )
                
                if options_df.empty:
                    st.error("No valid options data found.")
                    return
                
                spot_price = options_df['S'].iloc[0]
                st.success(f"Market data fetched - {ticker} @ ${spot_price:.2f}")
                
                st.info("Computing implied volatilities...")
                ivs, valid_options = calculate_ivs(options_df, risk_free_rate, dividend_yield)
                
                if len(ivs) < 10:
                    st.error("Insufficient valid options for analysis.")
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
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Market Overview")
                st.metric("Current Price", f"${spot_price:.2f}")
                st.metric("Valid Options", f"{len(ivs)} of {len(options_df)}")
                
                iv_stats = calculate_iv_statistics(valid_options, ivs, spot_price)
                
                st.markdown("### Volatility Analytics")
                st.metric(
                    "ATM IV",
                    f"{iv_stats['atm_iv']:.1f}%"
                )
                
                if iv_stats['iv_skew'] is not None:
                    st.metric(
                        "IV Skew",
                        f"{iv_stats['iv_skew']:.1f}%"
                    )
                else:
                    st.metric("IV Skew", "N/A")
                
                st.metric(
                    "Term Structure",
                    f"{iv_stats['term_structure']:.1f}%"
                )
                
                if len(valid_options) > 0:
                    st.markdown("---")
                    df_download = pd.DataFrame({
                        'Strike': [opt['strike'] for opt in valid_options],
                        'Days_to_Expiry': [opt['days_to_expiry'] for opt in valid_options],
                        'IV': [iv * 100 for iv in ivs],
                        'Option_Type': [opt['type'] for opt in valid_options]
                    })
                    
                    st.download_button(
                        "Export Analysis",
                        data=df_download.to_csv(index=False).encode('utf-8'),
                        file_name=f'{ticker}_iv_analysis.csv',
                        mime='text/csv',
                        use_container_width=True
                    )
            
        except Exception as e:
            st.error(f"Analysis Error: {str(e)}")

if __name__ == "__main__":
    main()
