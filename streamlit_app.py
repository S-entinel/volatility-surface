import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime
import time
from src.data.market_data import OptionDataFetcher
from src.calculators.implied_volatility import IVCalculator
from src.visualization.surface_plot import SurfacePlotter, SurfaceData

# Set page config
st.set_page_config(
    page_title="IV Surface Visualizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def calculate_ivs(options_df, risk_free_rate, dividend_yield):
    """Calculate IVs with proper progress tracking"""
    iv_calc = IVCalculator()
    ivs = []
    valid_options = []
    total_options = len(options_df)
    
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    for idx, row in enumerate(options_df.iterrows()):
        # Calculate progress as a value between 0 and 1
        current_progress = min(idx / (total_options - 1), 1.0) if total_options > 1 else 1.0
        progress_text.text(f"Calculating IV for option {idx + 1} of {total_options}")
        progress_bar.progress(current_progress)
        
        try:
            iv = iv_calc.calculate_iv(
                S=row[1]['S'],
                K=row[1]['strike'],
                T=row[1]['T'],
                r=risk_free_rate,
                q=dividend_yield,
                market_price=row[1]['price'],
                option_type=row[1]['type']
            )
            
            if iv is not None:
                ivs.append(iv)
                valid_options.append(row[1])
        except Exception as e:
            continue
    
    progress_bar.empty()
    progress_text.empty()
    
    return ivs, valid_options

def main():
    # Title
    st.title("Implied Volatility Surface")
    
    # Sidebar
    with st.sidebar:
        # Theme Selection (New!)
        theme = st.selectbox(
            "Choose Theme",
            options=["Dark", "Light"],
            index=0,  # Default to Dark theme
        )
        
        st.markdown("---")  # Separator
        
        st.header("Model Parameters")
        st.markdown("Adjust the parameters for the Black-Scholes model.")
        
        # Model Parameters
        risk_free_rate = st.number_input(
            "Risk-Free Rate (e.g., 0.015 for 1.5%)",
            min_value=0.0,
            max_value=0.25,
            value=0.015,
            step=0.001,
            format="%.4f"
        )

        dividend_yield = st.number_input(
            "Dividend Yield (e.g., 0.013 for 1.3%)",
            min_value=0.0,
            max_value=0.25,
            value=0.013,
            step=0.001,
            format="%.4f"
        )
        
        st.header("Visualization Parameters")
        y_axis_type = st.selectbox(
            "Select Y-axis:",
            options=["Strike Price ($)", "Moneyness"]
        )
        
        st.header("Ticker Symbol")
        ticker = st.text_input(
            "Enter Ticker Symbol",
            value="SPY"
        ).upper()
        
        st.header("Strike Price Filter Parameters")
        min_strike_pct = st.number_input(
            "Minimum Strike Price (% of Spot Price)",
            min_value=50.0,
            max_value=99.0,
            value=70.0,
            step=1.0,
            format="%.1f"
        )

        max_strike_pct = st.number_input(
            "Maximum Strike Price (% of Spot Price)",
            min_value=min_strike_pct + 1,
            max_value=200.0,
            value=130.0,
            step=1.0,
            format="%.1f"
        )
        
        # Volume filter
        min_volume = st.number_input(
            "Minimum Option Volume",
            min_value=0,
            value=10,
            step=1,
            help="Minimum trading volume for options to be included"
        )
        
        # Generate button
        generate = st.button("Generate Surface")

    # Main content area
    if generate:
        status_placeholder = st.empty()
        
        try:
            status_placeholder.info("Fetching option data...")
            
            # Initialize fetcher
            fetcher = OptionDataFetcher(ticker)
            
            try:
                # Get option data
                options_df = fetcher.prepare_for_iv(
                    min_strike_pct=min_strike_pct,
                    max_strike_pct=max_strike_pct,
                    min_volume=min_volume
                )
                
                if options_df.empty:
                    st.error("No options data found for this ticker.")
                    return

                spot_price = options_df['S'].iloc[0]
                status_placeholder.success(f"Current price for {ticker}: ${spot_price:.2f}")
                
            except Exception as e:
                status_placeholder.error(f"Error fetching options data: {str(e)}")
                st.info("Please verify the ticker symbol and try again.")
                return
            
            # Calculate IVs
            status_placeholder.info("Calculating implied volatilities...")
            ivs, valid_options = calculate_ivs(options_df, risk_free_rate, dividend_yield)
            
            if not ivs:
                status_placeholder.error("Could not calculate valid IVs with the current parameters.")
                return

            if len(ivs) < 10:
                status_placeholder.warning("Not enough valid options to generate a good surface. Try adjusting the parameters.")
                return
            
            # Prepare surface data
            status_placeholder.info("Generating surface plot...")
            if y_axis_type == "Moneyness":
                strikes = np.array([opt['strike']/spot_price for opt in valid_options])
            else:
                strikes = np.array([opt['strike'] for opt in valid_options])
            
            surface_data = SurfaceData(
                strikes=strikes,
                expiries=np.array([opt['T'] for opt in valid_options]),
                ivs=np.array(ivs),
                spot_price=spot_price,
                y_axis_type=y_axis_type.split()[0]  # Just take "Strike" or "Moneyness"
            )
            
            # Create and display plot
            plotter = SurfacePlotter(surface_data)
            fig = plotter.create_surface_plot(theme=theme.lower())
            fig = plotter.add_smile_slices(fig, theme=theme.lower())
            
            status_placeholder.empty()
            st.plotly_chart(fig, use_container_width=True)
            
            # Display summary statistics
            st.subheader("Summary Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Spot Price", f"${spot_price:.2f}")
            with col2:
                st.metric("Average IV", f"{np.mean(ivs)*100:.1f}%")
            with col3:
                st.metric("Valid Options", len(ivs))
            with col4:
                st.metric("Total Options", len(options_df))
            
            # Add download button for the data
            if len(valid_options) > 0:
                df_download = pd.DataFrame({
                    'Strike': [opt['strike'] for opt in valid_options],
                    'Days_to_Expiry': [opt['days_to_expiry'] for opt in valid_options],
                    'IV': [iv * 100 for iv in ivs],
                    'Option_Type': [opt['type'] for opt in valid_options]
                })
                
                st.download_button(
                    label="Download Data as CSV",
                    data=df_download.to_csv(index=False).encode('utf-8'),
                    file_name=f'{ticker}_iv_surface_data.csv',
                    mime='text/csv'
                )
            
        except Exception as e:
            st.error(f"Error: {str(e)}")
            if "yfinance" in str(e).lower():
                st.info("There might be an issue with the data source. Try again in a few moments or try a different ticker symbol.")

if __name__ == "__main__":
    main()