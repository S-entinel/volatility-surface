import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from src.data.market_data import OptionDataFetcher
from src.calculators.implied_volatility import IVCalculator
from src.visualization.surface_plot import SurfacePlotter, SurfaceData

def main():
    # Fetch option data
    fetcher = OptionDataFetcher('SPY')
    options_df = fetcher.prepare_for_iv()
    
    # Initialize IV calculator
    iv_calc = IVCalculator()
    
    # Get spot price
    spot_price = options_df['S'].iloc[0]
    
    # Filter options
    filtered_df = options_df[
        (options_df['days_to_expiry'] >= 7) & 
        (options_df['days_to_expiry'] <= 365) &
        (options_df['type'] == 'call') &
        (options_df['strike'].between(spot_price * 0.8, spot_price * 1.2))
    ]
    
    # Calculate IVs
    ivs = []
    valid_options = []
    
    for _, row in filtered_df.iterrows():
        iv = iv_calc.calculate_iv(
            S=row['S'],
            K=row['strike'],
            T=row['T'],
            r=row['r'],
            market_price=row['price'],
            option_type=row['type']
        )
        if iv is not None:
            ivs.append(iv)
            valid_options.append(row)
    
    # Prepare surface data
    surface_data = SurfaceData(
        strikes=np.array([opt['strike'] for opt in valid_options]),
        expiries=np.array([opt['T'] for opt in valid_options]),
        ivs=np.array(ivs),
        spot_price=spot_price
    )
    
    # Create plotter and generate surface
    plotter = SurfacePlotter(surface_data)
    fig = plotter.create_surface_plot()
    
    # Add smile slices
    fig = plotter.add_smile_slices(fig)
    
    # Save interactive plot
    fig.write_html("volatility_surface.html")
    print(f"Surface plot saved to volatility_surface.html")
    print(f"Spot price: ${spot_price:.2f}")
    print(f"Number of valid IVs: {len(ivs)}")

if __name__ == "__main__":
    main()