import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.market_data import OptionDataFetcher
from src.calculators.implied_volatility import IVCalculator
import pandas as pd

def main():
    # Initialize fetcher for a liquid stock/ETF
    fetcher = OptionDataFetcher('SPY')
    
    # Get formatted option data
    options_df = fetcher.prepare_for_iv()
    
    # Initialize IV calculator
    iv_calc = IVCalculator()
    
    print(f"Total options available: {len(options_df)}")
    
    # Get spot price
    spot_price = options_df['S'].iloc[0]
    print(f"\nCurrent SPY price: ${spot_price:.2f}")
    
    # Group options by expiration and get a sample for each month
    options_df['days_to_expiry'] = options_df['T'] * 365
    
    # Filter for reasonable expiration periods (7 days to 1 year)
    filtered_options = options_df[
        (options_df['days_to_expiry'] >= 7) & 
        (options_df['days_to_expiry'] <= 365) &
        (options_df['type'] == 'call') &
        (options_df['strike'].between(spot_price * 0.95, spot_price * 1.05))  # Near-the-money
    ]
    
    # Get a sample across different expiration periods
    expiry_groups = [
        ('1 week - 1 month', 7, 30),
        ('1-3 months', 31, 90),
        ('3-6 months', 91, 180),
        ('6-12 months', 181, 365)
    ]
    
    for period_name, min_days, max_days in expiry_groups:
        print(f"\n{period_name} options:")
        period_options = filtered_options[
            filtered_options['days_to_expiry'].between(min_days, max_days)
        ].sort_values('days_to_expiry').head(3)
        
        for _, row in period_options.iterrows():
            try:
                iv = iv_calc.calculate_iv(
                    S=row['S'],
                    K=row['strike'],
                    T=row['T'],
                    r=row['r'],
                    market_price=row['price'],
                    option_type=row['type']
                )
                if iv is not None:
                    print(f"Strike: ${row['strike']:<7.1f} | Days to Expiry: {row['days_to_expiry']:>3.0f} | IV: {iv:.1%}")
                else:
                    print(f"Strike: ${row['strike']:<7.1f} | Days to Expiry: {row['days_to_expiry']:>3.0f} | IV: Could not calculate")
            except Exception as e:
                print(f"Error processing option: {e}")

if __name__ == "__main__":
    main()