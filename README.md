# Volatility Surface Analyzer

A Python tool for fetching, analyzing, and visualizing implied volatility surfaces from market data. Built with Streamlit, this application provides an interactive interface for exploring option market dynamics through volatility surface analysis.

## Features

- **Real-time Market Data**: Fetches live option chain data using yfinance
- **Advanced Analytics**: 
  - Implied Volatility (IV) calculation using Black-Scholes model
  - Surface interpolation and smile analysis
  - Term structure visualization
  
- **Interactive Visualization**:
  - 3D volatility surface plots
  - Customizable themes and color schemes
  
- **Quantitative Analysis**:
  - At-the-money (ATM) IV tracking
  - Skew measurement
  - Term structure analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/volatility-surface.git
cd volatility-surface
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

2. Configure analysis parameters in the sidebar:
   - Enter ticker symbol (e.g., "SPY", "AAPL")
   - Adjust strike price range
   - Set minimum option volume
   - Modify risk-free rate and dividend yield
   - Choose visualization preferences

3. Click "Generate Analysis" to create the volatility surface visualization

## Project Structure

```
├── src/
│   ├── calculators/
│   │   ├── black_scholes.py    # Black-Scholes option pricing
│   │   └── implied_volatility.py # IV calculation
│   ├── data/
│   │   └── market_data.py      # Option data fetching
│   └── visualization/
│       └── surface_plot.py     # Surface plotting utilities
├── tests/
│   └── test_calculators/
│       ├── test_black_scholes.py
│       └── test_implied_volatility.py
├── examples/
│   ├── fetch_data.py           # Data fetching example
│   └── visualize_surface.py    # Basic visualization example
└── streamlit_app.py            # Main application
```

## Development

### Running Tests
```bash
pytest tests/
```

### Adding New Features
1. Create a feature branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make changes and run tests
3. Submit a pull request

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built using [yfinance](https://github.com/ranaroussi/yfinance) for market data
- Visualization powered by [Plotly](https://plotly.com/)
- UI built with [Streamlit](https://streamlit.io/)
