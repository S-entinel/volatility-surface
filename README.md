# Volatility Surface Analyzer

A professional-grade tool for analyzing and visualizing implied volatility surfaces from options market data.

## Features

- **Real-Time Market Data** - Live option chains via yfinance
- **3D Volatility Surface** - Interactive surface plots with multiple themes
- **Advanced Analytics** - ATM IV, skew, and term structure calculations
- **Export Capabilities** - Download analysis as CSV

## Live Demo

🔗 **[View Live App](https://volatility-surface-eompm5s2dtuksyhw7z5fea.streamlit.app)**

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/volatility-surface.git
cd volatility-surface

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
```

## Usage

1. Enter a ticker symbol (e.g., SPY, AAPL, TSLA)
2. Adjust parameters in the sidebar
3. Click "Generate Analysis"
4. Explore the interactive 3D surface

## Tech Stack

- **Python 3.8+**
- **Streamlit** - Web interface
- **yfinance** - Market data
- **Plotly** - 3D visualization
- **SciPy** - Implied volatility calculation (Brent's method)

## Project Structure

```
├── src/
│   ├── calculators/     # Black-Scholes & IV calculation
│   ├── data/            # Market data fetching
│   └── visualization/   # Surface plotting
├── tests/               # Unit tests
└── streamlit_app.py     # Main application
```

## License

MIT License - see [LICENSE](LICENSE) for details
