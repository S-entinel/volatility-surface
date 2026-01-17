# Volatility Surface Analyzer

A professional web application for analyzing and visualizing implied volatility surfaces from live options market data.

## Overview

The Volatility Surface Analyzer fetches real-time options data and computes implied volatilities using the Black-Scholes model with Brent's numerical method. The interactive 3D surface visualization helps traders and analysts understand volatility patterns across strikes and expirations.

**[→ Live Demo](https://volatility-surface-eompm5s2dtuksyhw7z5fea.streamlit.app)**

## Features

- **Real-Time Data**: Live option chains via Yahoo Finance
- **Interactive 3D Visualization**: Rotatable volatility surface with customizable themes
- **Comprehensive Analytics**: ATM IV, skew, term structure, and surface statistics
- **Professional Implementation**: 93% test coverage, type hints throughout, modular architecture
- **Export Capabilities**: Download analysis data as CSV

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/volatility-surface.git
cd volatility-surface

# Install dependencies
pip install -r requirements.txt

# Run application
streamlit run streamlit_app.py
```

Visit `http://localhost:8501` to use the application.

## Usage

1. Enter a ticker symbol (e.g., SPY, AAPL, TSLA)
2. Adjust parameters in the sidebar:
   - Strike price range
   - Risk-free rate and dividend yield
   - Visualization theme and colormap
3. Click "Generate Analysis"
4. Explore the interactive 3D surface and statistics

## Technology Stack

- **Python 3.11**
- **Streamlit** - Web interface
- **yfinance** - Market data
- **Plotly** - 3D visualization
- **SciPy** - Implied volatility calculation
- **NumPy & Pandas** - Data processing

## Project Structure

```
volatility-surface/
├── src/
│   ├── calculators/      # Black-Scholes & IV calculations
│   ├── data/             # Market data fetching
│   ├── visualization/    # 3D surface plotting
│   ├── config/           # Centralized configuration
│   └── utils/            # Logging utilities
├── tests/                # Comprehensive test suite
├── streamlit_app.py      # Main application
└── requirements.txt      # Dependencies
```

## Testing

```bash
# Run all tests with coverage
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
```

Current test coverage: **93%**

## Configuration

Customize defaults in `src/config/config.py`:
- Strike price ranges
- Risk-free rate and dividend yield
- Visualization settings
- Data quality thresholds

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with [Streamlit](https://streamlit.io) and powered by [yfinance](https://github.com/ranaroussi/yfinance).

---
