# Volatility Surface Analyser

A web application for analysing and visualising implied volatility surfaces from live options market data.

## Overview

The Volatility Surface Analyser fetches options data and computes implied volatilities using the Black-Scholes model with Brent's numerical method. The interactive 3D surface visualisation helps traders and analysts understand volatility patterns across strikes and expirations.

**[→ Live Demo](https://volatility-surface-eompm5s2dtuksyhw7z5fea.streamlit.app)**

## Features

- **Market Data**: Option chains via Yahoo Finance (delayed, unofficial feed)
- **Interactive 3D Visualisation**: Rotatable volatility surface with customisable themes
- **Analytics**: ATM IV, skew, term structure, and surface statistics
- **Engineering**: Type hints throughout, modular architecture, pytest suite
- **Export**: Download analysis data as CSV

## Quick Start

```bash
# Clone repository
git clone https://github.com/S-entinel/volatility-surface.git
cd volatility-surface

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

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
   - Visualisation theme and colourmap
3. Click "Generate Analysis"
4. Explore the interactive 3D surface and statistics

## Technology Stack

- **Python 3.11**
- **Streamlit** - Web interface
- **yfinance** - Market data
- **Plotly** - 3D visualisation
- **SciPy** - Implied volatility calculation
- **NumPy & Pandas** - Data processing

## Project Structure

```
volatility-surface/
├── src/
│   ├── calculators/      # Black-Scholes & IV calculations
│   ├── data/             # Market data fetching
│   ├── visualization/    # 3D surface plotting
│   ├── config/           # Centralised configuration
│   └── utils/            # Logging utilities
├── tests/                # Test suite
├── streamlit_app.py      # Main application
├── requirements.txt      # Runtime dependencies
└── requirements-dev.txt  # Test dependencies
```

## Development

```bash
# Install runtime + test dependencies
pip install -r requirements-dev.txt

# Run all tests with coverage
pytest

# Run specific test categories
pytest -m unit
pytest -m integration
```

## Configuration

Customise defaults in `src/config/config.py`:

- Strike price ranges
- Risk-free rate and dividend yield
- Visualisation settings
- Data quality thresholds

## Known Limitations

- Yahoo Finance data is delayed and unofficial; it may be stale or unavailable.
- Black-Scholes assumes European exercise; many listed equity options are American-style.
- Risk-free rate and dividend yield are user-supplied constants, not term-structure inputs.

These are being addressed in ongoing work.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with [Streamlit](https://streamlit.io) and powered by [yfinance](https://github.com/ranaroussi/yfinance).