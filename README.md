# Implied Volatility Surface Explorer

A web app that builds and visualises implied volatility surfaces from live equity option chains.

**[→ Live Demo](https://volatility-surface-eompm5s2dtuksyhw7z5fea.streamlit.app)**

[![Tests](https://github.com/S-entinel/volatility-surface/actions/workflows/ci.yml/badge.svg)](https://github.com/S-entinel/volatility-surface/actions/workflows/ci.yml)

<!-- Screenshots: add images to docs/ and link them here -->
<!-- ![Surface](docs/surface.png) -->

## What it does

Enter a ticker (SPY, QQQ, AAPL...) and the app fetches the option chain, solves implied volatilities and shows:

- **Key figures:** 30-day ATM implied volatility, 30-day implied move, 30-day 25-delta risk reversal, and term structure (90-day minus 30-day ATM volatility)
- **Surface tab:** a 3D surface or a heatmap, built either by interpolating the quotes or from a per-expiry SVI fit with arbitrage checks
- **Smile & Skew tab:** the smile for one expiry, with its ATM volatility, risk reversal and SVI fit
- **Data & Export tab:** every IV point in a table, with CSV export and an interactive HTML copy of the chart

The strike axis can be shown as strike price or as moneyness, ln(strike / forward), which puts every expiry on the same footing.

## How it works

1. **Data.** Option chains come from Yahoo Finance. Prices are the bid-ask midpoint when there is a live quote, otherwise the last traded price.
2. **Filtering.** Contracts are dropped if they have no usable price, a mid below $0.05, a bid-ask spread above 50% of the mid, or fewer than 7 days to expiry. The strike range and minimum open interest are set in the sidebar.
3. **Out-of-the-money selection.** Puts are used below the forward price and calls above it, where they are most liquid and least affected by early exercise.
4. **Implied volatility.** Each price is inverted through Black-Scholes with Brent's method, after checking it against no-arbitrage bounds.
5. **SVI fit.** An SVI curve is fitted to each expiry in log-moneyness. Fits are checked for butterfly arbitrage, and neighbouring expiries are checked for calendar arbitrage. The checks are reported, not enforced.
6. **Key figures.** Read from the SVI curves at the forward price, blended between the two nearest expiries.

## Quick start

```bash
git clone https://github.com/S-entinel/volatility-surface.git
cd volatility-surface

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Then open `http://localhost:8501`. The analysis runs automatically and updates as you change the sidebar.


## Project structure

```
volatility-surface/
├── src/
│   ├── calculators/      # Black-Scholes pricing and IV solver
│   ├── data/             # Market data fetching and OTM selection
│   ├── visualization/    # 3D surface plotting
│   ├── config/           # Central configuration
│   ├── utils/            # Logging
│   ├── analysis.py       # Pipeline: filter, solve, build figures and key figures
│   └── svi.py            # SVI fitting and arbitrage checks
├── tests/
├── streamlit_app.py      # Web interface
├── requirements.txt
└── requirements-dev.txt
```

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

Built with [Streamlit](https://streamlit.io), [Plotly](https://plotly.com) and [yfinance](https://github.com/ranaroussi/yfinance).