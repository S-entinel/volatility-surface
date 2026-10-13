"""
Central configuration for Implied Volatility Surface Explorer.

All magic numbers, default values, and thresholds are defined here
for easy maintenance and modification.

Type hints are used throughout to ensure type safety and improve
IDE support for autocomplete and error detection.
"""

from typing import List


class MarketDataConfig:
    """Configuration for market data fetching and filtering."""
    
    # Strike price range filters (as percentages of spot)
    DEFAULT_MIN_STRIKE_PCT: float = 75.0
    DEFAULT_MAX_STRIKE_PCT: float = 125.0
    
    # Liquidity filtering
    DEFAULT_MIN_OPEN_INTEREST: int = 0
    MIN_OPEN_INTEREST_THRESHOLD: int = 0  # Absolute minimum

    # Quote quality filtering
    MAX_RELATIVE_SPREAD: float = 0.5   # (ask - bid) / mid must not exceed 50%
    MIN_QUOTE_MID_PRICE: float = 0.05  # Below this, tick size dominates the implied volatility
    
    # Time to expiration filters
    MIN_DAYS_TO_EXPIRY: int = 7  # Exclude options expiring within a week
    
    # Data quality thresholds
    MIN_VALID_OPTIONS: int = 10  # Minimum options needed for analysis
    

class ModelConfig:
    """Configuration for Black-Scholes model parameters."""
    
    # Default risk-free rate (as decimal, e.g., 0.042 = 4.2%)
    DEFAULT_RISK_FREE_RATE: float = 0.042
    
    # Default dividend yield (as decimal, e.g., 0.01 = 1%)
    DEFAULT_DIVIDEND_YIELD: float = 0.01
    
    # Parameter bounds for validation
    MIN_RISK_FREE_RATE: float = 0.0
    MAX_RISK_FREE_RATE: float = 1.0  # 100%
    MIN_DIVIDEND_YIELD: float = 0.0
    MAX_DIVIDEND_YIELD: float = 1.0  # 100%


class IVCalculationConfig:
    """Configuration for implied volatility calculation."""
    
    # Convergence parameters
    IV_MIN_BOUND: float = 1e-6  # Minimum volatility (near zero)
    IV_MAX_BOUND: float = 5.0   # Maximum volatility (500%)
    IV_CONVERGENCE_TOLERANCE: float = 1e-8  # Absolute tolerance on sigma passed to brentq (xtol)
    
    # Intrinsic value tolerance (for arbitrage detection)
    INTRINSIC_VALUE_TOLERANCE: float = 0.99
    
    # Valid ranges for input validation
    MIN_SPOT_PRICE: float = 0.01
    MIN_STRIKE_PRICE: float = 0.01
    MIN_TIME_TO_EXPIRY: float = 1e-6  # Essentially zero
    MIN_MARKET_PRICE: float = 0.01


class VisualizationConfig:
    """Configuration for 3D surface plotting."""
    
    # Mesh resolution
    MESH_GRID_SIZE: int = 50  # Number of points in each dimension
    
    # Plot dimensions
    DEFAULT_PLOT_WIDTH: int = 900
    DEFAULT_PLOT_HEIGHT: int = 800
    
    # Lighting parameters
    LIGHTING_AMBIENT: float = 0.6
    LIGHTING_DIFFUSE: float = 0.8
    LIGHTING_FRESNEL: float = 0.0
    LIGHTING_SPECULAR: float = 0.1
    LIGHTING_ROUGHNESS: float = 0.9


class StatisticsConfig:
    """Configuration for IV statistics calculation."""
    
    # Display formatting
    IV_DISPLAY_MULTIPLIER: float = 100.0  # Convert to percentage


class UIConfig:
    """Configuration for Streamlit UI defaults."""
    
    # Input widget defaults
    DEFAULT_TICKER: str = "SPY"
    
    # Number input bounds
    MIN_STRIKE_PCT_LOWER: float = 50.0
    MAX_STRIKE_PCT_UPPER: float = 200.0
    
    RISK_FREE_RATE_MIN: float = 0.0
    RISK_FREE_RATE_MAX: float = 25.0
    RISK_FREE_RATE_STEP: float = 0.1
    
    DIVIDEND_YIELD_MIN: float = 0.0
    DIVIDEND_YIELD_MAX: float = 25.0
    DIVIDEND_YIELD_STEP: float = 0.1
    
    # Theme options
    AVAILABLE_THEMES: List[str] = ["Dark", "Light"]
    AVAILABLE_Y_AXIS_TYPES: List[str] = ["Strike Price ($)", "Moneyness"]
    
    # Cache settings
    CACHE_TTL_SECONDS: int = 900  # 15 minutes