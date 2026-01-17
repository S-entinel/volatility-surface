"""
Central configuration for Volatility Surface Analyser.

All magic numbers, default values, and thresholds are defined here
for easy maintenance and modification.

Type hints are used throughout to ensure type safety and improve
IDE support for autocomplete and error detection.
"""

from typing import Dict, List, Type


class MarketDataConfig:
    """Configuration for market data fetching and filtering."""
    
    # Strike price range filters (as percentages of spot)
    DEFAULT_MIN_STRIKE_PCT: float = 75.0
    DEFAULT_MAX_STRIKE_PCT: float = 125.0
    
    # Volume filtering
    DEFAULT_MIN_VOLUME: int = 10
    MIN_VOLUME_THRESHOLD: int = 0  # Absolute minimum
    
    # Time to expiration filters
    MIN_DAYS_TO_EXPIRY: int = 7  # Exclude options expiring within a week
    
    # Data quality thresholds
    MIN_VALID_OPTIONS: int = 10  # Minimum options needed for analysis
    

class ModelConfig:
    """Configuration for Black-Scholes model parameters."""
    
    # Default risk-free rate (as decimal, e.g., 0.015 = 1.5%)
    DEFAULT_RISK_FREE_RATE: float = 0.015
    
    # Default dividend yield (as decimal, e.g., 0.013 = 1.3%)
    DEFAULT_DIVIDEND_YIELD: float = 0.013
    
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
    IV_CONVERGENCE_TOLERANCE: float = 1e-4
    
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
    
    # Camera positioning
    CAMERA_EYE_X: float = 2.2
    CAMERA_EYE_Y: float = -2.2
    CAMERA_EYE_Z: float = 1.5
    CAMERA_CENTER_Z: float = -0.2
    
    # Lighting parameters
    LIGHTING_AMBIENT: float = 0.6
    LIGHTING_DIFFUSE: float = 0.8
    LIGHTING_FRESNEL: float = 0.0
    LIGHTING_SPECULAR: float = 0.1
    LIGHTING_ROUGHNESS: float = 0.9
    
    # Volatility smile slice days
    DEFAULT_SMILE_DAYS: List[int] = [30, 60, 90]
    
    # Line styling
    SMILE_LINE_WIDTH: int = 3


class StatisticsConfig:
    """Configuration for IV statistics calculation."""
    
    # ATM (At-The-Money) definition
    ATM_MONEYNESS_LOWER: float = 0.98  # 98% of spot
    ATM_MONEYNESS_UPPER: float = 1.02  # 102% of spot
    
    # Skew calculation bounds
    OTM_PUT_MONEYNESS: float = 0.95   # 95% of spot
    OTM_CALL_MONEYNESS: float = 1.05  # 105% of spot
    
    # Display formatting
    IV_DISPLAY_MULTIPLIER: float = 100.0  # Convert to percentage


class UIConfig:
    """Configuration for Streamlit UI defaults."""
    
    # Input widget defaults
    DEFAULT_TICKER: str = "SPY"
    
    # Number input bounds
    MIN_STRIKE_PCT_LOWER: float = 50.0
    MIN_STRIKE_PCT_UPPER: float = 99.0
    MAX_STRIKE_PCT_LOWER: float = 101.0
    MAX_STRIKE_PCT_UPPER: float = 200.0
    
    RISK_FREE_RATE_MIN: float = 0.0
    RISK_FREE_RATE_MAX: float = 25.0
    RISK_FREE_RATE_STEP: float = 0.1
    
    DIVIDEND_YIELD_MIN: float = 0.0
    DIVIDEND_YIELD_MAX: float = 25.0
    DIVIDEND_YIELD_STEP: float = 0.1
    
    # Theme options
    AVAILABLE_THEMES: List[str] = ["Dark", "Light"]
    AVAILABLE_COLORMAPS: List[str] = ["Hot", "Viridis", "Plasma", "Blues", "Rainbow", "Greyscale"]
    AVAILABLE_Y_AXIS_TYPES: List[str] = ["Strike Price ($)", "Moneyness"]
    
    # Cache settings
    CACHE_TTL_SECONDS: int = 300  # 5 minutes


class LoggingConfig:
    """Configuration for logging behavior."""
    
    # Log levels
    DEFAULT_LOG_LEVEL: str = "INFO"
    
    # Date format
    LOG_DATE_FORMAT: str = '%Y-%m-%d %H:%M:%S'
    
    # Message format
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# Convenience function to get all configs as a dict
def get_all_configs() -> Dict[str, Type]:
    """
    Get all configuration classes as a dictionary.
    
    Returns:
        Dictionary mapping config names to config classes
    
    Example:
        >>> configs = get_all_configs()
        >>> configs['market_data'].DEFAULT_MIN_STRIKE_PCT
        75.0
    """
    return {
        'market_data': MarketDataConfig,
        'model': ModelConfig,
        'iv_calculation': IVCalculationConfig,
        'visualization': VisualizationConfig,
        'statistics': StatisticsConfig,
        'ui': UIConfig,
        'logging': LoggingConfig
    }