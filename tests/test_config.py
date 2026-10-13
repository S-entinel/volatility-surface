"""
Tests for configuration module.

Verifies all config classes have correct default values and types.
"""

import pytest
from src.config.config import (
    MarketDataConfig,
    ModelConfig,
    IVCalculationConfig,
    VisualizationConfig,
    StatisticsConfig,
    UIConfig
)


class TestMarketDataConfig:
    """Test MarketDataConfig values."""
    
    @pytest.mark.unit
    def test_strike_range_defaults(self):
        """Test strike range default values."""
        assert MarketDataConfig.DEFAULT_MIN_STRIKE_PCT == 75.0
        assert MarketDataConfig.DEFAULT_MAX_STRIKE_PCT == 125.0
        assert isinstance(MarketDataConfig.DEFAULT_MIN_STRIKE_PCT, float)
        assert isinstance(MarketDataConfig.DEFAULT_MAX_STRIKE_PCT, float)
    
    @pytest.mark.unit
    def test_volume_defaults(self):
        """Test volume filter defaults."""
        assert MarketDataConfig.DEFAULT_MIN_OPEN_INTEREST == 0
        assert MarketDataConfig.MIN_OPEN_INTEREST_THRESHOLD == 0
        assert isinstance(MarketDataConfig.DEFAULT_MIN_OPEN_INTEREST, int)
        assert MarketDataConfig.MAX_RELATIVE_SPREAD == 0.5
        assert MarketDataConfig.MIN_QUOTE_MID_PRICE == 0.05
    
    @pytest.mark.unit
    def test_expiry_defaults(self):
        """Test expiry threshold defaults."""
        assert MarketDataConfig.MIN_DAYS_TO_EXPIRY == 7
        assert isinstance(MarketDataConfig.MIN_DAYS_TO_EXPIRY, int)
    
    @pytest.mark.unit
    def test_quality_thresholds(self):
        """Test data quality thresholds."""
        assert MarketDataConfig.MIN_VALID_OPTIONS == 10
        assert isinstance(MarketDataConfig.MIN_VALID_OPTIONS, int)


class TestModelConfig:
    """Test ModelConfig values."""
    
    @pytest.mark.unit
    def test_rate_defaults(self):
        """Test default rates."""
        assert ModelConfig.DEFAULT_RISK_FREE_RATE == 0.042
        assert ModelConfig.DEFAULT_DIVIDEND_YIELD == 0.01
        assert isinstance(ModelConfig.DEFAULT_RISK_FREE_RATE, float)
        assert isinstance(ModelConfig.DEFAULT_DIVIDEND_YIELD, float)
    
    @pytest.mark.unit
    def test_rate_bounds(self):
        """Test rate validation bounds."""
        assert ModelConfig.MIN_RISK_FREE_RATE == 0.0
        assert ModelConfig.MAX_RISK_FREE_RATE == 1.0
        assert ModelConfig.MIN_DIVIDEND_YIELD == 0.0
        assert ModelConfig.MAX_DIVIDEND_YIELD == 1.0
    
    @pytest.mark.unit
    def test_rate_bounds_valid(self):
        """Test rate bounds are sensible."""
        assert ModelConfig.MIN_RISK_FREE_RATE < ModelConfig.MAX_RISK_FREE_RATE
        assert ModelConfig.MIN_DIVIDEND_YIELD < ModelConfig.MAX_DIVIDEND_YIELD
        assert ModelConfig.DEFAULT_RISK_FREE_RATE >= ModelConfig.MIN_RISK_FREE_RATE
        assert ModelConfig.DEFAULT_RISK_FREE_RATE <= ModelConfig.MAX_RISK_FREE_RATE


class TestIVCalculationConfig:
    """Test IVCalculationConfig values."""
    
    @pytest.mark.unit
    def test_solver_bounds(self):
        """Test IV solver bounds."""
        assert IVCalculationConfig.IV_MIN_BOUND == 1e-6
        assert IVCalculationConfig.IV_MAX_BOUND == 5.0
        assert IVCalculationConfig.IV_MIN_BOUND < IVCalculationConfig.IV_MAX_BOUND
    
    @pytest.mark.unit
    def test_convergence_tolerance(self):
        """Test convergence tolerance."""
        assert IVCalculationConfig.IV_CONVERGENCE_TOLERANCE == 1e-8
        assert isinstance(IVCalculationConfig.IV_CONVERGENCE_TOLERANCE, float)
    
    @pytest.mark.unit
    def test_validation_thresholds(self):
        """Test validation thresholds."""
        assert IVCalculationConfig.INTRINSIC_VALUE_TOLERANCE == 0.99
        assert IVCalculationConfig.MIN_SPOT_PRICE == 0.01
        assert IVCalculationConfig.MIN_STRIKE_PRICE == 0.01
        assert IVCalculationConfig.MIN_TIME_TO_EXPIRY == 1e-6
        assert IVCalculationConfig.MIN_MARKET_PRICE == 0.01
    
    @pytest.mark.unit
    def test_all_positive(self):
        """Test all thresholds are positive."""
        assert IVCalculationConfig.IV_MIN_BOUND > 0
        assert IVCalculationConfig.IV_MAX_BOUND > 0
        assert IVCalculationConfig.MIN_SPOT_PRICE > 0
        assert IVCalculationConfig.MIN_STRIKE_PRICE > 0
        assert IVCalculationConfig.MIN_TIME_TO_EXPIRY > 0


class TestVisualizationConfig:
    """Test VisualizationConfig values."""
    
    @pytest.mark.unit
    def test_mesh_resolution(self):
        """Test mesh grid size."""
        assert VisualizationConfig.MESH_GRID_SIZE == 50
        assert isinstance(VisualizationConfig.MESH_GRID_SIZE, int)
        assert VisualizationConfig.MESH_GRID_SIZE > 0
    
    @pytest.mark.unit
    def test_plot_dimensions(self):
        """Test plot dimensions."""
        assert VisualizationConfig.DEFAULT_PLOT_WIDTH == 900
        assert VisualizationConfig.DEFAULT_PLOT_HEIGHT == 800
        assert isinstance(VisualizationConfig.DEFAULT_PLOT_WIDTH, int)
        assert isinstance(VisualizationConfig.DEFAULT_PLOT_HEIGHT, int)
    
    @pytest.mark.unit
    def test_lighting_parameters(self):
        """Test lighting parameters are in valid ranges."""
        assert 0 <= VisualizationConfig.LIGHTING_AMBIENT <= 1
        assert 0 <= VisualizationConfig.LIGHTING_DIFFUSE <= 1
        assert 0 <= VisualizationConfig.LIGHTING_FRESNEL <= 1
        assert 0 <= VisualizationConfig.LIGHTING_SPECULAR <= 1
        assert 0 <= VisualizationConfig.LIGHTING_ROUGHNESS <= 1


class TestStatisticsConfig:
    """Test StatisticsConfig values."""
    
    @pytest.mark.unit
    def test_display_multiplier(self):
        """Test IV display multiplier."""
        assert StatisticsConfig.IV_DISPLAY_MULTIPLIER == 100.0
        assert isinstance(StatisticsConfig.IV_DISPLAY_MULTIPLIER, float)


class TestUIConfig:
    """Test UIConfig values."""
    
    @pytest.mark.unit
    def test_default_ticker(self):
        """Test default ticker."""
        assert UIConfig.DEFAULT_TICKER == "SPY"
        assert isinstance(UIConfig.DEFAULT_TICKER, str)
        assert len(UIConfig.DEFAULT_TICKER) > 0
    
    @pytest.mark.unit
    def test_strike_bounds(self):
        """Test strike percentage input bounds."""
        assert UIConfig.MIN_STRIKE_PCT_LOWER == 50.0
        assert UIConfig.MAX_STRIKE_PCT_UPPER == 200.0
    
    @pytest.mark.unit
    def test_rate_input_bounds(self):
        """Test rate input bounds."""
        assert UIConfig.RISK_FREE_RATE_MIN == 0.0
        assert UIConfig.RISK_FREE_RATE_MAX == 25.0
        assert UIConfig.RISK_FREE_RATE_STEP == 0.1
        assert UIConfig.DIVIDEND_YIELD_MIN == 0.0
        assert UIConfig.DIVIDEND_YIELD_MAX == 25.0
        assert UIConfig.DIVIDEND_YIELD_STEP == 0.1
    
    @pytest.mark.unit
    def test_theme_options(self):
        """Test theme configuration."""
        assert "Dark" in UIConfig.AVAILABLE_THEMES
        assert "Light" in UIConfig.AVAILABLE_THEMES
        assert isinstance(UIConfig.AVAILABLE_THEMES, list)
    
    @pytest.mark.unit
    def test_cache_settings(self):
        """Test cache configuration."""
        assert UIConfig.CACHE_TTL_SECONDS == 300
        assert isinstance(UIConfig.CACHE_TTL_SECONDS, int)
        assert UIConfig.CACHE_TTL_SECONDS > 0


class TestConfigIntegration:
    """Integration tests for config module."""
    
    @pytest.mark.integration
    def test_no_config_conflicts(self):
        """Test config values don't conflict."""
        # Strike ranges should make sense
        assert MarketDataConfig.DEFAULT_MIN_STRIKE_PCT < MarketDataConfig.DEFAULT_MAX_STRIKE_PCT
        
        # UI bounds should encompass defaults
        assert UIConfig.MIN_STRIKE_PCT_LOWER <= MarketDataConfig.DEFAULT_MIN_STRIKE_PCT
        assert UIConfig.MAX_STRIKE_PCT_UPPER >= MarketDataConfig.DEFAULT_MAX_STRIKE_PCT
        
        # Model rates should be within bounds
        assert ModelConfig.MIN_RISK_FREE_RATE <= ModelConfig.DEFAULT_RISK_FREE_RATE <= ModelConfig.MAX_RISK_FREE_RATE
        assert ModelConfig.MIN_DIVIDEND_YIELD <= ModelConfig.DEFAULT_DIVIDEND_YIELD <= ModelConfig.MAX_DIVIDEND_YIELD