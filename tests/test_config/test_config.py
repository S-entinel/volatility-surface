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
    UIConfig,
    LoggingConfig,
    get_all_configs
)


class TestMarketDataConfig:
    """Test MarketDataConfig values."""
    
    @pytest.mark.unit
    def test_strike_range_defaults(self):
        """Test strike range default values."""
        assert MarketDataConfig.DEFAULT_MIN_STRIKE_PCT == 70.0
        assert MarketDataConfig.DEFAULT_MAX_STRIKE_PCT == 130.0
        assert isinstance(MarketDataConfig.DEFAULT_MIN_STRIKE_PCT, float)
        assert isinstance(MarketDataConfig.DEFAULT_MAX_STRIKE_PCT, float)
    
    @pytest.mark.unit
    def test_volume_defaults(self):
        """Test volume filter defaults."""
        assert MarketDataConfig.DEFAULT_MIN_VOLUME == 10
        assert MarketDataConfig.MIN_VOLUME_THRESHOLD == 0
        assert isinstance(MarketDataConfig.DEFAULT_MIN_VOLUME, int)
    
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
        assert ModelConfig.DEFAULT_RISK_FREE_RATE == 0.045
        assert ModelConfig.DEFAULT_DIVIDEND_YIELD == 0.013
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
        assert IVCalculationConfig.IV_CONVERGENCE_TOLERANCE == 1e-4
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
    def test_camera_position(self):
        """Test camera positioning."""
        assert VisualizationConfig.CAMERA_EYE_X == 2.2
        assert VisualizationConfig.CAMERA_EYE_Y == -2.2
        assert VisualizationConfig.CAMERA_EYE_Z == 1.5
        assert VisualizationConfig.CAMERA_CENTER_Z == -0.2
    
    @pytest.mark.unit
    def test_lighting_parameters(self):
        """Test lighting parameters are in valid ranges."""
        assert 0 <= VisualizationConfig.LIGHTING_AMBIENT <= 1
        assert 0 <= VisualizationConfig.LIGHTING_DIFFUSE <= 1
        assert 0 <= VisualizationConfig.LIGHTING_FRESNEL <= 1
        assert 0 <= VisualizationConfig.LIGHTING_SPECULAR <= 1
        assert 0 <= VisualizationConfig.LIGHTING_ROUGHNESS <= 1
    
    @pytest.mark.unit
    def test_smile_slices(self):
        """Test smile slice configuration."""
        assert VisualizationConfig.DEFAULT_SMILE_DAYS == [30, 60, 90]
        assert isinstance(VisualizationConfig.DEFAULT_SMILE_DAYS, list)
        assert all(isinstance(d, int) for d in VisualizationConfig.DEFAULT_SMILE_DAYS)
        assert VisualizationConfig.SMILE_LINE_WIDTH == 3


class TestStatisticsConfig:
    """Test StatisticsConfig values."""
    
    @pytest.mark.unit
    def test_atm_definition(self):
        """Test ATM moneyness definition."""
        assert StatisticsConfig.ATM_MONEYNESS_LOWER == 0.98
        assert StatisticsConfig.ATM_MONEYNESS_UPPER == 1.02
        assert StatisticsConfig.ATM_MONEYNESS_LOWER < 1.0 < StatisticsConfig.ATM_MONEYNESS_UPPER
    
    @pytest.mark.unit
    def test_otm_definition(self):
        """Test OTM moneyness definition."""
        assert StatisticsConfig.OTM_PUT_MONEYNESS == 0.95
        assert StatisticsConfig.OTM_CALL_MONEYNESS == 1.05
        assert StatisticsConfig.OTM_PUT_MONEYNESS < 1.0 < StatisticsConfig.OTM_CALL_MONEYNESS
    
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
        assert UIConfig.MIN_STRIKE_PCT_UPPER == 99.0
        assert UIConfig.MAX_STRIKE_PCT_LOWER == 101.0
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
    def test_colormap_options(self):
        """Test colormap options."""
        assert "Hot" in UIConfig.AVAILABLE_COLORMAPS
        assert "Viridis" in UIConfig.AVAILABLE_COLORMAPS
        assert isinstance(UIConfig.AVAILABLE_COLORMAPS, list)
    
    @pytest.mark.unit
    def test_cache_settings(self):
        """Test cache configuration."""
        assert UIConfig.CACHE_TTL_SECONDS == 300
        assert isinstance(UIConfig.CACHE_TTL_SECONDS, int)
        assert UIConfig.CACHE_TTL_SECONDS > 0


class TestLoggingConfig:
    """Test LoggingConfig values."""
    
    @pytest.mark.unit
    def test_log_level(self):
        """Test default log level."""
        assert LoggingConfig.DEFAULT_LOG_LEVEL == "INFO"
        assert isinstance(LoggingConfig.DEFAULT_LOG_LEVEL, str)
    
    @pytest.mark.unit
    def test_log_formats(self):
        """Test log format strings."""
        assert LoggingConfig.LOG_DATE_FORMAT == '%Y-%m-%d %H:%M:%S'
        assert '%(asctime)s' in LoggingConfig.LOG_FORMAT
        assert '%(name)s' in LoggingConfig.LOG_FORMAT
        assert '%(levelname)s' in LoggingConfig.LOG_FORMAT
        assert '%(message)s' in LoggingConfig.LOG_FORMAT


class TestConfigIntegration:
    """Integration tests for config module."""
    
    @pytest.mark.integration
    def test_get_all_configs(self):
        """Test get_all_configs function."""
        configs = get_all_configs()
        
        assert isinstance(configs, dict)
        assert len(configs) == 7
        
        assert 'market_data' in configs
        assert 'model' in configs
        assert 'iv_calculation' in configs
        assert 'visualization' in configs
        assert 'statistics' in configs
        assert 'ui' in configs
        assert 'logging' in configs
    
    @pytest.mark.integration
    def test_config_classes_accessible(self):
        """Test all config classes are accessible."""
        configs = get_all_configs()
        
        assert configs['market_data'] == MarketDataConfig
        assert configs['model'] == ModelConfig
        assert configs['iv_calculation'] == IVCalculationConfig
        assert configs['visualization'] == VisualizationConfig
        assert configs['statistics'] == StatisticsConfig
        assert configs['ui'] == UIConfig
        assert configs['logging'] == LoggingConfig
    
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