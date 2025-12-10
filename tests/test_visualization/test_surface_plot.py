"""
Tests for surface plotting visualization module.

Tests SurfaceData and SurfacePlotter functionality.
"""

import pytest
import numpy as np
import plotly.graph_objects as go
from src.visualization.surface_plot import SurfaceData, SurfacePlotter


class TestSurfaceData:
    """Test SurfaceData dataclass."""
    
    @pytest.mark.unit
    def test_surface_data_creation(self):
        """Test SurfaceData can be created."""
        data = SurfaceData(
            strikes=np.array([95, 100, 105]),
            expiries=np.array([0.5, 1.0, 1.5]),
            ivs=np.array([0.2, 0.22, 0.24]),
            spot_price=100.0,
            y_axis_type='Strike'
        )
        
        assert data.spot_price == 100.0
        assert data.y_axis_type == 'Strike'
        assert len(data.strikes) == 3
        assert len(data.expiries) == 3
        assert len(data.ivs) == 3
    
    @pytest.mark.unit
    def test_surface_data_default_y_axis(self):
        """Test default y_axis_type is Strike."""
        data = SurfaceData(
            strikes=np.array([100]),
            expiries=np.array([1.0]),
            ivs=np.array([0.2]),
            spot_price=100.0
        )
        
        assert data.y_axis_type == 'Strike'
    
    @pytest.mark.unit
    def test_surface_data_moneyness_type(self):
        """Test SurfaceData with Moneyness y-axis."""
        data = SurfaceData(
            strikes=np.array([0.95, 1.0, 1.05]),
            expiries=np.array([1.0]),
            ivs=np.array([0.2]),
            spot_price=100.0,
            y_axis_type='Moneyness'
        )
        
        assert data.y_axis_type == 'Moneyness'


class TestSurfacePlotterInit:
    """Test SurfacePlotter initialization."""
    
    @pytest.mark.unit
    def test_plotter_initialization(self):
        """Test plotter can be initialized."""
        # Create data with proper 2D spread (varied expiries AND strikes)
        data = SurfaceData(
            strikes=np.array([80, 90, 95, 100, 105, 110, 115, 120]),
            expiries=np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
            ivs=np.array([0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25]),
            spot_price=100.0
        )
        
        plotter = SurfacePlotter(data)
        
        assert plotter.data == data
        assert hasattr(plotter, 'strike_mesh')
        assert hasattr(plotter, 'expiry_mesh')
        assert hasattr(plotter, 'vol_mesh')
    
    @pytest.mark.unit
    def test_plotter_mesh_created(self):
        """Test mesh is created during initialization."""
        # Create data with proper 2D spread
        data = SurfaceData(
            strikes=np.array([80, 90, 95, 100, 105, 110, 115, 120]),
            expiries=np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]),
            ivs=np.array([0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24, 0.25]),
            spot_price=100.0
        )
        
        plotter = SurfacePlotter(data)
        
        assert plotter.strike_mesh is not None
        assert plotter.expiry_mesh is not None
        assert plotter.vol_mesh is not None
        assert plotter.strike_mesh.shape == plotter.expiry_mesh.shape


class TestPrepareMesh:
    """Test _prepare_mesh method."""
    
    @pytest.mark.unit
    def test_prepare_mesh_creates_grid(self, sample_surface_data):
        """Test mesh preparation creates proper grid."""
        data = SurfaceData(
            strikes=sample_surface_data['strikes'][:20],
            expiries=sample_surface_data['expiries'][:20],
            ivs=sample_surface_data['ivs'][:20],
            spot_price=sample_surface_data['spot_price']
        )
        
        plotter = SurfacePlotter(data)
        
        # Check mesh dimensions
        assert plotter.strike_mesh.ndim == 2
        assert plotter.expiry_mesh.ndim == 2
        assert plotter.vol_mesh.ndim == 2
    
    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_prepare_mesh_empty_data_raises(self):
        """Test mesh preparation with empty data raises error."""
        data = SurfaceData(
            strikes=np.array([]),
            expiries=np.array([]),
            ivs=np.array([]),
            spot_price=100.0
        )
        
        with pytest.raises(ValueError, match="Cannot create mesh with empty data"):
            SurfacePlotter(data)
    
    @pytest.mark.unit
    def test_prepare_mesh_interpolation(self, sample_surface_data):
        """Test mesh uses interpolation."""
        data = SurfaceData(
            strikes=sample_surface_data['strikes'][:15],
            expiries=sample_surface_data['expiries'][:15],
            ivs=sample_surface_data['ivs'][:15],
            spot_price=sample_surface_data['spot_price']
        )
        
        plotter = SurfacePlotter(data)
        
        # Interpolated mesh should have finite values where data exists
        assert np.isfinite(plotter.vol_mesh).any()


class TestCreateSurfacePlot:
    """Test create_surface_plot method."""
    
    @pytest.mark.unit
    def test_create_surface_returns_figure(self, sample_surface_data):
        """Test create_surface_plot returns a Plotly figure."""
        data = SurfaceData(
            strikes=sample_surface_data['strikes'][:20],
            expiries=sample_surface_data['expiries'][:20],
            ivs=sample_surface_data['ivs'][:20],
            spot_price=sample_surface_data['spot_price']
        )
        
        plotter = SurfacePlotter(data)
        fig = plotter.create_surface_plot()
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    @pytest.mark.unit
    def test_create_surface_dark_theme(self, sample_surface_data):
        """Test surface plot with dark theme."""
        data = SurfaceData(
            strikes=sample_surface_data['strikes'][:20],
            expiries=sample_surface_data['expiries'][:20],
            ivs=sample_surface_data['ivs'][:20],
            spot_price=sample_surface_data['spot_price']
        )
        
        plotter = SurfacePlotter(data)
        fig = plotter.create_surface_plot(theme='dark')
        
        assert isinstance(fig, go.Figure)
        # Dark theme should have dark background
        assert fig.layout.paper_bgcolor == 'rgb(0, 0, 0)'
    
    @pytest.mark.unit
    def test_create_surface_light_theme(self, sample_surface_data):
        """Test surface plot with light theme."""
        data = SurfaceData(
            strikes=sample_surface_data['strikes'][:20],
            expiries=sample_surface_data['expiries'][:20],
            ivs=sample_surface_data['ivs'][:20],
            spot_price=sample_surface_data['spot_price']
        )
        
        plotter = SurfacePlotter(data)
        fig = plotter.create_surface_plot(theme='light')
        
        assert isinstance(fig, go.Figure)
        # Light theme should have white background
        assert fig.layout.paper_bgcolor == 'white'
    
    @pytest.mark.unit
    def test_create_surface_different_colormaps(self, sample_surface_data):
        """Test surface plot with different colormaps."""
        data = SurfaceData(
            strikes=sample_surface_data['strikes'][:20],
            expiries=sample_surface_data['expiries'][:20],
            ivs=sample_surface_data['ivs'][:20],
            spot_price=sample_surface_data['spot_price']
        )
        
        plotter = SurfacePlotter(data)
        
        colormaps = ['Hot', 'Viridis', 'Plasma', 'Blues']
        for colormap in colormaps:
            fig = plotter.create_surface_plot(colormap=colormap)
            assert isinstance(fig, go.Figure)


class TestAddSmileSlices:
    """Test add_smile_slices method."""
    
    @pytest.mark.unit
    def test_add_smile_slices_returns_figure(self, sample_surface_data):
        """Test add_smile_slices returns updated figure."""
        data = SurfaceData(
            strikes=sample_surface_data['strikes'][:20],
            expiries=sample_surface_data['expiries'][:20],
            ivs=sample_surface_data['ivs'][:20],
            spot_price=sample_surface_data['spot_price']
        )
        
        plotter = SurfacePlotter(data)
        fig = plotter.create_surface_plot()
        initial_traces = len(fig.data)
        
        fig = plotter.add_smile_slices(fig)
        
        assert isinstance(fig, go.Figure)
        # Should add smile traces
        assert len(fig.data) > initial_traces
    
    @pytest.mark.unit
    def test_add_smile_slices_custom_days(self, sample_surface_data):
        """Test add_smile_slices with custom expiry days."""
        data = SurfaceData(
            strikes=sample_surface_data['strikes'][:20],
            expiries=sample_surface_data['expiries'][:20],
            ivs=sample_surface_data['ivs'][:20],
            spot_price=sample_surface_data['spot_price']
        )
        
        plotter = SurfacePlotter(data)
        fig = plotter.create_surface_plot()
        
        fig = plotter.add_smile_slices(fig, expiry_days=[15, 45, 75])
        
        assert isinstance(fig, go.Figure)
    
    @pytest.mark.unit
    def test_add_smile_slices_theme_colors(self, sample_surface_data):
        """Test smile slices use theme-appropriate colors."""
        data = SurfaceData(
            strikes=sample_surface_data['strikes'][:20],
            expiries=sample_surface_data['expiries'][:20],
            ivs=sample_surface_data['ivs'][:20],
            spot_price=sample_surface_data['spot_price']
        )
        
        plotter = SurfacePlotter(data)
        
        # Dark theme
        fig_dark = plotter.create_surface_plot(theme='dark')
        fig_dark = plotter.add_smile_slices(fig_dark, theme='dark')
        
        # Light theme
        fig_light = plotter.create_surface_plot(theme='light')
        fig_light = plotter.add_smile_slices(fig_light, theme='light')
        
        assert isinstance(fig_dark, go.Figure)
        assert isinstance(fig_light, go.Figure)


class TestColormapPresets:
    """Test COLORMAP_PRESETS configuration."""
    
    @pytest.mark.unit
    def test_colormap_presets_exist(self):
        """Test all colormap presets are defined."""
        expected_colormaps = ['Hot', 'Viridis', 'Plasma', 'Blues', 'Rainbow', 'Greyscale']
        
        for colormap in expected_colormaps:
            assert colormap in SurfacePlotter.COLORMAP_PRESETS
    
    @pytest.mark.unit
    def test_colormap_presets_valid_format(self):
        """Test colormap presets have valid format."""
        for name, colorscale in SurfacePlotter.COLORMAP_PRESETS.items():
            # Should be either string or list
            assert isinstance(colorscale, (str, list))
            
            # If list, should have proper format
            if isinstance(colorscale, list):
                for item in colorscale:
                    assert len(item) == 2
                    assert isinstance(item[0], (int, float))
                    assert isinstance(item[1], str)


class TestSurfacePlotterIntegration:
    """Integration tests for SurfacePlotter."""
    
    @pytest.mark.integration
    def test_full_plotting_workflow(self, sample_surface_data):
        """Test complete plotting workflow."""
        # Create surface data
        data = SurfaceData(
            strikes=sample_surface_data['strikes'][:25],
            expiries=sample_surface_data['expiries'][:25],
            ivs=sample_surface_data['ivs'][:25],
            spot_price=sample_surface_data['spot_price']
        )
        
        # Create plotter
        plotter = SurfacePlotter(data)
        
        # Create surface
        fig = plotter.create_surface_plot(theme='dark', colormap='Viridis')
        
        # Add smile slices
        fig = plotter.add_smile_slices(fig, theme='dark', expiry_days=[30, 60, 90])
        
        # Verify final figure
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 4  # Surface + 3 smile slices
    
    @pytest.mark.integration
    def test_strike_vs_moneyness_plotting(self, sample_surface_data):
        """Test plotting with both Strike and Moneyness."""
        # Strike data
        strike_data = SurfaceData(
            strikes=sample_surface_data['strikes'][:20],
            expiries=sample_surface_data['expiries'][:20],
            ivs=sample_surface_data['ivs'][:20],
            spot_price=sample_surface_data['spot_price'],
            y_axis_type='Strike'
        )
        
        # Moneyness data
        moneyness_data = SurfaceData(
            strikes=sample_surface_data['strikes'][:20] / sample_surface_data['spot_price'],
            expiries=sample_surface_data['expiries'][:20],
            ivs=sample_surface_data['ivs'][:20],
            spot_price=sample_surface_data['spot_price'],
            y_axis_type='Moneyness'
        )
        
        # Create plots
        strike_plotter = SurfacePlotter(strike_data)
        moneyness_plotter = SurfacePlotter(moneyness_data)
        
        strike_fig = strike_plotter.create_surface_plot()
        moneyness_fig = moneyness_plotter.create_surface_plot()
        
        assert isinstance(strike_fig, go.Figure)
        assert isinstance(moneyness_fig, go.Figure)
    
    @pytest.mark.integration
    def test_all_themes_and_colormaps(self, sample_surface_data):
        """Test all combinations of themes and colormaps work."""
        data = SurfaceData(
            strikes=sample_surface_data['strikes'][:15],
            expiries=sample_surface_data['expiries'][:15],
            ivs=sample_surface_data['ivs'][:15],
            spot_price=sample_surface_data['spot_price']
        )
        
        plotter = SurfacePlotter(data)
        
        themes = ['dark', 'light']
        colormaps = ['Hot', 'Viridis', 'Plasma']
        
        for theme in themes:
            for colormap in colormaps:
                fig = plotter.create_surface_plot(theme=theme, colormap=colormap)
                assert isinstance(fig, go.Figure)


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.edge_case
    def test_minimal_data_points(self):
        """Test plotting with small dataset."""
        # Need a 2D grid of points (multiple strikes at multiple expiries)
        # Create a small grid: 4 strikes x 4 expiries = 16 points
        strikes_base = [90.0, 95.0, 105.0, 110.0]
        expiries_base = [0.5, 0.75, 1.0, 1.5]
        
        strikes = []
        expiries = []
        ivs = []
        
        for i, exp in enumerate(expiries_base):
            for j, strike in enumerate(strikes_base):
                strikes.append(strike)
                expiries.append(exp)
                # Create some variation in IVs
                iv = 0.20 + 0.01 * i + 0.005 * j
                ivs.append(iv)
        
        data = SurfaceData(
            strikes=np.array(strikes),
            expiries=np.array(expiries),
            ivs=np.array(ivs),
            spot_price=100.0
        )
        
        plotter = SurfacePlotter(data)
        fig = plotter.create_surface_plot()
        
        assert isinstance(fig, go.Figure)
    
    @pytest.mark.edge_case
    def test_high_volatility_values(self):
        """Test plotting with very high IVs."""
        data = SurfaceData(
            strikes=np.array([85, 90, 95, 100, 105, 110, 115]),
            expiries=np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]),
            ivs=np.array([1.8, 2.0, 2.2, 2.5, 2.7, 3.0, 3.2]),  # Very high IVs
            spot_price=100.0
        )
        
        plotter = SurfacePlotter(data)
        fig = plotter.create_surface_plot()
        
        assert isinstance(fig, go.Figure)
    
    @pytest.mark.edge_case
    def test_low_volatility_values(self):
        """Test plotting with very low IVs."""
        data = SurfaceData(
            strikes=np.array([85, 90, 95, 100, 105, 110, 115]),
            expiries=np.array([0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]),
            ivs=np.array([0.008, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03]),  # Very low IVs
            spot_price=100.0
        )
        
        plotter = SurfacePlotter(data)
        fig = plotter.create_surface_plot()
        
        assert isinstance(fig, go.Figure)