"""
3D surface plotting for implied volatility visualization.

Creates interactive Plotly 3D surface plots with comprehensive type hints
for all classes and methods.
"""

import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
from typing import List, Tuple, Literal, Dict, Any, Optional
from dataclasses import dataclass
import copy
from src.config.config import VisualizationConfig

# Type alias for Y-axis types
YAxisType = Literal['Strike', 'Moneyness']

@dataclass
class SurfaceData:
    """
    Container for volatility surface data.
    
    Attributes:
        strikes: Array of strike prices or moneyness values
        expiries: Array of expiration times (in years)
        ivs: Array of implied volatilities (in decimal form)
        spot_price: Current spot price of the underlying
        y_axis_type: Type of Y-axis ('Strike' or 'Moneyness')
    """
    strikes: np.ndarray
    expiries: np.ndarray
    ivs: np.ndarray
    spot_price: float
    y_axis_type: YAxisType = 'Strike'

class SurfacePlotter:
    """
    3D surface plotter for implied volatility visualization.
    
    Creates interactive Plotly surface plots with customizable themes,
    colormaps, and volatility smile overlays.
    
    Attributes:
        COLORMAP_PRESETS: Dictionary of available colormap configurations
        data: SurfaceData instance containing the volatility surface data
        strike_mesh: 2D array of strike values for surface mesh
        expiry_mesh: 2D array of expiry values for surface mesh
        vol_mesh: 2D array of interpolated volatility values
    """
    
    # Define available colormaps
    COLORMAP_PRESETS: Dict[str, Any] = {
        'Hot': [
            [0, 'rgb(0,0,0)'],      # Black
            [0.25, 'rgb(87,0,0)'],   # Dark red
            [0.5, 'rgb(255,0,0)'],   # Bright red
            [0.75, 'rgb(255,165,0)'], # Orange
            [1.0, 'rgb(255,255,0)']   # Yellow
        ],
        'Viridis': 'Viridis',
        'Plasma': 'Plasma',
        'Blues': [
            [0, 'rgb(8,48,107)'],     # Dark blue
            [0.5, 'rgb(66,146,198)'],  # Medium blue
            [1, 'rgb(198,219,239)']    # Light blue
        ],
        'Rainbow': [
            [0, 'rgb(150,0,90)'],     # Purple
            [0.25, 'rgb(0,0,200)'],   # Blue
            [0.5, 'rgb(0,200,0)'],    # Green
            [0.75, 'rgb(200,200,0)'], # Yellow
            [1, 'rgb(200,0,0)']       # Red
        ],
        'Greyscale': [
            [0, 'rgb(0,0,0)'],       # Black
            [0.5, 'rgb(128,128,128)'], # Grey
            [1, 'rgb(255,255,255)']    # White
        ]
    }

    def __init__(self, surface_data: SurfaceData):
        self.data = surface_data
        self._prepare_mesh()

    def _prepare_mesh(self) -> None:
        """
        Create interpolated mesh for surface plotting.
        
        Generates a regular grid of strike and expiry values, then interpolates
        the implied volatility values onto this grid using linear interpolation.
        
        Raises:
            ValueError: If data contains empty arrays
            
        Returns:
            None (sets instance attributes strike_mesh, expiry_mesh, vol_mesh)
        """
        # Add validation
        if len(self.data.strikes) == 0 or len(self.data.expiries) == 0:
            raise ValueError("Cannot create mesh with empty data")
        
        grid_size = VisualizationConfig.MESH_GRID_SIZE
        
        self.strike_mesh, self.expiry_mesh = np.meshgrid(
            np.linspace(self.data.strikes.min(), self.data.strikes.max(), grid_size),
            np.linspace(self.data.expiries.min(), self.data.expiries.max(), grid_size)
        )
        
        points = np.column_stack((self.data.expiries, self.data.strikes))
        self.vol_mesh = griddata(
            points, self.data.ivs,
            (self.expiry_mesh, self.strike_mesh),
            method='linear'
        )
        
        self.vol_mesh = np.ma.array(self.vol_mesh, mask=np.isnan(self.vol_mesh))
    
    def create_surface_plot(self, theme: str = 'dark', colormap: str = 'Hot') -> go.Figure:
        """
        Generate interactive 3D surface plot with theme and colormap support.
        
        Args:
            theme: Theme name ('dark' or 'light')
            colormap: Colormap name from COLORMAP_PRESETS
            
        Returns:
            Plotly Figure object with configured 3D surface
            
        Example:
            >>> plotter = SurfacePlotter(surface_data)
            >>> fig = plotter.create_surface_plot(theme='dark', colormap='Viridis')
            >>> fig.show()
        """
        is_dark = theme.lower() == 'dark'
        text_color = 'white' if is_dark else 'black'
        bg_color = 'rgb(0, 0, 0)' if is_dark else 'white'
        grid_color = 'rgba(255, 255, 255, 0.2)' if is_dark else 'rgb(180, 180, 180)'
        
        # FIXED: Deep copy to prevent mutation of class variable
        colorscale = copy.deepcopy(self.COLORMAP_PRESETS[colormap])
        
        # Adjust colorscale based on theme
        if isinstance(colorscale, list) and colormap == 'Hot':
            if is_dark:
                colorscale[0][1] = 'rgb(0,0,0)'
            else:
                colorscale[0][1] = 'rgb(255,255,255)'

        fig = go.Figure(data=[
            go.Surface(
                x=self.expiry_mesh,
                y=self.strike_mesh,
                z=self.vol_mesh * StatisticsConfig.IV_DISPLAY_MULTIPLIER,
                colorscale=colorscale,
                lighting=dict(
                    ambient=VisualizationConfig.LIGHTING_AMBIENT,
                    diffuse=VisualizationConfig.LIGHTING_DIFFUSE,
                    fresnel=VisualizationConfig.LIGHTING_FRESNEL,
                    specular=VisualizationConfig.LIGHTING_SPECULAR,
                    roughness=VisualizationConfig.LIGHTING_ROUGHNESS
                ),
                colorbar=dict(
                    title=dict(
                        text='Implied Volatility (%)',
                        side='right',
                        font=dict(color=text_color)
                    ),
                    x=1.02,
                    thickness=20,
                    len=0.85,
                    tickfont=dict(color=text_color)
                )
            )
        ])

        # Update layout with theme and config values
        fig.update_layout(
            scene=dict(
                xaxis_title='Time to Expiration (Years)',
                yaxis_title='Strike Price ($)' if self.data.y_axis_type == 'Strike' else 'Moneyness (Strike/Spot)',
                zaxis_title='Implied Volatility (%)',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=VisualizationConfig.CAMERA_CENTER_Z),
                    eye=dict(
                        x=VisualizationConfig.CAMERA_EYE_X, 
                        y=VisualizationConfig.CAMERA_EYE_Y, 
                        z=VisualizationConfig.CAMERA_EYE_Z
                    )
                ),
                xaxis=dict(
                    gridcolor=grid_color,
                    showbackground=True,
                    backgroundcolor=bg_color,
                    title_font=dict(color=text_color),
                    tickfont=dict(color=text_color),
                    zerolinecolor=grid_color
                ),
                yaxis=dict(
                    gridcolor=grid_color,
                    showbackground=True,
                    backgroundcolor=bg_color,
                    title_font=dict(color=text_color),
                    tickfont=dict(color=text_color),
                    zerolinecolor=grid_color
                ),
                zaxis=dict(
                    gridcolor=grid_color,
                    showbackground=True,
                    backgroundcolor=bg_color,
                    title_font=dict(color=text_color),
                    tickfont=dict(color=text_color),
                    zerolinecolor=grid_color
                ),
                bgcolor=bg_color
            ),
            width=VisualizationConfig.DEFAULT_PLOT_WIDTH,
            height=VisualizationConfig.DEFAULT_PLOT_HEIGHT,
            margin=dict(l=0, r=100, t=0, b=0),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=text_color)
        )

        return fig

    def add_smile_slices(self, fig: go.Figure, theme: str = 'dark', 
                        expiry_days: Optional[List[int]] = None) -> go.Figure:
        """
        Add volatility smile curves for specific expiries.
        
        Args:
            fig: Existing Plotly Figure to add smile slices to
            theme: Theme name ('dark' or 'light') for line color
            expiry_days: List of expiry days to show slices (default from config)
            
        Returns:
            Updated Plotly Figure with smile slice overlays
            
        Example:
            >>> fig = plotter.create_surface_plot()
            >>> fig = plotter.add_smile_slices(fig, expiry_days=[30, 60, 90])
        """
        if expiry_days is None:
            expiry_days = VisualizationConfig.DEFAULT_SMILE_DAYS

        # Theme-dependent line color
        line_color = 'rgba(255,255,255,0.8)' if theme.lower() == 'dark' else 'rgba(0,0,0,0.8)'
        colors = [line_color] * len(expiry_days)
        
        for days, color in zip(expiry_days, colors):
            expiry_year = days/365
            idx = np.abs(self.expiry_mesh[0] - expiry_year).argmin()
            
            # Add validation to prevent index errors
            if idx < len(self.expiry_mesh) and idx < len(self.vol_mesh):
                fig.add_trace(
                    go.Scatter3d(
                        x=self.expiry_mesh[idx],
                        y=self.strike_mesh[idx],
                        z=self.vol_mesh[idx] * StatisticsConfig.IV_DISPLAY_MULTIPLIER,
                        mode='lines',
                        line=dict(color=color, width=VisualizationConfig.SMILE_LINE_WIDTH),
                        showlegend=False
                    )
                )
        
        return fig


# Import StatisticsConfig for IV_DISPLAY_MULTIPLIER
from src.config.config import StatisticsConfig