"""
3D surface plotting for implied volatility visualisation.

Creates interactive Plotly 3D surface plots with comprehensive type hints
for all classes and methods.
"""

import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
from typing import List, Tuple, Literal, Dict, Any, Optional
from dataclasses import dataclass
import copy
from src.config.config import VisualizationConfig, StatisticsConfig

# Type alias for Y-axis types
YAxisType = Literal['Strike', 'Moneyness']

@dataclass
class SurfaceData:
    """
    Container for volatility surface data.
    
    Attributes:
        strikes: Array of strike prices or moneyness values
        expiries: Array of times to expiry (in years)
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
    3D surface plotter for implied volatility visualisation.
    
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
        # Filled in by add_smile_slices: which requested slices were drawn / skipped (in days)
        self.drawn_slice_days: List[int] = []
        self.skipped_slice_days: List[int] = []
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
    
    def create_surface_plot(self, theme: str = 'dark', colormap: str = 'Hot', ticker: str = '') -> go.Figure:
        """
        Generate interactive 3D surface plot with theme and colormap support.
        
        Args:
            theme: Theme name ('dark' or 'light')
            colormap: Colormap name from COLORMAP_PRESETS
            ticker: Stock ticker symbol for title display
            
        Returns:
            Plotly Figure object with configured 3D surface
            
        Example:
            >>> plotter = SurfacePlotter(surface_data)
            >>> fig = plotter.create_surface_plot(theme='dark', colormap='Viridis', ticker='SPY')
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
                        text='IV (%)',
                        side='top',
                        font=dict(color=text_color, size=13, family='Arial, sans-serif')
                    ),
                    x=1.0,  # Moved closer to plot
                    y=0.5,
                    thickness=15,
                    len=0.75,
                    tickfont=dict(color=text_color, size=11),
                    tickformat='.1f'
                )
            )
        ])

        # Create title with ticker
        title_text = f"{ticker} - Implied Volatility Surface" if ticker else "Implied Volatility Surface"

        # Update layout with theme and config values
        fig.update_layout(
            title=dict(
                text=title_text,
                font=dict(size=20, color=text_color, family='Arial, sans-serif', weight='bold'),
                x=0.5,  # Centre align
                xanchor='center',
                y=0.98,
                yanchor='top'
            ),
            scene=dict(
                xaxis_title='Time to Expiry (Years)',
                yaxis_title='Strike ($)' if self.data.y_axis_type == 'Strike' else 'Moneyness',
                zaxis_title='IV (%)',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=-0.1),
                    eye=dict(
                        x=1.8, 
                        y=-1.8, 
                        z=1.3
                    )
                ),
                xaxis=dict(
                    gridcolor=grid_color,
                    showbackground=True,
                    backgroundcolor=bg_color,
                    title_font=dict(color=text_color, size=12),
                    tickfont=dict(color=text_color, size=10),
                    zerolinecolor=grid_color,
                    showspikes=False
                ),
                yaxis=dict(
                    gridcolor=grid_color,
                    showbackground=True,
                    backgroundcolor=bg_color,
                    title_font=dict(color=text_color, size=12),
                    tickfont=dict(color=text_color, size=10),
                    zerolinecolor=grid_color,
                    showspikes=False
                ),
                zaxis=dict(
                    gridcolor=grid_color,
                    showbackground=True,
                    backgroundcolor=bg_color,
                    title_font=dict(color=text_color, size=12),
                    tickfont=dict(color=text_color, size=10),
                    zerolinecolor=grid_color,
                    showspikes=False
                ),
                bgcolor=bg_color
            ),
            width=VisualizationConfig.DEFAULT_PLOT_WIDTH,
            height=VisualizationConfig.DEFAULT_PLOT_HEIGHT,
            margin=dict(l=0, r=120, t=60, b=0),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=text_color, family='Arial, sans-serif'),
            showlegend=False,
            hovermode='closest'
        )

        return fig

    def add_smile_slices(self, fig: go.Figure, theme: str = 'dark',
                        expiry_days: Optional[List[int]] = None) -> go.Figure:
        """
        Add labelled volatility smile curves at specific times to expiry.

        Each slice is interpolated at *exactly* the requested maturity (between the two
        nearest mesh rows), so it lies on the plotted surface and the label is truthful.
        A requested maturity outside the range covered by the data is not drawn (drawing
        the nearest available expiry under the wrong label would be misleading); such
        requests are recorded in ``skipped_slice_days`` so the caller can tell the user.

        Args:
            fig: Existing Plotly Figure to add smile slices to
            theme: Theme name ('dark' or 'light') for line colour
            expiry_days: List of expiry days to show slices (default from config)

        Returns:
            Updated Plotly Figure with smile slice overlays

        Example:
            >>> fig = plotter.create_surface_plot()
            >>> fig = plotter.add_smile_slices(fig, expiry_days=[30, 60, 90])
            >>> plotter.skipped_slice_days   # e.g. [30] if the data starts at 45 days
        """
        if expiry_days is None:
            expiry_days = VisualizationConfig.DEFAULT_SMILE_DAYS

        line_color = 'rgba(255,255,255,0.8)' if theme.lower() == 'dark' else 'rgba(0,0,0,0.8)'

        maturities = self.expiry_mesh[:, 0]   # ascending: one value per mesh row
        strikes = self.strike_mesh[0]         # identical for every row
        vols = np.ma.filled(self.vol_mesh.astype(float), np.nan)
        half_day = 0.5 / 365                  # accept a request for the first/last expiry day itself

        self.drawn_slice_days = []
        self.skipped_slice_days = []

        for days in expiry_days:
            maturity = days / 365

            if maturity < maturities[0] - half_day or maturity > maturities[-1] + half_day:
                self.skipped_slice_days.append(days)
                continue

            maturity = float(np.clip(maturity, maturities[0], maturities[-1]))

            # Linear interpolation between the two mesh rows either side of the maturity
            upper = int(np.clip(np.searchsorted(maturities, maturity), 1, len(maturities) - 1))
            lower = upper - 1
            span = maturities[upper] - maturities[lower]
            weight = (maturity - maturities[lower]) / span if span > 0 else 0.0
            slice_vols = (1 - weight) * vols[lower] + weight * vols[upper]

            finite = np.isfinite(slice_vols)
            if not finite.any():
                # The surface has no values at this maturity (all interpolated points missing)
                self.skipped_slice_days.append(days)
                continue

            # Put the text label on the last point that has a value
            labels = [''] * len(strikes)
            labels[int(np.flatnonzero(finite)[-1])] = f"{days}d"

            fig.add_trace(
                go.Scatter3d(
                    x=np.full(len(strikes), maturity),
                    y=strikes,
                    z=slice_vols * StatisticsConfig.IV_DISPLAY_MULTIPLIER,
                    mode='lines+text',
                    text=labels,
                    textposition='top center',
                    textfont=dict(color=line_color, size=12),
                    line=dict(color=line_color, width=VisualizationConfig.SMILE_LINE_WIDTH),
                    name=f"{days}d",
                    hovertemplate=(f"{days}d<br>{self.data.y_axis_type}: %{{y:.3g}}"
                                   "<br>IV: %{z:.1f}%<extra></extra>"),
                    showlegend=False
                )
            )
            self.drawn_slice_days.append(days)

        return fig