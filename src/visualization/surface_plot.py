import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
from typing import List, Tuple, Literal
from dataclasses import dataclass

@dataclass
class SurfaceData:
    strikes: np.ndarray
    expiries: np.ndarray
    ivs: np.ndarray
    spot_price: float
    y_axis_type: Literal['Strike', 'Moneyness'] = 'Strike'

class SurfacePlotter:
    def __init__(self, surface_data: SurfaceData):
        self.data = surface_data
        self._prepare_mesh()

    def _prepare_mesh(self):
        """Create interpolated mesh for surface plotting"""
        self.strike_mesh, self.expiry_mesh = np.meshgrid(
            np.linspace(self.data.strikes.min(), self.data.strikes.max(), 50),
            np.linspace(self.data.expiries.min(), self.data.expiries.max(), 50)
        )
        
        points = np.column_stack((self.data.expiries, self.data.strikes))
        self.vol_mesh = griddata(
            points, self.data.ivs,
            (self.expiry_mesh, self.strike_mesh),
            method='linear'
        )
        
        self.vol_mesh = np.ma.array(self.vol_mesh, mask=np.isnan(self.vol_mesh))
    
    def create_surface_plot(self, theme: str = 'dark') -> go.Figure:
        """Generate interactive 3D surface plot with theme support"""
        # Theme-dependent colors
        is_dark = theme.lower() == 'dark'
        text_color = 'white' if is_dark else 'black'
        bg_color = 'rgb(0, 0, 0)' if is_dark else 'white'
        grid_color = 'rgba(255, 255, 255, 0.2)' if is_dark else 'rgb(180, 180, 180)'
        
        # Hot colorscale
        hot_colorscale = [
            [0.0, 'rgb(0,0,0)' if is_dark else 'rgb(255,255,255)'],
            [0.25, 'rgb(87,0,0)'],    # Dark red
            [0.5, 'rgb(255,0,0)'],    # Bright red
            [0.75, 'rgb(255,165,0)'], # Orange
            [1.0, 'rgb(255,255,0)']   # Yellow
        ]

        fig = go.Figure(data=[
            go.Surface(
                x=self.expiry_mesh,
                y=self.strike_mesh,
                z=self.vol_mesh * 100,
                colorscale=hot_colorscale,
                lighting=dict(
                    ambient=0.6,
                    diffuse=0.8,
                    fresnel=0.2,
                    specular=0.4,
                    roughness=0.9
                ),
                colorbar=dict(
                    title='Implied Volatility (%)',
                    titleside='right',
                    x=1.02,
                    thickness=20,
                    len=0.85,
                    tickfont=dict(color=text_color),
                    title_font=dict(color=text_color)
                )
            )
        ])

        # Update layout with theme
        fig.update_layout(
            scene=dict(
                xaxis_title='Time to Expiration (Years)',
                yaxis_title='Strike Price ($)' if self.data.y_axis_type == 'Strike' else 'Moneyness (Strike/Spot)',
                zaxis_title='Implied Volatility (%)',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=-0.2),
                    eye=dict(x=2.2, y=-2.2, z=1.5)
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
            width=900,
            height=800,
            margin=dict(l=0, r=100, t=0, b=0),
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font=dict(color=text_color)
        )

        return fig

    def add_smile_slices(self, fig: go.Figure, theme: str = 'dark', expiry_days: List[int] = None) -> go.Figure:
        """Add volatility smile curves for specific expiries"""
        if expiry_days is None:
            expiry_days = [30, 60, 90]

        # Theme-dependent line color
        line_color = 'rgba(255,255,255,0.8)' if theme.lower() == 'dark' else 'rgba(0,0,0,0.8)'
        colors = [line_color] * len(expiry_days)
        
        for days, color in zip(expiry_days, colors):
            expiry_year = days/365
            idx = np.abs(self.expiry_mesh[0] - expiry_year).argmin()
            
            fig.add_trace(
                go.Scatter3d(
                    x=self.expiry_mesh[idx],
                    y=self.strike_mesh[idx],
                    z=self.vol_mesh[idx] * 100,
                    mode='lines',
                    line=dict(color=color, width=3),
                    showlegend=False
                )
            )
        
        return fig