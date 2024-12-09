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
    
    def create_surface_plot(self) -> go.Figure:
        """Generate interactive 3D surface plot"""
        # Using "hot" colorscale
        hot_colorscale = [
            [0.0, 'rgb(0,0,0)'],      # Black
            [0.25, 'rgb(87,0,0)'],    # Dark red
            [0.5, 'rgb(255,0,0)'],    # Bright red
            [0.75, 'rgb(255,165,0)'], # Orange
            [1.0, 'rgb(255,255,0)']   # Yellow
        ]

        fig = go.Figure(data=[
            go.Surface(
                x=self.expiry_mesh,
                y=self.strike_mesh,
                z=self.vol_mesh * 100,  # Convert to percentage
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
                    tickfont=dict(color='white'),  # White text for colorbar ticks
                    title_font=dict(color='white')  # White text for colorbar title
                )
            )
        ])

        # Update layout with dark theme
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
                    gridcolor='rgba(255, 255, 255, 0.2)',  # Subtle white grid
                    showbackground=True,
                    backgroundcolor='rgb(0, 0, 0)',  # Black background
                    title_font=dict(color='white'),   # White axis title
                    tickfont=dict(color='white'),     # White tick labels
                    zerolinecolor='rgba(255, 255, 255, 0.2)'
                ),
                yaxis=dict(
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    showbackground=True,
                    backgroundcolor='rgb(0, 0, 0)',
                    title_font=dict(color='white'),
                    tickfont=dict(color='white'),
                    zerolinecolor='rgba(255, 255, 255, 0.2)'
                ),
                zaxis=dict(
                    gridcolor='rgba(255, 255, 255, 0.2)',
                    showbackground=True,
                    backgroundcolor='rgb(0, 0, 0)',
                    title_font=dict(color='white'),
                    tickfont=dict(color='white'),
                    zerolinecolor='rgba(255, 255, 255, 0.2)'
                ),
                bgcolor='rgb(0, 0, 0)'  # Black background for the 3D scene
            ),
            width=900,
            height=800,
            margin=dict(l=0, r=100, t=0, b=0),
            paper_bgcolor='rgb(0, 0, 0)',  # Black background for the figure
            plot_bgcolor='rgb(0, 0, 0)',   # Black background for the plot
            font=dict(color='white')       # White text for all other text elements
        )

        return fig

    def add_smile_slices(self, fig: go.Figure, expiry_days: List[int] = None) -> go.Figure:
        """Add volatility smile curves for specific expiries"""
        if expiry_days is None:
            expiry_days = [30, 60, 90]

        colors = ['rgba(255,255,255,0.8)'] * len(expiry_days)  # White lines for visibility
        
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