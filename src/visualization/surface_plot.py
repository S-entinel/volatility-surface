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
        # Create mesh grid similar to the reference implementation
        self.expiry_mesh, self.strike_mesh = np.meshgrid(
            np.linspace(self.data.expiries.min(), self.data.expiries.max(), 50),
            np.linspace(self.data.strikes.min(), self.data.strikes.max(), 50)
        )
        
        # Use linear interpolation as in the reference
        self.vol_mesh = griddata(
            (self.data.expiries, self.data.strikes),
            self.data.ivs,
            (self.expiry_mesh, self.strike_mesh),
            method='linear'
        )
        
        # Mask NaN values
        self.vol_mesh = np.ma.array(self.vol_mesh, mask=np.isnan(self.vol_mesh))
    
    def create_surface_plot(self) -> go.Figure:
        """Generate interactive 3D surface plot"""
        fig = go.Figure(data=[
            go.Surface(
                x=self.expiry_mesh,
                y=self.strike_mesh,
                z=self.vol_mesh * 100,  # Convert to percentage
                colorscale='Viridis',  # Use Viridis colorscale like the reference
                lighting=dict(
                    ambient=0.8,
                    diffuse=0.9,
                    fresnel=0.2,
                    specular=0.1,
                    roughness=0.9
                ),
                colorbar=dict(
                    title='Implied Volatility (%)',
                    titleside='right',
                    x=1.02,
                    thickness=20,
                    len=0.85,
                    tickformat='.0f'
                ),
                contours=dict(
                    x=dict(show=True, color='rgb(200,200,200)', width=1),
                    y=dict(show=True, color='rgb(200,200,200)', width=1),
                    z=dict(show=True, color='rgb(200,200,200)', width=1)
                )
            )
        ])

        # Update layout to match reference
        fig.update_layout(
            scene=dict(
                xaxis_title='Time to Expiration (years)',
                yaxis_title='Strike Price ($)' if self.data.y_axis_type == 'Strike' else 'Moneyness (Strike/Spot)',
                zaxis_title='Implied Volatility (%)',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=-0.2),
                    eye=dict(x=1.5, y=-1.5, z=1.2)
                ),
                aspectratio=dict(x=1.2, y=1.2, z=0.8),
                xaxis=dict(
                    gridcolor='rgb(230,230,230)',
                    showbackground=True,
                    backgroundcolor='white'
                ),
                yaxis=dict(
                    gridcolor='rgb(230,230,230)',
                    showbackground=True,
                    backgroundcolor='white'
                ),
                zaxis=dict(
                    gridcolor='rgb(230,230,230)',
                    showbackground=True,
                    backgroundcolor='white'
                )
            ),
            width=900,
            height=800,
            margin=dict(l=65, r=50, b=65, t=90),
            showlegend=False,
            paper_bgcolor='white',
            plot_bgcolor='white'
        )

        return fig

    def add_smile_slices(self, fig: go.Figure, expiry_days: List[int] = None) -> go.Figure:
        """Add volatility smile curves for specific expiries"""
        if expiry_days is None:
            expiry_days = [30, 60, 90]

        colors = ['rgba(255,255,255,0.8)'] * len(expiry_days)
        
        for days, color in zip(expiry_days, colors):
            expiry_year = days/365
            idx = np.abs(self.expiry_mesh[0] - expiry_year).argmin()
            
            fig.add_trace(
                go.Scatter3d(
                    x=self.expiry_mesh[idx],
                    y=self.strike_mesh[idx],
                    z=self.vol_mesh[idx] * 100,
                    mode='lines',
                    line=dict(color=color, width=2),
                    showlegend=False
                )
            )
        
        return fig