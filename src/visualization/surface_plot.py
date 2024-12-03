import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class SurfaceData:
    strikes: np.ndarray
    expiries: np.ndarray
    ivs: np.ndarray
    spot_price: float

class SurfacePlotter:
    def __init__(self, surface_data: SurfaceData):
        self.data = surface_data
        self._prepare_mesh()
    
    def _prepare_mesh(self):
        """Create interpolated mesh for surface plotting"""
        self.strike_mesh, self.expiry_mesh = np.meshgrid(
            np.linspace(self.data.strikes.min(), self.data.strikes.max(), 100),
            np.linspace(self.data.expiries.min(), self.data.expiries.max(), 100)
        )
        
        points = np.column_stack((self.data.strikes, self.data.expiries))
        self.vol_mesh = griddata(
            points, self.data.ivs,
            (self.strike_mesh, self.expiry_mesh),
            method='cubic',
            fill_value=np.nan
        )
    
    def create_surface_plot(self) -> go.Figure:
        """Generate interactive 3D surface plot"""
        fig = go.Figure(data=[
            go.Surface(
                x=self.strike_mesh/self.data.spot_price,  # Normalize strikes
                y=self.expiry_mesh * 365,  # Convert to days
                z=self.vol_mesh * 100,  # Convert to percentage
                coloraxis='coloraxis',
                name='Surface'  # Add name for surface
            )
        ])
        
        fig.update_layout(
            title='SPY Implied Volatility Surface',
            scene=dict(
                xaxis_title='Moneyness (Strike/Spot)',
                yaxis_title='Days to Expiry',
                zaxis_title='IV (%)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            coloraxis=dict(
                colorscale='RdYlBu_r',
                colorbar=dict(
                    title='IV (%)',
                    x=1.1,  # Move colorbar more to the right
                    y=0.5   # Center vertically
                )
            ),
            # Add legend for smile curves
            showlegend=True,
            legend=dict(
                x=1.2,     # Position legend to the right of colorbar
                y=0.9,     # Position near the top
                xanchor='left',
                yanchor='top'
            ),
            width=1200,    # Increase width to accommodate legends
            height=800,
            margin=dict(r=150)  # Add right margin for legends
        )
        
        return fig

    def add_smile_slices(self, fig: go.Figure, expiry_days: List[int] = [30, 90, 180]) -> go.Figure:
        """Add volatility smile curves for specific expiries"""
        colors = ['black', 'darkblue', 'darkred']  # Different colors for each slice
        
        for days, color in zip(expiry_days, colors):
            expiry_year = days/365
            # Find nearest expiry in our data
            idx = np.abs(self.expiry_mesh[0] - expiry_year).argmin()
            
            fig.add_trace(
                go.Scatter3d(
                    x=self.strike_mesh[idx]/self.data.spot_price,
                    y=[days] * len(self.strike_mesh[idx]),
                    z=self.vol_mesh[idx] * 100,
                    name=f'{days}d Smile',
                    line=dict(color=color, width=4)
                )
            )
        
        return fig