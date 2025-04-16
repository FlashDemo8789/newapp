"""
Enhanced Interactive Visualization System for FlashDNA

This module provides advanced visualization capabilities for the FlashDNA platform
with multiple rendering backends, interactive components, and robust error handling.

Key features:
1. Support for multiple backends (Plotly, Matplotlib, Altair)
2. Consistent styling across visualization types
3. Robust error handling with graceful fallbacks
4. Advanced chart types for startup analysis
5. Exportable chart objects for dashboard and PDF integration
6. Theme support for light/dark mode and branding customization
"""

import numpy as np
import pandas as pd
import io
import logging
import base64
import traceback
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum, auto
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("enhanced_viz")

# Type definitions
DataDict = Dict[str, Any]
ChartObject = Any  # Can be a Plotly figure, Matplotlib figure, etc.

# Define color palettes for consistent styling
class ColorTheme:
    """Color themes for visualizations with light/dark mode support."""
    
    BLUE = "#1f77b4"
    GREEN = "#2ca02c"
    RED = "#d62728"
    ORANGE = "#ff7f0e"
    PURPLE = "#9467bd"
    
    # Main color palette
    COLORS = [
        BLUE, ORANGE, GREEN, RED, PURPLE,
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    
    # Color scales
    SEQUENTIAL = "Viridis"
    DIVERGING = "RdBu"
    
    # Light mode (default)
    LIGHT = {
        "background": "#ffffff",
        "paper_background": "#f8f9fa",
        "text": "#343a40",
        "grid": "#e9ecef",
        "axis": "#6c757d"
    }
    
    # Dark mode
    DARK = {
        "background": "#212529",
        "paper_background": "#343a40",
        "text": "#f8f9fa",
        "grid": "#495057",
        "axis": "#adb5bd"
    }
    
    @classmethod
    def get_theme(cls, dark_mode: bool = False) -> Dict[str, str]:
        """Get the appropriate theme based on mode."""
        return cls.DARK if dark_mode else cls.LIGHT
    
    @classmethod
    def get_color(cls, index: int) -> str:
        """Get a color from the palette with wrapping."""
        return cls.COLORS[index % len(cls.COLORS)]

class ChartType(Enum):
    """Enumeration of supported chart types."""
    LINE = auto()
    BAR = auto()
    SCATTER = auto()
    PIE = auto()
    RADAR = auto()
    HEATMAP = auto()
    BOX = auto()
    VIOLIN = auto()
    HISTOGRAM = auto()
    AREA = auto()
    SCATTER_3D = auto()
    SURFACE_3D = auto()
    NETWORK = auto()
    SANKEY = auto()
    SUNBURST = auto()
    TREEMAP = auto()
    FUNNEL = auto()
    GAUGE = auto()
    CANDLESTICK = auto()
    WATERFALL = auto()
    PARALLEL = auto()
    BUBBLE = auto()

class Backend(Enum):
    """Enumeration of supported visualization backends."""
    PLOTLY = auto()
    MATPLOTLIB = auto()
    ALTAIR = auto()
    D3 = auto()  # For specialized visualizations

# Try to import visualization libraries with proper error handling
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    logger.warning("Plotly not available - some visualizations will have limited functionality")
    PLOTLY_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available - some visualizations will have limited functionality")
    MATPLOTLIB_AVAILABLE = False

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    logger.warning("Altair not available - some visualizations will have limited functionality")
    ALTAIR_AVAILABLE = False

class EnhancedVisualization:
    """
    Enhanced visualization system for FlashDNA with multiple backends and advanced chart types.
    """
    
    def __init__(self, dark_mode: bool = False, default_backend: Backend = Backend.PLOTLY):
        """
        Initialize the visualization system.
        
        Args:
            dark_mode: Whether to use dark mode theme
            default_backend: Default visualization backend
        """
        self.dark_mode = dark_mode
        self.theme = ColorTheme.get_theme(dark_mode)
        self.default_backend = self._get_available_backend(default_backend)
        
        # Keep track of available backends
        self.available_backends = self._get_available_backends()
        logger.info(f"Enhanced visualization initialized with {len(self.available_backends)} backends")
        
        # Tracker for fallback events
        self.fallback_count = 0
    
    def _get_available_backends(self) -> List[Backend]:
        """Determine which visualization backends are available."""
        available = []
        
        if PLOTLY_AVAILABLE:
            available.append(Backend.PLOTLY)
        
        if MATPLOTLIB_AVAILABLE:
            available.append(Backend.MATPLOTLIB)
        
        if ALTAIR_AVAILABLE:
            available.append(Backend.ALTAIR)
        
        # D3 always available through our custom implementation
        available.append(Backend.D3)
        
        return available
    
    def _get_available_backend(self, preferred: Backend) -> Backend:
        """Get the best available backend, falling back if necessary."""
        if preferred in self._get_available_backends():
            return preferred
        
        # Fallback priority: Plotly > Matplotlib > Altair > D3
        if PLOTLY_AVAILABLE:
            return Backend.PLOTLY
        elif MATPLOTLIB_AVAILABLE:
            return Backend.MATPLOTLIB
        elif ALTAIR_AVAILABLE:
            return Backend.ALTAIR
        else:
            return Backend.D3
    
    def _fallback_to_available_backend(self, current_backend: Backend) -> Backend:
        """Fallback to next available backend if current one fails."""
        self.fallback_count += 1
        
        # Fallback priority based on what's left
        if current_backend == Backend.PLOTLY and Backend.MATPLOTLIB in self.available_backends:
            logger.info("Falling back from Plotly to Matplotlib")
            return Backend.MATPLOTLIB
        elif current_backend == Backend.MATPLOTLIB and Backend.ALTAIR in self.available_backends:
            logger.info("Falling back from Matplotlib to Altair")
            return Backend.ALTAIR
        elif current_backend == Backend.ALTAIR and Backend.D3 in self.available_backends:
            logger.info("Falling back from Altair to D3")
            return Backend.D3
        else:
            logger.warning("No more fallback options available")
            return Backend.D3  # Ultimate fallback
    
    def _dict_to_dataframe(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Convert a dictionary to a DataFrame suitable for visualization."""
        # Handle various dictionary formats
        if all(isinstance(data[key], (list, np.ndarray)) for key in data):
            # Dictionary of lists/arrays -> columns in DataFrame
            return pd.DataFrame(data)
        elif all(isinstance(data[key], dict) for key in data):
            # Dictionary of dictionaries -> nested structure
            return pd.DataFrame.from_dict(data, orient='index')
        else:
            # Simple key-value pairs -> two columns
            return pd.DataFrame(list(data.items()), columns=['key', 'value'])

    def create_chart(self, 
                    chart_type: ChartType, 
                    data: Union[pd.DataFrame, Dict[str, Any], List[Any]],
                    x: Optional[str] = None,
                    y: Optional[Union[str, List[str]]] = None,
                    color: Optional[str] = None,
                    size: Optional[str] = None,
                    title: Optional[str] = None,
                    xlabel: Optional[str] = None,
                    ylabel: Optional[str] = None,
                    backend: Optional[Backend] = None,
                    **kwargs) -> Tuple[ChartObject, Backend]:
        """
        Create a chart based on the specified type and data.
        
        Args:
            chart_type: Type of chart to create
            data: Data for the chart
            x: X-axis data field
            y: Y-axis data field(s)
            color: Field to use for color encoding
            size: Field to use for size encoding
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            backend: Visualization backend to use
            **kwargs: Additional chart-specific parameters
            
        Returns:
            Tuple of (chart_object, backend_used)
        """
        # Use default backend if none specified
        backend = backend or self.default_backend
        
        # Convert data to DataFrame if it's a dict or list
        if isinstance(data, dict):
            try:
                data = pd.DataFrame(data)
            except Exception as e:
                logger.warning(f"Failed to convert dict to DataFrame: {e}")
                # Try to convert based on chart type
                if chart_type in [ChartType.LINE, ChartType.BAR, ChartType.SCATTER]:
                    data = self._dict_to_dataframe(data)
        elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
            try:
                data = pd.DataFrame(data)
            except Exception as e:
                logger.warning(f"Failed to convert list of dicts to DataFrame: {e}")
        
        # Try to create the chart with the specified backend
        try:
            if backend == Backend.PLOTLY and PLOTLY_AVAILABLE:
                chart = self._create_plotly_chart(chart_type, data, x, y, color, size, title, xlabel, ylabel, **kwargs)
                return chart, Backend.PLOTLY
            elif backend == Backend.MATPLOTLIB and MATPLOTLIB_AVAILABLE:
                chart = self._create_matplotlib_chart(chart_type, data, x, y, color, size, title, xlabel, ylabel, **kwargs)
                return chart, Backend.MATPLOTLIB
            elif backend == Backend.ALTAIR and ALTAIR_AVAILABLE:
                chart = self._create_altair_chart(chart_type, data, x, y, color, size, title, xlabel, ylabel, **kwargs)
                return chart, Backend.ALTAIR
            elif backend == Backend.D3:
                chart = self._create_d3_chart(chart_type, data, x, y, color, size, title, xlabel, ylabel, **kwargs)
                return chart, Backend.D3
            else:
                # If specified backend is not available, fallback
                new_backend = self._fallback_to_available_backend(backend)
                logger.warning(f"Backend {backend} not available, falling back to {new_backend}")
                return self.create_chart(chart_type, data, x, y, color, size, title, xlabel, ylabel, new_backend, **kwargs)
        except Exception as e:
            logger.error(f"Error creating chart with {backend}: {e}")
            logger.error(traceback.format_exc())
            
            # Try with fallback backend
            try:
                new_backend = self._fallback_to_available_backend(backend)
                if new_backend != backend:
                    logger.info(f"Attempting with fallback backend {new_backend}")
                    return self.create_chart(chart_type, data, x, y, color, size, title, xlabel, ylabel, new_backend, **kwargs)
            except Exception as e2:
                logger.error(f"Error with fallback backend: {e2}")
            
            # Ultimate fallback: simple error chart
            return self._create_error_chart(str(e)), Backend.MATPLOTLIB

    def _create_plotly_chart(self, chart_type, data, x, y, color, size, title, xlabel, ylabel, **kwargs):
        """Create a chart using Plotly."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is not available")
        
        # Apply theme
        template = "plotly_dark" if self.dark_mode else "plotly_white"
        
        # Create the appropriate chart based on type
        if chart_type == ChartType.LINE:
            if isinstance(y, list):
                # Multiple lines
                fig = go.Figure()
                for i, y_col in enumerate(y):
                    fig.add_trace(go.Scatter(
                        x=data[x] if x in data else data.index,
                        y=data[y_col],
                        mode='lines',
                        name=y_col,
                        line=dict(color=ColorTheme.get_color(i), width=2)
                    ))
            else:
                # Single line
                fig = px.line(
                    data, x=x, y=y, color=color,
                    title=title,
                    template=template
                )
            
        elif chart_type == ChartType.BAR:
            fig = px.bar(
                data, x=x, y=y, color=color,
                title=title,
                template=template
            )
            
        elif chart_type == ChartType.SCATTER:
            fig = px.scatter(
                data, x=x, y=y, color=color, size=size,
                title=title,
                template=template
            )
            
        elif chart_type == ChartType.PIE:
            fig = px.pie(
                data, values=y, names=x, color=x,
                title=title,
                template=template,
                color_discrete_sequence=ColorTheme.COLORS
            )
            
        elif chart_type == ChartType.RADAR:
            # For radar charts, we need to close the loop
            if isinstance(data, pd.DataFrame):
                categories = data[x].tolist() if x else data.index.tolist()
                values = data[y].tolist() if y else data.iloc[:, 0].tolist()
            else:
                categories = kwargs.get('categories', [f'Category {i+1}' for i in range(len(data))])
                values = kwargs.get('values', data if isinstance(data, list) else [])
            
            # Close the loop
            categories = categories + [categories[0]]
            values = values + [values[0]]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                line=dict(color=ColorTheme.BLUE, width=2),
                fillcolor=f'rgba{tuple(int(ColorTheme.BLUE.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}'
            ))
            
        elif chart_type == ChartType.HEATMAP:
            fig = px.imshow(
                data if isinstance(data, (list, np.ndarray)) else data.pivot(index=y, columns=x, values=kwargs.get('values')),
                title=title,
                color_continuous_scale=ColorTheme.SEQUENTIAL
            )
            
        elif chart_type == ChartType.SCATTER_3D:
            fig = px.scatter_3d(
                data, x=x, y=y, z=kwargs.get('z'), color=color, size=size,
                title=title,
                template=template
            )
            
        elif chart_type == ChartType.SURFACE_3D:
            # Prepare data for 3D surface
            if 'z_matrix' in kwargs:
                z_matrix = kwargs['z_matrix']
            elif all(col in data.columns for col in [x, y, kwargs.get('z', 'z')]):
                # Convert from long to wide format
                z_matrix = data.pivot(index=y, columns=x, values=kwargs.get('z', 'z')).values
            else:
                z_matrix = data.values if isinstance(data, pd.DataFrame) else np.array(data)
            
            fig = go.Figure(data=[go.Surface(z=z_matrix, colorscale=ColorTheme.SEQUENTIAL)])
            
        elif chart_type == ChartType.NETWORK:
            # Network graph needs nodes and edges
            nodes = kwargs.get('nodes', [])
            edges = kwargs.get('edges', [])
            
            # Convert node and edge data if needed
            if isinstance(data, dict) and 'nodes' in data and 'edges' in data:
                nodes = data['nodes']
                edges = data['edges']
            
            # Create a network graph
            node_x = []
            node_y = []
            node_colors = []
            node_sizes = []
            
            for node in nodes:
                x = node.get('x', np.random.random())
                y = node.get('y', np.random.random())
                node_x.append(x)
                node_y.append(y)
                node_colors.append(node.get('color', 0.5))
                node_sizes.append(node.get('size', 10))
            
            edge_x = []
            edge_y = []
            
            for edge in edges:
                source_idx = next((i for i, n in enumerate(nodes) if n.get('id') == edge.get('source')), None)
                target_idx = next((i for i, n in enumerate(nodes) if n.get('id') == edge.get('target')), None)
                
                if source_idx is not None and target_idx is not None:
                    x0, y0 = node_x[source_idx], node_y[source_idx]
                    x1, y1 = node_x[target_idx], node_y[target_idx]
                    
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
            
            # Create the figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines'
            ))
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=node_x, y=node_y,
                mode='markers',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale=ColorTheme.SEQUENTIAL,
                    line_width=2
                ),
                text=[node.get('id', f"Node {i}") for i, node in enumerate(nodes)],
                hoverinfo='text'
            ))
            
        elif chart_type == ChartType.SANKEY:
            # Sankey diagram for flow analysis
            if 'source' in kwargs and 'target' in kwargs and 'value' in kwargs:
                source = kwargs['source']
                target = kwargs['target']
                value = kwargs['value']
            elif isinstance(data, pd.DataFrame):
                source = data[x].tolist() if x else data.iloc[:, 0].tolist()
                target = data[y].tolist() if y else data.iloc[:, 1].tolist()
                value = data[kwargs.get('value_col', 'value')].tolist() if kwargs.get('value_col', 'value') in data else [1] * len(source)
            
            fig = go.Figure(data=[go.Sankey(
                node=dict(
                    pad=15,
                    thickness=20,
                    line=dict(color="black", width=0.5),
                    label=list(set(source + target))
                ),
                link=dict(
                    source=[list(set(source + target)).index(s) for s in source],
                    target=[list(set(source + target)).index(t) for t in target],
                    value=value
                )
            )])
            
        else:
            # For unsupported chart types, use a simple scatter plot
            logger.warning(f"Unsupported chart type {chart_type} for Plotly, using scatter plot")
            fig = px.scatter(
                data, x=x, y=y, color=color, size=size,
                title=title + " (Fallback)",
                template=template
            )
        
        # Set axis labels
        if xlabel:
            fig.update_xaxes(title_text=xlabel)
        if ylabel:
            fig.update_yaxes(title_text=ylabel)
        
        # Set title if not already set
        if title and 'title' not in fig.layout:
            fig.update_layout(title=title)
        
        # Apply consistent styling
        fig.update_layout(
            template=template,
            paper_bgcolor=self.theme['paper_background'],
            plot_bgcolor=self.theme['background'],
            font=dict(color=self.theme['text']),
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
        return fig

    def _create_matplotlib_chart(self, chart_type, data, x, y, color, size, title, xlabel, ylabel, **kwargs):
        """Create a chart using Matplotlib."""
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError("Matplotlib is not available")
        
        # Set the style based on theme
        plt.style.use('dark_background' if self.dark_mode else 'default')
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 5)))
        
        # Create the appropriate chart based on type
        if chart_type == ChartType.LINE:
            if isinstance(y, list):
                # Multiple lines
                for i, y_col in enumerate(y):
                    ax.plot(
                        data[x] if x in data.columns else data.index,
                        data[y_col],
                        color=ColorTheme.get_color(i),
                        label=y_col,
                        linewidth=2
                    )
                ax.legend()
            else:
                # Single line
                ax.plot(
                    data[x] if x in data.columns else data.index,
                    data[y],
                    color=ColorTheme.BLUE,
                    linewidth=2
                )
                
        elif chart_type == ChartType.BAR:
            if isinstance(y, list):
                # Grouped bar chart
                x_pos = np.arange(len(data[x] if x in data.columns else data.index))
                width = 0.8 / len(y)
                
                for i, y_col in enumerate(y):
                    ax.bar(
                        x_pos + i * width - 0.4 + width/2,
                        data[y_col],
                        width=width,
                        color=ColorTheme.get_color(i),
                        label=y_col
                    )
                
                ax.set_xticks(x_pos)
                ax.set_xticklabels(data[x] if x in data.columns else data.index)
                ax.legend()
            else:
                # Simple bar chart
                if color and color in data.columns:
                    # Color bars by a column
                    colors = [plt.cm.viridis(x/max(data[color])) for x in data[color]]
                    ax.bar(
                        data[x] if x in data.columns else data.index,
                        data[y],
                        color=colors
                    )
                else:
                    ax.bar(
                        data[x] if x in data.columns else data.index,
                        data[y],
                        color=ColorTheme.BLUE
                    )
                
        elif chart_type == ChartType.SCATTER:
            scatter_kwargs = {'s': 100}  # Default size
            
            # Use size column if specified
            if size and size in data.columns:
                scatter_kwargs['s'] = data[size] * 100
            
            # Use color column if specified
            if color and color in data.columns:
                scatter = ax.scatter(
                    data[x],
                    data[y],
                    c=data[color],
                    cmap='viridis',
                    **scatter_kwargs
                )
                plt.colorbar(scatter, ax=ax, label=color)
            else:
                ax.scatter(
                    data[x],
                    data[y],
                    color=ColorTheme.BLUE,
                    **scatter_kwargs
                )
                
        elif chart_type == ChartType.PIE:
            if isinstance(data, pd.DataFrame):
                values = data[y].values if y in data.columns else data.iloc[:, 0].values
                labels = data[x].values if x in data.columns else data.index
            else:
                values = y if isinstance(y, list) else data
                labels = x if isinstance(x, list) else [f'Category {i+1}' for i in range(len(values))]
                
            ax.pie(
                values,
                labels=labels,
                autopct='%1.1f%%',
                colors=[ColorTheme.get_color(i) for i in range(len(values))],
                shadow=False,
                startangle=90
            )
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
                
        elif chart_type == ChartType.RADAR:
            # For radar charts, we need special handling
            if isinstance(data, pd.DataFrame):
                categories = data[x].tolist() if x else data.index.tolist()
                values = data[y].tolist() if y else data.iloc[:, 0].tolist()
            else:
                categories = kwargs.get('categories', [f'Category {i+1}' for i in range(len(data))])
                values = kwargs.get('values', data if isinstance(data, list) else [])
            
            # Close the loop
            categories = categories + [categories[0]]
            values = values + [values[0]]
            
            # Calculate angles for each category
            N = len(categories) - 1
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            # Create polar plot
            ax = plt.subplot(111, polar=True)
            
            # Draw the chart
            ax.plot(angles, values, color=ColorTheme.BLUE, linewidth=2)
            ax.fill(angles, values, color=ColorTheme.BLUE, alpha=0.25)
            
            # Set category labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories[:-1])
            
            # Set y-axis limits
            ax.set_ylim(0, max(values) * 1.1)
                
        elif chart_type == ChartType.HEATMAP:
            if isinstance(data, pd.DataFrame):
                if x and y:
                    # Pivot data for heatmap
                    try:
                        pivot_data = data.pivot(index=y, columns=x, values=kwargs.get('values', data.columns[0]))
                        im = ax.imshow(pivot_data.values, cmap='viridis')
                        ax.set_xticks(np.arange(len(pivot_data.columns)))
                        ax.set_yticks(np.arange(len(pivot_data.index)))
                        ax.set_xticklabels(pivot_data.columns)
                        ax.set_yticklabels(pivot_data.index)
                    except Exception as e:
                        logger.warning(f"Error pivoting data for heatmap: {e}")
                        im = ax.imshow(data.values, cmap='viridis')
                else:
                    im = ax.imshow(data.values, cmap='viridis')
            else:
                im = ax.imshow(data, cmap='viridis')
            
            plt.colorbar(im, ax=ax)
                
        elif chart_type == ChartType.SCATTER_3D:
            # 3D scatter plot needs a 3D projection
            fig = plt.figure(figsize=kwargs.get('figsize', (10, 8)))
            ax = fig.add_subplot(111, projection='3d')
            
            # Get z-axis data
            z = kwargs.get('z')
            
            if color and color in data.columns:
                scatter = ax.scatter(
                    data[x],
                    data[y],
                    data[z] if z in data.columns else np.zeros(len(data)),
                    c=data[color],
                    s=data[size] * 100 if size and size in data.columns else 100,
                    cmap='viridis'
                )
                plt.colorbar(scatter, ax=ax, label=color)
            else:
                ax.scatter(
                    data[x],
                    data[y],
                    data[z] if z in data.columns else np.zeros(len(data)),
                    color=ColorTheme.BLUE,
                    s=data[size] * 100 if size and size in data.columns else 100
                )
            
            # Set z-axis label
            ax.set_zlabel(z if z else 'z')
                
        elif chart_type == ChartType.SURFACE_3D:
            # 3D surface plot needs a 3D projection
            fig = plt.figure(figsize=kwargs.get('figsize', (10, 8)))
            ax = fig.add_subplot(111, projection='3d')
            
            # Prepare data for 3D surface
            if 'z_matrix' in kwargs:
                z_matrix = kwargs['z_matrix']
            elif all(col in data.columns for col in [x, y, kwargs.get('z', 'z')]):
                # Convert from long to wide format
                z_matrix = data.pivot(index=y, columns=x, values=kwargs.get('z', 'z')).values
            else:
                z_matrix = data.values if isinstance(data, pd.DataFrame) else np.array(data)
            
            # Create x and y coordinates
            x_coords = np.arange(z_matrix.shape[1])
            y_coords = np.arange(z_matrix.shape[0])
            x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
            
            # Create the surface plot
            surf = ax.plot_surface(
                x_mesh, y_mesh, z_matrix,
                cmap='viridis',
                linewidth=0,
                antialiased=False
            )
            
            plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
            
        else:
            # For unsupported chart types, use a simple scatter plot
            logger.warning(f"Unsupported chart type {chart_type} for Matplotlib, using scatter plot")
            ax.scatter(
                data[x] if isinstance(data, pd.DataFrame) and x in data.columns else range(len(data)),
                data[y] if isinstance(data, pd.DataFrame) and y in data.columns else data,
                color=ColorTheme.BLUE,
                s=100
            )
        
        # Set axis labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        elif x and not isinstance(ax, plt.PolarAxes) and not hasattr(ax, 'projection'):
            ax.set_xlabel(x)
            
        if ylabel:
            ax.set_ylabel(ylabel)
        elif y and not isinstance(y, list) and not isinstance(ax, plt.PolarAxes) and not hasattr(ax, 'projection'):
            ax.set_ylabel(y)
            
        if title:
            plt.title(title)
        
        # Apply grid
        if not isinstance(ax, plt.PolarAxes) and chart_type not in [ChartType.PIE, ChartType.HEATMAP]:
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Tight layout for better spacing
        plt.tight_layout()
        
        return fig

    def _create_altair_chart(self, chart_type, data, x, y, color, size, title, xlabel, ylabel, **kwargs):
        """Create a chart using Altair."""
        if not ALTAIR_AVAILABLE:
            raise ImportError("Altair is not available")
        
        # Altair requires a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            if isinstance(data, dict):
                data = pd.DataFrame(data)
            elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
                data = pd.DataFrame(data)
            else:
                # Convert list or array to DataFrame
                if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                    data = pd.DataFrame({'value': data, 'index': range(len(data))})
                    x = 'index'
                    y = 'value'
                else:
                    raise ValueError("Data must be a pandas DataFrame, dict, or list of dicts for Altair")
        
        # Set base chart properties
        if x and y:
            base = alt.Chart(data)
            encoding = {}
            
            if x:
                encoding['x'] = alt.X(x, title=xlabel or x)
            if y:
                if isinstance(y, list):
                    # Handle multiple y values below
                    encoding['y'] = alt.Y(y[0], title=ylabel or y[0])
                else:
                    encoding['y'] = alt.Y(y, title=ylabel or y)
            if color:
                encoding['color'] = color
            if size:
                encoding['size'] = size
            
            # Create different chart types
            if chart_type == ChartType.LINE:
                if isinstance(y, list):
                    # Multiple lines - create tidy data first
                    tidy_data = pd.melt(data, id_vars=[x], value_vars=y, var_name='series', value_name='value')
                    chart = alt.Chart(tidy_data).mark_line().encode(
                        x=alt.X(x, title=xlabel or x),
                        y=alt.Y('value:Q', title=ylabel or 'Value'),
                        color='series:N'
                    )
                else:
                    chart = base.mark_line().encode(**encoding)
                    
            elif chart_type == ChartType.BAR:
                if isinstance(y, list):
                    # Multiple bars - create tidy data first
                    tidy_data = pd.melt(data, id_vars=[x], value_vars=y, var_name='series', value_name='value')
                    chart = alt.Chart(tidy_data).mark_bar().encode(
                        x=alt.X(x, title=xlabel or x),
                        y=alt.Y('value:Q', title=ylabel or 'Value'),
                        color='series:N'
                    )
                else:
                    chart = base.mark_bar().encode(**encoding)
                    
            elif chart_type == ChartType.SCATTER:
                chart = base.mark_circle().encode(**encoding)
                
            elif chart_type == ChartType.PIE:
                # Pie charts in Altair are created using a transformed bar chart
                if y and x:
                    chart = alt.Chart(data).mark_arc().encode(
                        theta=alt.Theta(field=y, type='quantitative'),
                        color=alt.Color(field=x, type='nominal')
                    )
                
            elif chart_type == ChartType.HEATMAP:
                if x and y and 'values' in kwargs:
                    chart = alt.Chart(data).mark_rect().encode(
                        x=alt.X(x, title=xlabel or x),
                        y=alt.Y(y, title=ylabel or y),
                        color=alt.Color(kwargs['values'], title=kwargs.get('values_label', 'Value'))
                    )
                
            else:
                # Default to scatter plot for unsupported types
                logger.warning(f"Unsupported chart type {chart_type} for Altair, using scatter plot")
                chart = base.mark_circle().encode(**encoding)
            
            # Set title if provided
            if title:
                chart = chart.properties(title=title)
            
            return chart
            
        else:
            # Minimal chart when x or y is missing
            return alt.Chart(data).mark_point()
    
    def _create_d3_chart(self, chart_type, data, x, y, color, size, title, xlabel, ylabel, **kwargs):
        """
        Create a D3.js-compatible specification (for advanced visualizations).
        
        This creates a specification that can be rendered by D3.js in the frontend.
        For PDF reports, this will fall back to static images.
        """
        # Create a chart specification that can be rendered by D3.js
        spec = {
            "type": chart_type.name.lower(),
            "data": self._prepare_data_for_d3(data, x, y, color, size),
            "options": {
                "title": title,
                "xlabel": xlabel,
                "ylabel": ylabel,
                "darkMode": self.dark_mode,
                "colors": ColorTheme.COLORS
            }
        }
        
        # Add any additional options from kwargs
        spec["options"].update(kwargs)
        
        return spec
    
    def _prepare_data_for_d3(self, data, x, y, color, size):
        """Prepare data in a format suitable for D3.js."""
        if isinstance(data, pd.DataFrame):
            # Convert DataFrame to list of records
            return data.to_dict(orient='records')
        elif isinstance(data, dict):
            # Handle dictionary format
            if 'nodes' in data and 'edges' in data:
                # Network data
                return data
            elif all(isinstance(data[key], (list, np.ndarray)) for key in data):
                # Convert columns format to records
                records = []
                keys = list(data.keys())
                for i in range(len(data[keys[0]])):
                    record = {}
                    for key in keys:
                        record[key] = data[key][i]
                    records.append(record)
                return records
            else:
                # Simple key-value pairs
                return [{"key": k, "value": v} for k, v in data.items()]
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                # Already in the right format
                return data
            else:
                # Convert simple list to key-value pairs
                return [{"index": i, "value": v} for i, v in enumerate(data)]
        else:
            # Default fallback
            logger.warning(f"Unsupported data type for D3: {type(data)}")
            return []

    def _create_error_chart(self, error_message: str) -> ChartObject:
        """Create a simple error chart showing the error message."""
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.text(0.5, 0.5, f"Chart Generation Error:\n{error_message}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            ax.set_axis_off()
            return fig
        else:
            # Text-based fallback if matplotlib is not available
            return {"error": error_message}
    
    def chart_to_image(self, chart: ChartObject, backend: Backend, 
                      format: str = 'png', width: int = 800, height: int = 600,
                      scale: float = 1.0) -> bytes:
        """
        Convert a chart to an image format for embedding in reports.
        
        Args:
            chart: The chart object
            backend: The backend used to create the chart
            format: Output format ('png', 'jpg', 'svg', 'pdf')
            width: Image width in pixels
            height: Image height in pixels
            scale: Scale factor for resolution
            
        Returns:
            bytes: The image data
        """
        try:
            if backend == Backend.PLOTLY and PLOTLY_AVAILABLE:
                buf = io.BytesIO()
                if format == 'svg':
                    chart.write_image(buf, format='svg', width=width, height=height, scale=scale)
                elif format == 'pdf':
                    chart.write_image(buf, format='pdf', width=width, height=height, scale=scale)
                else:
                    chart.write_image(buf, format=format, width=width, height=height, scale=scale)
                buf.seek(0)
                return buf.getvalue()
                
            elif backend == Backend.MATPLOTLIB and MATPLOTLIB_AVAILABLE:
                buf = io.BytesIO()
                chart.savefig(buf, format=format, dpi=100*scale, bbox_inches='tight')
                buf.seek(0)
                return buf.getvalue()
                
            elif backend == Backend.ALTAIR and ALTAIR_AVAILABLE:
                if format == 'svg':
                    return chart.to_json().encode('utf-8')
                else:
                    # Convert to vega-lite spec and render with headless browser
                    try:
                        import selenium
                        from selenium import webdriver
                        
                        # Get chart as HTML
                        html = chart.to_html()
                        
                        # Use selenium to render and capture
                        options = webdriver.ChromeOptions()
                        options.add_argument('--headless')
                        options.add_argument('--no-sandbox')
                        options.add_argument('--disable-dev-shm-usage')
                        
                        driver = webdriver.Chrome(options=options)
                        driver.get(f"data:text/html;charset=utf-8,{html}")
                        
                        # Wait for chart to render
                        import time
                        time.sleep(2)
                        
                        # Capture screenshot
                        img_data = driver.get_screenshot_as_png()
                        driver.quit()
                        
                        return img_data
                    except Exception as e:
                        logger.error(f"Error rendering Altair chart: {e}")
                        # Fallback - render simple image with matplotlib
                        return self._error_image_fallback(format, width, height, "Altair rendering failed")
                        
            elif backend == Backend.D3:
                # For D3 specs, render a simple placeholder with matplotlib
                return self._error_image_fallback(format, width, height, "Interactive visualization available only in dashboard view")
                
            else:
                logger.error(f"Unsupported backend for image conversion: {backend}")
                return self._error_image_fallback(format, width, height, "Unsupported visualization backend")
                
        except Exception as e:
            logger.error(f"Error converting chart to image: {e}")
            logger.error(traceback.format_exc())
            return self._error_image_fallback(format, width, height, str(e))
    
    def _error_image_fallback(self, format: str, width: int, height: int, message: str) -> bytes:
        """Generate a fallback error image."""
        if MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(width/100, height/100))
            ax.text(0.5, 0.5, f"Chart Generation Error:\n{message}",
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            ax.set_axis_off()
            
            buf = io.BytesIO()
            fig.savefig(buf, format=format, dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            return buf.getvalue()
        else:
            # Ultimate fallback - generate a simple SVG
            if format == 'svg':
                svg = f"""
                <svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
                    <rect width="100%" height="100%" fill="#f8f9fa" />
                    <text x="50%" y="50%" font-family="Arial" font-size="16" fill="red" text-anchor="middle">
                        Chart Generation Error:
                        <tspan x="50%" dy="20">{message}</tspan>
                    </text>
                </svg>
                """
                return svg.encode('utf-8')
            else:
                # For other formats, return empty bytes with error logged
                logger.error(f"Cannot generate fallback image: {message}")
                return b''
    
    def chart_to_base64(self, chart: ChartObject, backend: Backend, 
                       format: str = 'png', width: int = 800, height: int = 600,
                       scale: float = 1.0) -> str:
        """
        Convert a chart to a base64-encoded string for embedding in HTML.
        
        Args:
            chart: The chart object
            backend: The backend used to create the chart
            format: Output format ('png', 'jpg', 'svg', 'pdf')
            width: Image width in pixels
            height: Image height in pixels
            scale: Scale factor for resolution
            
        Returns:
            str: Base64-encoded image string
        """
        image_data = self.chart_to_image(chart, backend, format, width, height, scale)
        
        if not image_data:
            return ""
            
        if format == 'svg':
            # SVG is already text, so encode differently
            if isinstance(image_data, bytes):
                svg_text = image_data.decode('utf-8')
            else:
                svg_text = image_data
                
            return f"data:image/svg+xml;base64,{base64.b64encode(svg_text.encode('utf-8')).decode('utf-8')}"
        else:
            return f"data:image/{format};base64,{base64.b64encode(image_data).decode('utf-8')}"

    def generate_scenario_visualization(self, scenario_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate 3D scenario visualization data.
        
        Args:
            scenario_data: List of scenario dictionaries with parameters and outcomes
            
        Returns:
            Dict with chart data and options
        """
        if not scenario_data or not isinstance(scenario_data, list):
            return self._generate_sample_scenario_data()
        
        try:
            # Convert to DataFrame for easier analysis
            df = pd.DataFrame(scenario_data)
            
            # Ensure required columns exist
            if 'final_users' not in df.columns:
                df['final_users'] = 1000
            if 'success_probability' not in df.columns:
                df['success_probability'] = 0.5
            
            # Find important parameters (exclude outcome variables)
            params = [c for c in df.columns if c not in ['final_users', 'success_probability']]
            
            # Calculate correlation with final_users for each parameter
            correlations = {p: abs(df[p].corr(df['final_users'])) for p in params}
            
            # Sort by correlation and take top 3
            top_params = sorted(correlations.items(), key=lambda x: x[1], reverse=True)[:3]
            top_param_names = [tp[0] for tp in top_params]
            
            # If we don't have 3 parameters, add some default ones
            while len(top_param_names) < 3:
                for p in ['churn_rate', 'referral_rate', 'user_growth_rate']:
                    if p not in top_param_names and p in params:
                        top_param_names.append(p)
                        break
                if len(top_param_names) < 3:
                    # If we still don't have 3, add dummy parameter
                    dummy_name = f"param_{len(top_param_names) + 1}"
                    df[dummy_name] = np.random.random(len(df))
                    top_param_names.append(dummy_name)
            
            # Limit to top 3 parameters
            top_param_names = top_param_names[:3]
            
            # Create chart using the best available backend
            chart_data = {
                'x': {'name': top_param_names[0], 'values': df[top_param_names[0]].tolist()},
                'y': {'name': top_param_names[1], 'values': df[top_param_names[1]].tolist()},
                'z': {'name': top_param_names[2], 'values': df[top_param_names[2]].tolist()},
                'color': {'name': 'final_users', 'values': df['final_users'].tolist()},
                'size': {'name': 'success_probability', 'values': df['success_probability'].tolist()}
            }
            
            # Try to create a 3D scatter plot for preview
            try:
                if self.default_backend == Backend.PLOTLY and PLOTLY_AVAILABLE:
                    fig = px.scatter_3d(
                        df, 
                        x=top_param_names[0],
                        y=top_param_names[1], 
                        z=top_param_names[2],
                        color='final_users', 
                        size='success_probability',
                        opacity=0.7,
                        title="Scenario Exploration"
                    )
                    chart_data['figure'] = fig
                elif self.default_backend == Backend.MATPLOTLIB and MATPLOTLIB_AVAILABLE:
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    scatter = ax.scatter(
                        df[top_param_names[0]], 
                        df[top_param_names[1]], 
                        df[top_param_names[2]],
                        c=df['final_users'], 
                        s=df['success_probability'] * 100,
                        cmap='viridis',
                        alpha=0.7
                    )
                    
                    ax.set_xlabel(top_param_names[0])
                    ax.set_ylabel(top_param_names[1])
                    ax.set_zlabel(top_param_names[2])
                    
                    plt.colorbar(scatter, ax=ax, label='Final Users')
                    plt.title("Scenario Exploration")
                    
                    chart_data['figure'] = fig
            except Exception as e:
                logger.warning(f"Error creating 3D scatter plot: {e}")
            
            # Add visualization type
            chart_data['type'] = '3d_scatter'
            
            return chart_data
            
        except Exception as e:
            logger.error(f"Error in generate_scenario_visualization: {e}")
            logger.error(traceback.format_exc())
            return self._generate_sample_scenario_data()
    
    def _generate_sample_scenario_data(self) -> Dict[str, Any]:
        """Generate sample scenario data for demonstration."""
        import numpy as np
        churn_vals = np.linspace(0.02, 0.2, 10)
        referral_vals = np.linspace(0.01, 0.1, 10)
        x = []
        y = []
        z = []
        color = []
        size = []
        
        for c in churn_vals:
            for r in referral_vals:
                g = 0.1
                users = 1000 * (1 + (g + r - c) * 12)
                sp = min(1.0, max(0.0, 0.5 + 0.5 * (r / 0.1 - c / 0.1)))
                x.append(float(c))
                y.append(float(r))
                z.append(float(g))
                color.append(float(users))
                size.append(float(sp))
        
        # Try to create a 3D scatter plot for preview
        try:
            if PLOTLY_AVAILABLE:
                df = pd.DataFrame({
                    'churn_rate': x,
                    'referral_rate': y,
                    'growth_rate': z,
                    'final_users': color,
                    'success_probability': size
                })
                
                fig = px.scatter_3d(
                    df, 
                    x='churn_rate',
                    y='referral_rate', 
                    z='growth_rate',
                    color='final_users', 
                    size='success_probability',
                    opacity=0.7,
                    title="Sample Scenario Exploration"
                )
                
                figure = fig
            elif MATPLOTLIB_AVAILABLE:
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                scatter = ax.scatter(
                    x, y, z,
                    c=color, 
                    s=[s * 100 for s in size],
                    cmap='viridis',
                    alpha=0.7
                )
                
                ax.set_xlabel('churn_rate')
                ax.set_ylabel('referral_rate')
                ax.set_zlabel('growth_rate')
                
                plt.colorbar(scatter, ax=ax, label='Final Users')
                plt.title("Sample Scenario Exploration")
                
                figure = fig
            else:
                figure = None
        except Exception as e:
            logger.warning(f"Error creating sample 3D scatter plot: {e}")
            figure = None
        
        return {
            'type': '3d_scatter',
            'x': {'name': 'churn_rate', 'values': x},
            'y': {'name': 'referral_rate', 'values': y},
            'z': {'name': 'growth_rate', 'values': z},
            'color': {'name': 'final_users', 'values': color},
            'size': {'name': 'success_probability', 'values': size},
            'figure': figure
        }
    
    def generate_cohort_visualization(self, cohort_data: Any) -> Dict[str, Any]:
        """
        Generate visualization data for cohort analysis.
        
        Args:
            cohort_data: Cohort analysis data (DataFrame or dict)
            
        Returns:
            Dict with chart data and options
        """
        if not cohort_data:
            return self._generate_sample_cohort_data()
        
        try:
            # Extract retention data if available
            ret = None
            if hasattr(cohort_data, 'retention'):
                ret = cohort_data.retention
            elif isinstance(cohort_data, dict) and 'retention' in cohort_data:
                ret = cohort_data['retention']
            
            if ret is None or not isinstance(ret, pd.DataFrame):
                logger.warning("Invalid cohort data format, using sample data")
                return self._generate_sample_cohort_data()
            
            # Convert index and columns to strings to ensure JSON serialization
            cohorts = ret.index.astype(str).tolist()
            periods = [str(c) for c in ret.columns.tolist()]
            
            # Create data in long format for 3D surface
            x = []  # Cohort indices
            y = []  # Period indices
            z = []  # Retention values
            
            for ci, cval in enumerate(cohorts):
                for pi, pval in enumerate(periods):
                    x.append(ci)
                    y.append(pi)
                    z.append(float(ret.iloc[ci, pi]))
            
            # Try to create a 3D surface plot for preview
            figure = None
            try:
                if PLOTLY_AVAILABLE:
                    # Reshape z values into a matrix for Plotly surface
                    z_matrix = np.zeros((len(cohorts), len(periods)))
                    for cohort_idx, period_idx, value in zip(x, y, z):
                        z_matrix[cohort_idx, period_idx] = value
                    
                    # Create figure with Plotly
                    fig = go.Figure(data=[go.Surface(z=z_matrix, x=periods, y=cohorts)])
                    
                    # Update layout
                    fig.update_layout(
                        title='Cohort Retention Analysis',
                        scene=dict(
                            xaxis_title='Period',
                            yaxis_title='Cohort',
                            zaxis_title='Retention (%)'
                        ),
                        margin=dict(l=0, r=0, b=0, t=30)
                    )
                    
                    figure = fig
                elif MATPLOTLIB_AVAILABLE:
                    # Reshape z values into a matrix for Matplotlib
                    z_matrix = np.zeros((len(cohorts), len(periods)))
                    for cohort_idx, period_idx, value in zip(x, y, z):
                        z_matrix[cohort_idx, period_idx] = value
                    
                    # Create figure with Matplotlib
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    
                    x_mesh, y_mesh = np.meshgrid(np.arange(len(periods)), np.arange(len(cohorts)))
                    
                    surf = ax.plot_surface(
                        x_mesh, y_mesh, z_matrix,
                        cmap='viridis',
                        linewidth=0,
                        antialiased=False
                    )
                    
                    # Set labels and title
                    ax.set_xlabel('Period')
                    ax.set_ylabel('Cohort')
                    ax.set_zlabel('Retention (%)')
                    ax.set_title('Cohort Retention Analysis')
                    
                    # Set tick labels
                    ax.set_xticks(np.arange(len(periods)))
                    ax.set_xticklabels(periods)
                    ax.set_yticks(np.arange(len(cohorts)))
                    ax.set_yticklabels(cohorts)
                    
                    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                    
                    figure = fig
            except Exception as e:
                logger.warning(f"Error creating 3D surface plot: {e}")
            
            # Return the visualization data
            return {
                'type': '3d_surface',
                'x': {'name': 'Cohort', 'values': x, 'labels': cohorts},
                'y': {'name': 'Period', 'values': y, 'labels': periods},
                'z': {'name': 'Retention(%)', 'values': z},
                'figure': figure
            }
            
        except Exception as e:
            logger.error(f"Error in generate_cohort_visualization: {e}")
            logger.error(traceback.format_exc())
            return self._generate_sample_cohort_data()
    
    def _generate_sample_cohort_data(self) -> Dict[str, Any]:
        """Generate sample cohort data for demonstration."""
        cohorts = [f"2023-{m:02d}" for m in range(1, 7)]
        periods = [str(p) for p in range(6)]
        x = []
        y = []
        z = []
        
        for ci, cval in enumerate(cohorts):
            base = 100 - ci * 5
            for pi, pval in enumerate(periods):
                ret = base * (0.85 ** pi)
                x.append(ci)
                y.append(pi)
                z.append(ret)
        
        # Try to create a 3D surface plot for preview
        figure = None
        try:
            if PLOTLY_AVAILABLE:
                # Reshape z values into a matrix for Plotly surface
                z_matrix = np.zeros((len(cohorts), len(periods)))
                for cohort_idx, period_idx, value in zip(x, y, z):
                    z_matrix[cohort_idx, period_idx] = value
                
                # Create figure with Plotly
                fig = go.Figure(data=[go.Surface(z=z_matrix, x=periods, y=cohorts)])
                
                # Update layout
                fig.update_layout(
                    title='Sample Cohort Retention Analysis',
                    scene=dict(
                        xaxis_title='Period',
                        yaxis_title='Cohort',
                        zaxis_title='Retention (%)'
                    ),
                    margin=dict(l=0, r=0, b=0, t=30)
                )
                
                figure = fig
            elif MATPLOTLIB_AVAILABLE:
                # Reshape z values into a matrix for Matplotlib
                z_matrix = np.zeros((len(cohorts), len(periods)))
                for cohort_idx, period_idx, value in zip(x, y, z):
                    z_matrix[cohort_idx, period_idx] = value
                
                # Create figure with Matplotlib
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                x_mesh, y_mesh = np.meshgrid(np.arange(len(periods)), np.arange(len(cohorts)))
                
                surf = ax.plot_surface(
                    x_mesh, y_mesh, z_matrix,
                    cmap='viridis',
                    linewidth=0,
                    antialiased=False
                )
                
                # Set labels and title
                ax.set_xlabel('Period')
                ax.set_ylabel('Cohort')
                ax.set_zlabel('Retention (%)')
                ax.set_title('Sample Cohort Retention Analysis')
                
                # Set tick labels
                ax.set_xticks(np.arange(len(periods)))
                ax.set_xticklabels(periods)
                ax.set_yticks(np.arange(len(cohorts)))
                ax.set_yticklabels(cohorts)
                
                plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
                
                figure = fig
        except Exception as e:
            logger.warning(f"Error creating sample 3D surface plot: {e}")
        
        return {
            'type': '3d_surface',
            'x': {'name': 'Cohort', 'values': x, 'labels': cohorts},
            'y': {'name': 'Period', 'values': y, 'labels': periods},
            'z': {'name': 'Retention(%)', 'values': z},
            'figure': figure
        }

    def generate_network_visualization(self, interaction_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Generate network visualization from interaction data.
        
        Args:
            interaction_data: DataFrame with columns user_id_from, user_id_to
            
        Returns:
            Dict with network graph data
        """
        if not isinstance(interaction_data, pd.DataFrame):
            return self._generate_sample_network_data()
        
        try:
            # Extract user IDs
            from_users = interaction_data['user_id_from'].unique()
            to_users = interaction_data['user_id_to'].unique()
            all_users = np.union1d(from_users, to_users)
            
            # Create nodes with size based on interaction count
            nodes = []
            for uid in all_users:
                outgoing = interaction_data[interaction_data['user_id_from'] == uid].shape[0]
                incoming = interaction_data[interaction_data['user_id_to'] == uid].shape[0]
                
                # Size based on log of interaction count
                size = float(np.log1p(outgoing + incoming) + 1)
                
                # Color based on ratio of incoming to total interactions
                color = (incoming / (outgoing + incoming)) if (outgoing + incoming) > 0 else 0.5
                
                # Add position coordinates for visualizing the network
                x = np.random.random()
                y = np.random.random()
                
                nodes.append({
                    'id': str(uid),
                    'size': size,
                    'color': float(color),
                    'x': float(x),
                    'y': float(y)
                })
            
            # Create edges
            edges = []
            for _, row in interaction_data.iterrows():
                edges.append({
                    'source': str(row['user_id_from']),
                    'target': str(row['user_id_to']),
                    'weight': 1
                })
            
            # Combine duplicate edges by summing weights
            edge_dict = {}
            for e in edges:
                key = f"{e['source']}_{e['target']}"
                if key in edge_dict:
                    edge_dict[key]['weight'] += e['weight']
                else:
                    edge_dict[key] = e
            
            unique_edges = list(edge_dict.values())
            
            # Try to create a network visualization preview
            figure = None
            try:
                if PLOTLY_AVAILABLE:
                    # Create a network graph with plotly
                    edges_x = []
                    edges_y = []
                    
                    # Create a lookup for node positions
                    node_positions = {n['id']: (n['x'], n['y']) for n in nodes}
                    
                    # Create edge path data
                    for edge in unique_edges:
                        source = edge['source']
                        target = edge['target']
                        if source in node_positions and target in node_positions:
                            x0, y0 = node_positions[source]
                            x1, y1 = node_positions[target]
                            edges_x.extend([x0, x1, None])
                            edges_y.extend([y0, y1, None])
                    
                    # Create edge trace
                    edge_trace = go.Scatter(
                        x=edges_x, y=edges_y,
                        line=dict(width=0.5, color='#888'),
                        hoverinfo='none',
                        mode='lines'
                    )
                    
                    # Create node trace
                    node_trace = go.Scatter(
                        x=[n['x'] for n in nodes],
                        y=[n['y'] for n in nodes],
                        mode='markers',
                        hoverinfo='text',
                        text=[n['id'] for n in nodes],
                        marker=dict(
                            showscale=True,
                            colorscale='YlGnBu',
                            color=[n['color'] for n in nodes],
                            size=[n['size'] * 10 for n in nodes],
                            colorbar=dict(
                                thickness=15,
                                title='Interaction Balance',
                                xanchor='left',
                                titleside='right'
                            ),
                            line=dict(width=2)
                        )
                    )
                    
                    # Create figure
                    fig = go.Figure(data=[edge_trace, node_trace],
                                   layout=go.Layout(
                                       title='Network Analysis',
                                       showlegend=False,
                                       hovermode='closest',
                                       margin=dict(b=20, l=5, r=5, t=40),
                                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                                   ))
                    
                    figure = fig
            except Exception as e:
                logger.warning(f"Error creating network visualization: {e}")
            
            # Return the network data
            return {
                'type': 'network_graph',
                'nodes': nodes,
                'edges': unique_edges,
                'figure': figure
            }
            
        except Exception as e:
            logger.error(f"Error in generate_network_visualization: {e}")
            logger.error(traceback.format_exc())
            return self._generate_sample_network_data()
    
    def _generate_sample_network_data(self) -> Dict[str, Any]:
        """Generate sample network data for demonstration."""
        import numpy as np
        
        # Create random nodes
        nodes = []
        for i in range(30):
            size = np.random.uniform(1, 5)
            color = np.random.uniform(0, 1)
            x = np.random.random()
            y = np.random.random()
            nodes.append({
                'id': f"user_{i}",
                'size': float(size),
                'color': float(color),
                'x': float(x),
                'y': float(y)
            })
        
        # Create random edges
        edges = []
        for i in range(50):
            s = np.random.randint(0, 30)
            t = np.random.randint(0, 30)
            if s != t:
                edges.append({
                    'source': f"user_{s}",
                    'target': f"user_{t}",
                    'weight': float(np.random.uniform(0.5, 2.0))
                })
        
        # Try to create a network visualization preview
        figure = None
        try:
            if PLOTLY_AVAILABLE:
                # Create a network graph with plotly
                edges_x = []
                edges_y = []
                
                # Create edge path data
                for edge in edges:
                    source_id = edge['source']
                    target_id = edge['target']
                    source_idx = int(source_id.split('_')[1])
                    target_idx = int(target_id.split('_')[1])
                    x0, y0 = nodes[source_idx]['x'], nodes[source_idx]['y']
                    x1, y1 = nodes[target_idx]['x'], nodes[target_idx]['y']
                    edges_x.extend([x0, x1, None])
                    edges_y.extend([y0, y1, None])
                
                # Create edge trace
                edge_trace = go.Scatter(
                    x=edges_x, y=edges_y,
                    line=dict(width=0.5, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                
                # Create node trace
                node_trace = go.Scatter(
                    x=[n['x'] for n in nodes],
                    y=[n['y'] for n in nodes],
                    mode='markers',
                    hoverinfo='text',
                    text=[n['id'] for n in nodes],
                    marker=dict(
                        showscale=True,
                        colorscale='YlGnBu',
                        color=[n['color'] for n in nodes],
                        size=[n['size'] * 10 for n in nodes],
                        colorbar=dict(
                            thickness=15,
                            title='Interaction Balance',
                            xanchor='left',
                            titleside='right'
                        ),
                        line=dict(width=2)
                    )
                )
                
                # Create figure
                fig = go.Figure(data=[edge_trace, node_trace],
                               layout=go.Layout(
                                   title='Sample Network Analysis',
                                   showlegend=False,
                                   hovermode='closest',
                                   margin=dict(b=20, l=5, r=5, t=40),
                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                               ))
                
                figure = fig
        except Exception as e:
            logger.warning(f"Error creating sample network visualization: {e}")
        
        return {
            'type': 'network_graph',
            'nodes': nodes,
            'edges': edges,
            'figure': figure
        }
    
    def generate_camp_radar_chart(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a CAMP framework radar chart from document data.
        
        Args:
            doc: Document with CAMP scores
            
        Returns:
            Dict with chart object and metadata
        """
        try:
            # Extract CAMP scores
            capital_score = doc.get("capital_score", 0)
            advantage_score = doc.get("advantage_score", 0)
            market_score = doc.get("market_score", 0)
            people_score = doc.get("people_score", 0)
            
            # Create radar chart data
            categories = ['Capital Efficiency', 'Market Dynamics', 'Advantage Moat', 'People & Performance']
            values = [capital_score, market_score, advantage_score, people_score]
            
            # Create the radar chart
            chart, backend = self.create_chart(
                ChartType.RADAR,
                data=pd.DataFrame({
                    'Category': categories,
                    'Score': values
                }),
                x='Category',
                y='Score',
                title="CAMP Framework Analysis"
            )
            
            # Return the chart with metadata
            return {
                'type': 'radar',
                'categories': categories,
                'values': values,
                'chart': chart,
                'backend': backend
            }
        except Exception as e:
            logger.error(f"Error generating CAMP radar chart: {e}")
            logger.error(traceback.format_exc())
            
            # Return error chart
            chart, backend = self.create_chart(
                ChartType.RADAR,
                data=pd.DataFrame({
                    'Category': ['Capital', 'Market', 'Advantage', 'People'],
                    'Score': [50, 50, 50, 50]
                }),
                x='Category',
                y='Score',
                title="CAMP Framework Analysis (Error)"
            )
            
            return {
                'type': 'radar',
                'categories': ['Capital', 'Market', 'Advantage', 'People'],
                'values': [50, 50, 50, 50],
                'chart': chart,
                'backend': backend,
                'error': str(e)
            }
    
    def generate_financial_chart(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate financial projection chart.
        
        Args:
            doc: Document with financial forecast data
            
        Returns:
            Dict with chart object and metadata
        """
        try:
            # Extract financial forecast data
            forecast = doc.get("financial_forecast", {})
            
            if not isinstance(forecast, dict) or "months" not in forecast or "revenue" not in forecast:
                # Generate sample data if missing
                months = list(range(24))
                base_revenue = doc.get("monthly_revenue", 50000)
                growth_rate = doc.get("revenue_growth_rate", 0.1) / 100  # Convert from percentage
                
                revenue = [base_revenue * (1 + growth_rate) ** i for i in months]
                
                # Generate profit data (if available in doc, otherwise estimate)
                profit_margin = doc.get("operating_margin_percent", 10) / 100  # Convert from percentage
                profit = [rev * profit_margin for rev in revenue]
                
                # Create financial forecast data
                forecast = {
                    "months": months,
                    "revenue": revenue,
                    "profit": profit
                }
            
            # Extract months, revenue, and profit
            months = forecast.get("months", [])
            revenue = forecast.get("revenue", [])
            profit = forecast.get("profit", []) if "profit" in forecast else None
            
            # Create DataFrame for plotting
            if profit:
                df = pd.DataFrame({
                    "Month": months,
                    "Revenue": revenue,
                    "Profit": profit
                })
                
                # Create line chart with both revenue and profit
                chart, backend = self.create_chart(
                    ChartType.LINE,
                    data=df,
                    x="Month",
                    y=["Revenue", "Profit"],
                    title="Financial Projections"
                )
            else:
                df = pd.DataFrame({
                    "Month": months,
                    "Revenue": revenue
                })
                
                # Create line chart with just revenue
                chart, backend = self.create_chart(
                    ChartType.LINE,
                    data=df,
                    x="Month",
                    y="Revenue",
                    title="Revenue Projection"
                )
            
            # Return the chart with metadata
            return {
                'type': 'line',
                'months': months,
                'revenue': revenue,
                'profit': profit,
                'chart': chart,
                'backend': backend
            }
        except Exception as e:
            logger.error(f"Error generating financial chart: {e}")
            logger.error(traceback.format_exc())
            
            # Return error chart
            months = list(range(12))
            chart, backend = self.create_chart(
                ChartType.LINE,
                data=pd.DataFrame({
                    "Month": months,
                    "Revenue": [50000 * (1.1 ** i) for i in months]
                }),
                x="Month",
                y="Revenue",
                title="Financial Projection (Error)"
            )
            
            return {
                'type': 'line',
                'months': months,
                'revenue': [50000 * (1.1 ** i) for i in months],
                'chart': chart,
                'backend': backend,
                'error': str(e)
            }

    def generate_competitive_chart(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate competitive positioning chart.
        
        Args:
            doc: Document with competitive positioning data
            
        Returns:
            Dict with chart object and metadata
        """
        try:
            # Extract competitive positioning data
            positioning = doc.get("competitive_positioning", {})
            
            if not isinstance(positioning, dict):
                # Return empty chart if no data
                logger.warning("No competitive positioning data available")
                return {
                    'type': 'scatter',
                    'error': "No competitive positioning data available"
                }
            
            # Extract dimensions, company position, and competitor positions
            dimensions = positioning.get("dimensions", ["Innovation", "Market Share"])
            company_position = positioning.get("company_position", {})
            competitor_positions = positioning.get("competitor_positions", {})
            
            if not dimensions or not company_position or not competitor_positions or len(dimensions) < 2:
                # Return empty chart if missing required data
                logger.warning("Incomplete competitive positioning data")
                return {
                    'type': 'scatter',
                    'error': "Incomplete competitive positioning data"
                }
            
            # Pick first two dimensions
            x_dim = dimensions[0]
            y_dim = dimensions[1]
            
            # Create dataset including company and competitors
            data = []
            
            # Add company data point
            company_x = company_position.get(x_dim, 50)
            company_y = company_position.get(y_dim, 50)
            data.append({
                'Company': "Your Company",
                x_dim: company_x,
                y_dim: company_y
            })
            
            # Add competitor data points
            for comp_name, comp_pos in competitor_positions.items():
                comp_x = comp_pos.get(x_dim, 50)
                comp_y = comp_pos.get(y_dim, 50)
                data.append({
                    'Company': comp_name,
                    x_dim: comp_x,
                    y_dim: comp_y
                })
            
            # Create scatter plot
            chart, backend = self.create_chart(
                ChartType.SCATTER,
                data=pd.DataFrame(data),
                x=x_dim,
                y=y_dim,
                color='Company',
                title=f"Competitive Positioning: {x_dim} vs {y_dim}"
            )
            
            # If using Plotly, add quadrant lines
            if backend == Backend.PLOTLY and PLOTLY_AVAILABLE:
                chart.add_hline(y=50, line_dash="dash", line_color="gray")
                chart.add_vline(x=50, line_dash="dash", line_color="gray")
                
                # Set axis ranges
                chart.update_xaxes(range=[0, 100])
                chart.update_yaxes(range=[0, 100])
            
            # Return the chart with metadata
            return {
                'type': 'scatter',
                'x_dimension': x_dim,
                'y_dimension': y_dim,
                'company_position': (company_x, company_y),
                'competitor_positions': {name: (pos.get(x_dim, 50), pos.get(y_dim, 50)) 
                                        for name, pos in competitor_positions.items()},
                'chart': chart,
                'backend': backend
            }
        except Exception as e:
            logger.error(f"Error generating competitive chart: {e}")
            logger.error(traceback.format_exc())
            
            # Return error chart
            chart, backend = self.create_chart(
                ChartType.SCATTER,
                data=pd.DataFrame({
                    'Company': ['Your Company', 'Competitor A', 'Competitor B'],
                    'Innovation': [60, 40, 70],
                    'Market Share': [30, 60, 45]
                }),
                x='Innovation',
                y='Market Share',
                color='Company',
                title="Competitive Positioning (Error)"
            )
            
            return {
                'type': 'scatter',
                'chart': chart,
                'backend': backend,
                'error': str(e)
            }

    def generate_user_growth_chart(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate user growth projection chart.
        
        Args:
            doc: Document with system dynamics data
            
        Returns:
            Dict with chart object and metadata
        """
        try:
            # Extract system dynamics data
            sys_dynamics = doc.get("system_dynamics", {})
            
            if not isinstance(sys_dynamics, dict) or "users" not in sys_dynamics:
                # Generate sample data if missing
                months = list(range(24))
                base_users = doc.get("current_users", 1000)
                growth_rate = doc.get("user_growth_rate", 0.1) / 100  # Convert from percentage
                
                users = [base_users * (1 + growth_rate) ** i for i in months]
                
                # Create system dynamics data
                sys_dynamics = {
                    "months": months,
                    "users": users
                }
            
            # Extract months and users
            if "months" in sys_dynamics:
                months = sys_dynamics.get("months", [])
            else:
                months = list(range(len(sys_dynamics.get("users", []))))
            
            users = sys_dynamics.get("users", [])
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                "Month": months,
                "Users": users
            })
            
            # Create line chart
            chart, backend = self.create_chart(
                ChartType.LINE,
                data=df,
                x="Month",
                y="Users",
                title="User Growth Projection"
            )
            
            # Add trend line if possible
            if backend == Backend.PLOTLY and PLOTLY_AVAILABLE and len(users) > 1:
                import numpy as np
                
                # Calculate trend line
                z = np.polyfit(months, users, 1)
                p = np.poly1d(z)
                trend_line = p(months)
                
                # Add trend line to chart
                chart.add_trace(go.Scatter(
                    x=months,
                    y=trend_line,
                    mode='lines',
                    name='Trend',
                    line=dict(dash='dash', color='red')
                ))
            
            # Return the chart with metadata
            return {
                'type': 'line',
                'months': months,
                'users': users,
                'chart': chart,
                'backend': backend
            }
        except Exception as e:
            logger.error(f"Error generating user growth chart: {e}")
            logger.error(traceback.format_exc())
            
            # Return error chart
            months = list(range(12))
            chart, backend = self.create_chart(
                ChartType.LINE,
                data=pd.DataFrame({
                    "Month": months,
                    "Users": [1000 * (1.1 ** i) for i in months]
                }),
                x="Month",
                y="Users",
                title="User Growth Projection (Error)"
            )
            
            return {
                'type': 'line',
                'months': months,
                'users': [1000 * (1.1 ** i) for i in months],
                'chart': chart,
                'backend': backend,
                'error': str(e)
            }

# Create a convenient function to get an instance of the visualization system
def get_enhanced_visualization(dark_mode: bool = False) -> EnhancedVisualization:
    """
    Get an instance of the enhanced visualization system.
    
    Args:
        dark_mode: Whether to use dark mode theme
        
    Returns:
        EnhancedVisualization instance
    """
    return EnhancedVisualization(dark_mode=dark_mode) 