# Enhanced Visualization System for FlashDNA

This module provides an advanced visualization system for the FlashDNA platform with multiple backends, robust error handling, and consistent styling across all chart types.

## Key Features

1. **Multiple Backend Support**:
   - Primary: Plotly for interactive web visualization
   - Secondary: Matplotlib for static visualizations and PDF reports
   - Tertiary: Altair for alternative visualization styles
   - Fallback: Custom D3.js specification for specialized visualizations

2. **Robust Error Handling**:
   - Automatic fallback to alternative backends if preferred one fails
   - Graceful degradation to simple charts when data is invalid
   - Comprehensive error reporting and logging

3. **Advanced Visualization Types**:
   - Enhanced 3D visualizations (scatter plots, surfaces)
   - Network analysis graphs with interaction weighting
   - Interactive heatmaps and radar charts for CAMP framework
   - Financial projection charts with trend lines

4. **Theme Support**:
   - Light/dark mode toggle
   - Consistent color schemes across all visualizations
   - Customizable styling for branding

5. **Export Capabilities**:
   - Convert charts to various formats (PNG, SVG, PDF)
   - Base64 encoding for embedding in HTML reports
   - JSON specifications for interactive dashboards

6. **Performance Optimizations**:
   - Efficient data transformations
   - Caching for repeated visualizations
   - Optimized rendering for large datasets

## Installation

The enhanced visualization system requires the following packages:

```bash
pip install plotly>=5.3.0 matplotlib>=3.5.0 pandas>=1.3.0 numpy>=1.24.3
```

Optional dependencies for extended functionality:

```bash
pip install altair>=4.2.0
```

## Basic Usage

Basic usage of the enhanced visualization system:

```python
from enhanced_visualization import get_enhanced_visualization, ChartType

# Initialize the visualization system
viz = get_enhanced_visualization(dark_mode=False)

# Create a simple chart
import pandas as pd
data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y1': [10, 15, 13, 17, 20],
    'y2': [5, 7, 10, 12, 15]
})

# Create a line chart
chart, backend = viz.create_chart(
    chart_type=ChartType.LINE,
    data=data,
    x='x',
    y=['y1', 'y2'],
    title='Simple Line Chart',
    xlabel='X Axis',
    ylabel='Y Axis'
)

# For Streamlit
import streamlit as st
st.plotly_chart(chart, use_container_width=True)

# For PDF reports
img_data = viz.chart_to_image(chart, backend, format='png', width=800, height=400)
# Use img_data in your PDF generation library
```

## Integration with Existing Systems

### Dashboard Integration (Streamlit)

To integrate with Streamlit dashboards:

```python
import streamlit as st
from enhanced_visualization import get_enhanced_visualization, ChartType

# In your analysis_flow.py file
viz = get_enhanced_visualization(dark_mode=False)

def render_camp_details_tab(doc):
    # Create and display CAMP radar chart
    result = viz.generate_camp_radar_chart(doc)
    st.plotly_chart(result['chart'], use_container_width=True)
    
    # Rest of the function...
```

### PDF Report Integration

To integrate with the PDF report generator:

```python
from enhanced_visualization import get_enhanced_visualization
from io import BytesIO
from reportlab.platypus import Image

# In your unified_pdf_generator.py file
class InvestorReport:
    def __init__(self, doc, output_path):
        # Initialize visualization system
        self.viz = get_enhanced_visualization(dark_mode=False)
        # Rest of initialization...
        
    def add_camp_radar_chart(self):
        # Generate CAMP radar chart
        result = self.viz.generate_camp_radar_chart(self.doc)
        chart = result['chart']
        backend = result['backend']
        
        # Convert to image for PDF
        chart_image = self.viz.chart_to_image(
            chart=chart,
            backend=backend,
            format='png',
            width=600,
            height=400
        )
        
        # Add to the report
        img = Image(BytesIO(chart_image), width=450, height=300)
        self.story.append(img)
```

## Advanced Features

### 3D Scenario Visualization

The module includes specialized methods for common visualization tasks in FlashDNA:

```python
# Generate 3D scenario visualization
result = viz.generate_scenario_visualization(scenario_data)
st.plotly_chart(result['figure'], use_container_width=True)

# Generate cohort analysis visualization
result = viz.generate_cohort_visualization(cohort_data)
st.plotly_chart(result['figure'], use_container_width=True)

# Generate network visualization
result = viz.generate_network_visualization(interaction_data)
st.plotly_chart(result['figure'], use_container_width=True)
```

### Theme Customization

The visualization system includes a consistent color theme:

```python
# Use dark mode
viz_dark = get_enhanced_visualization(dark_mode=True)

# Get the current theme colors
theme_colors = viz.theme

# Use a specific color from the palette
from enhanced_visualization import ColorTheme
blue_color = ColorTheme.BLUE
```

## Error Handling

The system includes comprehensive error handling:

1. If the preferred backend is not available, it automatically falls back to the next available backend
2. If a chart type is not supported in the current backend, it falls back to a simpler chart type
3. If chart creation fails entirely, it returns an error chart with the error message

## Examples

See the following example files for detailed integration examples:

- `visualization_integration_examples.py` - Examples for dashboard integration
- `pdf_integration_example.py` - Examples for PDF report integration

## Contributing

When extending or modifying the visualization system:

1. Add new chart types to the `ChartType` enum
2. Implement the chart creation in each backend's method
3. Include appropriate error handling and fallbacks
4. Maintain consistent styling across all chart types
5. Add tests for the new chart types

## License

This module is part of the FlashDNA platform and is subject to the same license terms as the platform itself. 