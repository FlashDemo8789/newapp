import io
import base64
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image, ImageDraw, ImageFont
import os
import re
import json
import logging
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import asdict, is_dataclass

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("utils")

def make_json_serializable(obj):
    """
    Convert an object to a JSON serializable format.
    This handles custom classes like SimulationResult.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON serializable version of the object (dict, list, etc.)
    """
    # Handle None
    if obj is None:
        return None
    
    # Handle basic types
    if isinstance(obj, (str, int, float, bool)):
        return obj
    
    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    
    # Handle dictionaries
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    
    # Handle numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Handle pandas DataFrame or Series
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return obj.to_dict()
    
    # Handle dataclasses (like SimulationResult)
    if is_dataclass(obj):
        return make_json_serializable(asdict(obj))
    
    # Handle objects with __dict__ attribute (convert to dict)
    if hasattr(obj, '__dict__'):
        return make_json_serializable(obj.__dict__)
    
    # For other types, try simple conversion or return string representation
    try:
        return dict(obj)
    except:
        return str(obj)

def create_placeholder(width: int, height: int, text: str = "") -> Image.Image:
    """
    Creates a placeholder image with optional text.
    
    Args:
        width: Width of the placeholder image
        height: Height of the placeholder image
        text: Optional text to display in the placeholder
        
    Returns:
        PIL Image object
    """
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    if text:
        draw = ImageDraw.Draw(img)
        
        # Try to load a font, fallback to default if necessary
        try:
            font_size = min(height // 4, 36)  # Scale font size based on height
            font_path = os.path.join(os.path.dirname(__file__), "static", "fonts", "OpenSans-Regular.ttf")
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.load_default()
        except Exception as e:
            logger.warning(f"Font loading error: {e}")
            font = ImageFont.load_default()
        
        # Calculate text position for centering
        text_width, text_height = draw.textsize(text, font=font) if hasattr(draw, 'textsize') else (width // 2, height // 2)
        position = ((width - text_width) // 2, (height - text_height) // 2)
        
        # Draw text
        draw.text(position, text, fill=(70, 130, 180), font=font)
        
        # Add a border
        draw.rectangle([(0, 0), (width - 1, height - 1)], outline=(200, 200, 200))
    
    return img

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extracts text from a PDF file.
    
    Args:
        pdf_file: Uploaded PDF file object from Streamlit
        
    Returns:
        Extracted text as a string
    """
    try:
        # First try using PyMuPDF (fitz) which handles most PDFs well
        import fitz
        
        # Read the file content
        pdf_bytes = pdf_file.read()
        
        # Create a BytesIO object for fitz to read from
        with io.BytesIO(pdf_bytes) as pdf_stream:
            # Open the PDF
            pdf_doc = fitz.open(stream=pdf_stream, filetype="pdf")
            
            # Extract text from each page
            all_text = []
            for page_num in range(len(pdf_doc)):
                page = pdf_doc[page_num]
                text = page.get_text("text")
                all_text.append(text)
            
            return "\n\n".join(all_text)
    
    except ImportError:
        logger.warning("PyMuPDF not available, trying PyPDF2")
        
        try:
            # Fall back to PyPDF2
            import PyPDF2
            
            # Reset the file pointer to the beginning
            if hasattr(pdf_file, 'seek'):
                pdf_file.seek(0)
            
            # Create a PDF reader object
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract text from each page
            all_text = []
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text() or ""
                all_text.append(text)
            
            return "\n\n".join(all_text)
        
        except ImportError:
            logger.error("No PDF extraction libraries available")
            return "ERROR: PDF extraction libraries (PyMuPDF or PyPDF2) not available."
        
        except Exception as e:
            logger.error(f"PyPDF2 extraction error: {str(e)}")
            return f"ERROR extracting PDF text: {str(e)}"
    
    except Exception as e:
        logger.error(f"PDF extraction error: {str(e)}")
        return f"ERROR extracting PDF text: {str(e)}"

def extract_tables_from_pdf(pdf_file) -> List[pd.DataFrame]:
    """
    Extracts tables from a PDF file.
    
    Args:
        pdf_file: Uploaded PDF file object from Streamlit
        
    Returns:
        List of pandas DataFrames containing extracted tables
    """
    try:
        # Try using tabula-py for table extraction
        import tabula
        
        # Read the file content
        pdf_bytes = pdf_file.read()
        
        # Save to a temporary file for tabula to read
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_file.write(pdf_bytes)
        
        # Extract tables
        tables = tabula.read_pdf(temp_filename, pages='all', multiple_tables=True)
        
        # Clean up the temporary file
        os.unlink(temp_filename)
        
        return tables
    
    except ImportError:
        logger.warning("tabula-py not available for table extraction")
        return []
    
    except Exception as e:
        logger.error(f"Table extraction error: {str(e)}")
        return []

def fig_to_base64(fig) -> str:
    """
    Converts a Plotly figure to a base64 encoded PNG string.
    
    Args:
        fig: Plotly figure object
        
    Returns:
        Base64 encoded string of the figure
    """
    try:
        img_bytes = fig.to_image(format="png", scale=2)
        img_base64 = base64.b64encode(img_bytes).decode('ascii')
        return img_base64
    except Exception as e:
        logger.error(f"Error converting figure to base64: {str(e)}")
        return ""

def create_gauge_chart(value: float, min_val: float = 0, max_val: float = 100, 
                       title: str = "Score", threshold_ranges: List[Dict] = None) -> go.Figure:
    """
    Creates a gauge chart for displaying metrics.
    
    Args:
        value: The value to display on the gauge
        min_val: Minimum value on the gauge scale
        max_val: Maximum value on the gauge scale
        title: Title for the gauge
        threshold_ranges: List of dictionaries defining color ranges
                          [{"min": 0, "max": 33, "color": "red"}, ...]
    
    Returns:
        Plotly Figure object
    """
    # Default threshold ranges if none provided
    if threshold_ranges is None:
        threshold_ranges = [
            {"min": min_val, "max": (max_val - min_val) * 0.33 + min_val, "color": "red"},
            {"min": (max_val - min_val) * 0.33 + min_val, "max": (max_val - min_val) * 0.67 + min_val, "color": "gold"},
            {"min": (max_val - min_val) * 0.67 + min_val, "max": max_val, "color": "green"}
        ]
    
    # Create steps for the gauge
    steps = []
    for range_def in threshold_ranges:
        steps.append({
            "range": [range_def["min"], range_def["max"]], 
            "color": range_def["color"]
        })
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": title},
        gauge={
            "axis": {"range": [min_val, max_val]},
            "bar": {"color": "darkblue"},
            "steps": steps,
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": value
            }
        }
    ))
    
    return fig

def create_radar_chart(categories: List[str], values: List[float], 
                       reference_values: List[float] = None,
                       title: str = "Radar Chart") -> go.Figure:
    """
    Creates a radar chart for visualizing multiple metrics.
    
    Args:
        categories: List of category names
        values: List of values for the primary dataset
        reference_values: Optional list of values for a reference dataset
        title: Title for the chart
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    # Add primary dataset
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Your Data'
    ))
    
    # Add reference dataset if provided
    if reference_values is not None:
        fig.add_trace(go.Scatterpolar(
            r=reference_values,
            theta=categories,
            fill='toself',
            name='Reference'
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(max(values), max(reference_values or [0]) + 1)]
            )
        ),
        showlegend=True,
        title=title
    )
    
    return fig

def create_funnel_chart(labels: List[str], values: List[float], 
                        title: str = "Funnel Chart") -> go.Figure:
    """
    Creates a funnel chart for visualizing conversion or filtering processes.
    
    Args:
        labels: List of step labels
        values: List of values for each step
        title: Title for the chart
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(go.Funnel(
        y=labels,
        x=values,
        textinfo="value+percent initial",
        textposition="inside",
        marker={"color": px.colors.sequential.Blues}
    ))
    
    fig.update_layout(title=title)
    
    return fig

def create_heatmap(z_values: List[List[float]], x_labels: List[str], y_labels: List[str],
                   title: str = "Heatmap", colorscale: str = "Blues") -> go.Figure:
    """
    Creates a heatmap for visualizing 2D data.
    
    Args:
        z_values: 2D list of values
        x_labels: Labels for the x-axis
        y_labels: Labels for the y-axis
        title: Title for the chart
        colorscale: Color scale for the heatmap
    
    Returns:
        Plotly Figure object
    """
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        hoverongaps=False
    ))
    
    fig.update_layout(title=title)
    
    return fig

def normalize_data(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalizes specified columns in a DataFrame to the range [0, 1].
    
    Args:
        data: Input DataFrame
        columns: List of column names to normalize
    
    Returns:
        DataFrame with normalized columns
    """
    result = data.copy()
    for column in columns:
        if column in result.columns:
            min_val = result[column].min()
            max_val = result[column].max()
            if max_val > min_val:
                result[column] = (result[column] - min_val) / (max_val - min_val)
            else:
                result[column] = 0  # Handle the case where all values are the same
    
    return result

def filter_outliers(data: pd.DataFrame, columns: List[str], method: str = 'iqr', 
                    threshold: float = 1.5) -> pd.DataFrame:
    """
    Filters outliers from specified columns in a DataFrame.
    
    Args:
        data: Input DataFrame
        columns: List of column names to filter
        method: Method for outlier detection ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        DataFrame with outliers removed
    """
    result = data.copy()
    
    for column in columns:
        if column in result.columns:
            if method == 'iqr':
                # IQR method
                Q1 = result[column].quantile(0.25)
                Q3 = result[column].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                mask = (result[column] >= lower_bound) & (result[column] <= upper_bound)
                result = result[mask]
            
            elif method == 'zscore':
                # Z-score method
                mean = result[column].mean()
                std = result[column].std()
                if std > 0:  # Prevent division by zero
                    z_scores = (result[column] - mean) / std
                    mask = z_scores.abs() <= threshold
                    result = result[mask]
    
    return result

def parse_csv_data(csv_content: str) -> pd.DataFrame:
    """
    Parses CSV data to a DataFrame with robust error handling.
    
    Args:
        csv_content: CSV content as a string
    
    Returns:
        Pandas DataFrame containing the CSV data
    """
    try:
        # Try with different options to handle common CSV variations
        import io
        from pandas.errors import ParserError
        
        # First try with automatic detection
        try:
            return pd.read_csv(io.StringIO(csv_content))
        except ParserError:
            pass
        
        # Try with different separators
        for sep in [',', ';', '\t', '|']:
            try:
                return pd.read_csv(io.StringIO(csv_content), sep=sep)
            except ParserError:
                continue
        
        # Try with different encodings
        for encoding in ['utf-8', 'latin1', 'iso-8859-1']:
            try:
                return pd.read_csv(io.StringIO(csv_content), encoding=encoding)
            except (ParserError, UnicodeDecodeError):
                continue
        
        # If all else fails, try a very permissive approach
        return pd.read_csv(io.StringIO(csv_content), sep=None, engine='python')
    
    except Exception as e:
        logger.error(f"CSV parsing error: {str(e)}")
        # Return an empty DataFrame with a message
        return pd.DataFrame({'Error': [f"Failed to parse CSV: {str(e)}"]})

def convert_to_numeric(df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Converts specified columns (or all if None) to numeric data types.
    
    Args:
        df: Input DataFrame
        columns: List of column names to convert, or None to convert all possible
    
    Returns:
        DataFrame with converted columns
    """
    result = df.copy()
    
    # If no columns specified, try to convert all columns
    if columns is None:
        columns = result.columns
    
    for col in columns:
        if col in result.columns:
            # Try to convert to numeric, coercing errors to NaN
            result[col] = pd.to_numeric(result[col], errors='coerce')
    
    return result

def summarize_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generates a summary of a DataFrame including statistics and data quality metrics.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with summary statistics and metrics
    """
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_values": df.isnull().sum().to_dict(),
        "missing_percentage": (df.isnull().sum() / len(df) * 100).to_dict(),
        "numeric_summary": {},
        "categorical_summary": {}
    }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        summary["numeric_summary"] = df[numeric_cols].describe().to_dict()
    
    # Categorical columns summary
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            summary["categorical_summary"][col] = {
                "unique_values": df[col].nunique(),
                "top_values": df[col].value_counts().head(5).to_dict()
            }
    
    return summary

def detect_anomalies(df: pd.DataFrame, column: str, method: str = 'iqr', 
                    threshold: float = 1.5) -> pd.Series:
    """
    Detects anomalies in a specific column of a DataFrame.
    
    Args:
        df: Input DataFrame
        column: Column name to check for anomalies
        method: Method for anomaly detection ('iqr' or 'zscore')
        threshold: Threshold for anomaly detection
    
    Returns:
        Boolean Series indicating anomalies (True for anomalies)
    """
    if column not in df.columns:
        return pd.Series([False] * len(df))
    
    series = df[column]
    
    if method == 'iqr':
        # IQR method
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    elif method == 'zscore':
        # Z-score method
        mean = series.mean()
        std = series.std()
        if std > 0:  # Prevent division by zero
            z_scores = (series - mean) / std
            return z_scores.abs() > threshold
        else:
            return pd.Series([False] * len(df))
    
    # Default fallback
    return pd.Series([False] * len(df))

def resample_time_series(df: pd.DataFrame, date_column: str, 
                         value_column: str, freq: str = 'M',
                         agg_func: str = 'mean') -> pd.DataFrame:
    """
    Resamples a time series DataFrame to a different frequency.
    
    Args:
        df: Input DataFrame
        date_column: Name of the column containing dates
        value_column: Name of the column containing values
        freq: Frequency to resample to ('D' for daily, 'W' for weekly, 'M' for monthly, etc.)
        agg_func: Aggregation function ('mean', 'sum', 'min', 'max', etc.)
    
    Returns:
        Resampled DataFrame
    """
    try:
        # Ensure the date column is in datetime format
        df_copy = df.copy()
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
        
        # Set the date column as index
        df_copy = df_copy.set_index(date_column)
        
        # Resample the data
        if agg_func == 'mean':
            resampled = df_copy[value_column].resample(freq).mean()
        elif agg_func == 'sum':
            resampled = df_copy[value_column].resample(freq).sum()
        elif agg_func == 'min':
            resampled = df_copy[value_column].resample(freq).min()
        elif agg_func == 'max':
            resampled = df_copy[value_column].resample(freq).max()
        else:
            resampled = df_copy[value_column].resample(freq).mean()
        
        # Convert back to DataFrame
        result = resampled.reset_index()
        result.columns = [date_column, value_column]
        
        return result
    
    except Exception as e:
        logger.error(f"Time series resampling error: {str(e)}")
        return df

def calculate_growth_rates(series: pd.Series, periods: int = 1, 
                          method: str = 'percentage') -> pd.Series:
    """
    Calculates growth rates for a time series.
    
    Args:
        series: Input Series with time series data
        periods: Number of periods to use for calculation
        method: Method for calculation ('percentage', 'log', or 'simple')
    
    Returns:
        Series with growth rates
    """
    if method == 'percentage':
        # Percentage change
        return series.pct_change(periods=periods) * 100
    
    elif method == 'log':
        # Log change (continuously compounded rate)
        return np.log(series / series.shift(periods)) * 100
    
    elif method == 'simple':
        # Simple rate (new - old) / old
        return (series - series.shift(periods)) / series.shift(periods) * 100
    
    else:
        # Default to percentage change
        return series.pct_change(periods=periods) * 100

def create_correlation_heatmap(df: pd.DataFrame, include_cols: List[str] = None,
                              title: str = "Correlation Heatmap") -> go.Figure:
    """
    Creates a correlation heatmap for numeric columns in a DataFrame.
    
    Args:
        df: Input DataFrame
        include_cols: List of columns to include, or None for all numeric columns
        title: Title for the heatmap
    
    Returns:
        Plotly Figure object with the correlation heatmap
    """
    # Select only numeric columns if not specified
    if include_cols is None:
        numeric_df = df.select_dtypes(include=['number'])
    else:
        numeric_df = df[include_cols].select_dtypes(include=['number'])
    
    # Calculate correlation matrix
    corr_matrix = numeric_df.corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu_r',  # Red-Blue reversed (red for negative, blue for positive)
        zmid=0,  # Center the colorscale at 0
        text=np.round(corr_matrix.values, 2),  # Show correlation values
        texttemplate="%{text:.2f}",
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Features",
        yaxis_title="Features",
        height=600,
        width=800
    )
    
    return fig

def perform_pca_analysis(df: pd.DataFrame, n_components: int = 2, 
                         include_cols: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Performs Principal Component Analysis (PCA) on a DataFrame.
    
    Args:
        df: Input DataFrame
        n_components: Number of principal components to calculate
        include_cols: List of columns to include, or None for all numeric columns
    
    Returns:
        Tuple of (DataFrame with PCA results, Dict with PCA metadata)
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Select only numeric columns if not specified
        if include_cols is None:
            numeric_df = df.select_dtypes(include=['number'])
        else:
            numeric_df = df[include_cols].select_dtypes(include=['number'])
        
        # Standardize the data
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_df)
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)
        
        # Create DataFrame with PCA results
        pca_df = pd.DataFrame(
            data=principal_components,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Add metadata
        pca_metadata = {
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_).tolist(),
            "components": pca.components_.tolist(),
            "feature_names": numeric_df.columns.tolist()
        }
        
        return pca_df, pca_metadata
    
    except Exception as e:
        logger.error(f"PCA analysis error: {str(e)}")
        # Return empty DataFrame and metadata
        return pd.DataFrame(), {"error": str(e)}

def calculate_cohort_retention(df: pd.DataFrame, cohort_col: str, 
                              time_col: str, user_col: str) -> pd.DataFrame:
    """
    Calculates cohort retention based on user activity data.
    
    Args:
        df: Input DataFrame with user activity data
        cohort_col: Column name for cohort assignment (e.g., first_purchase_month)
        time_col: Column name for activity time (e.g., purchase_month)
        user_col: Column name for user identifier
    
    Returns:
        DataFrame with cohort retention rates
    """
    try:
        # Ensure cohort and time columns are in datetime format
        df[cohort_col] = pd.to_datetime(df[cohort_col])
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Extract cohort period (typically month)
        df['cohort'] = df[cohort_col].dt.to_period('M')
        df['period'] = df[time_col].dt.to_period('M')
        
        # Calculate periods since first activity
        df['periods_since_cohort'] = (df['period'] - df['cohort']).apply(lambda x: x.n)
        
        # Count unique users by cohort and period
        cohort_data = df.groupby(['cohort', 'periods_since_cohort'])[user_col].nunique().reset_index()
        cohort_counts = cohort_data.pivot_table(index='cohort', columns='periods_since_cohort', values=user_col)
        
        # Calculate retention rates
        cohort_sizes = cohort_counts[0]
        retention_rates = cohort_counts.divide(cohort_sizes, axis=0) * 100
        
        return retention_rates
    
    except Exception as e:
        logger.error(f"Cohort retention calculation error: {str(e)}")
        return pd.DataFrame()

def generate_growth_metrics(user_data: List[int], 
                           periods: List[str] = None) -> Dict[str, Any]:
    """
    Generates growth metrics from a list of user counts.
    
    Args:
        user_data: List of user counts per period
        periods: Optional list of period labels (e.g., months)
    
    Returns:
        Dictionary with growth metrics
    """
    metrics = {}
    
    if len(user_data) < 2:
        return {"error": "Insufficient data for growth metrics"}
    
    # Convert to numpy array for calculations
    user_array = np.array(user_data)
    
    # Basic metrics
    metrics["initial_users"] = float(user_array[0])
    metrics["final_users"] = float(user_array[-1])
    metrics["total_users_gained"] = float(user_array[-1] - user_array[0])
    
    # Growth metrics
    metrics["growth_multiple"] = float(user_array[-1] / user_array[0]) if user_array[0] > 0 else 0
    metrics["total_growth_percent"] = float((user_array[-1] / user_array[0] - 1) * 100) if user_array[0] > 0 else 0
    
    # Average period-over-period growth
    pct_changes = [(user_array[i] / user_array[i-1] - 1) * 100 
                  for i in range(1, len(user_array)) if user_array[i-1] > 0]
    
    metrics["avg_period_growth_pct"] = float(np.mean(pct_changes)) if pct_changes else 0
    metrics["min_period_growth_pct"] = float(np.min(pct_changes)) if pct_changes else 0
    metrics["max_period_growth_pct"] = float(np.max(pct_changes)) if pct_changes else 0

# Compound growth rate
    n_periods = len(user_array) - 1
    if n_periods > 0 and user_array[0] > 0:
        cagr = (user_array[-1] / user_array[0]) ** (1 / n_periods) - 1
        metrics["cagr"] = float(cagr * 100)
    else:
        metrics["cagr"] = 0.0
    
    # Period-by-period data
    if periods is not None and len(periods) == len(user_array):
        period_data = []
        for i in range(len(user_array)):
            period_dict = {
                "period": periods[i],
                "users": float(user_array[i])
            }
            if i > 0:
                period_dict["growth_pct"] = float((user_array[i] / user_array[i-1] - 1) * 100) if user_array[i-1] > 0 else 0
            period_data.append(period_dict)
        metrics["period_data"] = period_data
    
    return metrics

def calculate_ltv_cac_metrics(arpu: float, gross_margin: float, churn_rate: float,
                             cac: float) -> Dict[str, float]:
    """
    Calculates Customer Lifetime Value (LTV) and LTV:CAC metrics.
    
    Args:
        arpu: Average Revenue Per User per month
        gross_margin: Gross margin as a decimal (e.g., 0.7 for 70%)
        churn_rate: Monthly churn rate as a decimal (e.g., 0.05 for 5%)
        cac: Customer Acquisition Cost
    
    Returns:
        Dictionary with LTV and related metrics
    """
    # Validate inputs
    if churn_rate <= 0:
        churn_rate = 0.01  # Prevent division by zero
    
    if gross_margin <= 0:
        gross_margin = 0.01  # Prevent negative LTV
    
    # Calculate LTV components
    monthly_contribution = arpu * gross_margin
    avg_lifetime_months = 1 / churn_rate
    
    # Calculate LTV
    ltv = monthly_contribution * avg_lifetime_months
    
    # Calculate LTV:CAC ratio
    ltv_cac_ratio = ltv / cac if cac > 0 else float('inf')
    
    # Calculate CAC payback period (months)
    cac_payback_months = cac / monthly_contribution if monthly_contribution > 0 else float('inf')
    
    return {
        "arpu": arpu,
        "monthly_contribution": monthly_contribution,
        "avg_lifetime_months": avg_lifetime_months,
        "ltv": ltv,
        "cac": cac,
        "ltv_cac_ratio": ltv_cac_ratio,
        "cac_payback_months": cac_payback_months,
        "gross_margin": gross_margin,
        "churn_rate": churn_rate
    }

def forecast_runway(cash: float, burn_rate: float, 
                   monthly_revenue: float = 0.0,
                   rev_growth: float = 0.0,
                   cost_growth: float = 0.0,
                   max_months: int = 36) -> Tuple[int, float, List[float]]:
    """
    Forecasts cash runway based on current burn rate, revenue, and growth rates.
    
    Args:
        cash: Current cash on hand
        burn_rate: Monthly burn rate (expenses)
        monthly_revenue: Monthly revenue
        rev_growth: Monthly revenue growth rate as a decimal (e.g., 0.1 for 10%)
        cost_growth: Monthly cost growth rate as a decimal (e.g., 0.05 for 5%)
        max_months: Maximum number of months to forecast
    
    Returns:
        Tuple of (runway in months, ending cash balance, monthly cash flow list)
    """
    flow = []
    runway = -1
    remaining_cash = cash
    revenue = monthly_revenue
    expenses = burn_rate
    
    for month in range(1, max_months + 1):
        net_burn = expenses - revenue
        remaining_cash -= net_burn
        flow.append(remaining_cash)
        
        # Check if we've run out of cash
        if remaining_cash < 0 and runway < 0:
            runway = month
        
        # Apply growth rates for next month
        revenue *= (1 + rev_growth)
        expenses *= (1 + cost_growth)
    
    # If we never ran out of cash
    if runway < 0:
        runway = max_months
    
    return runway, remaining_cash, flow

def calculate_virality_metrics(k_factor: float, cycle_length_days: int = 7, 
                              initial_users: int = 1000,
                              cycles: int = 12) -> Dict[str, Any]:
    """
    Calculates virality metrics based on K-factor.
    
    Args:
        k_factor: Viral coefficient (average number of new users each user brings)
        cycle_length_days: Length of a viral cycle in days
        initial_users: Number of initial users
        cycles: Number of cycles to simulate
    
    Returns:
        Dictionary with virality metrics and projections
    """
    results = {
        "k_factor": k_factor,
        "is_viral": k_factor > 1.0,
        "cycle_length_days": cycle_length_days,
        "initial_users": initial_users,
        "users": [initial_users],
        "new_users": [initial_users],
        "days": [0],
        "time_to_10x": None,
        "time_to_100x": None
    }
    
    total_users = initial_users
    
    for cycle in range(1, cycles):
        new_users = results["users"][-1] * k_factor
        total_users += new_users
        
        results["new_users"].append(new_users)
        results["users"].append(total_users)
        results["days"].append(cycle * cycle_length_days)
        
        # Track time to reach growth milestones
        if results["time_to_10x"] is None and total_users >= initial_users * 10:
            results["time_to_10x"] = cycle * cycle_length_days
        
        if results["time_to_100x"] is None and total_users >= initial_users * 100:
            results["time_to_100x"] = cycle * cycle_length_days
    
    results["final_users"] = results["users"][-1]
    results["growth_multiple"] = results["final_users"] / initial_users
    
    # Calculate doubling time (in days) if viral
    if k_factor > 1:
        results["doubling_time_days"] = cycle_length_days * (math.log(2) / math.log(k_factor))
    else:
        results["doubling_time_days"] = float('inf')
    
    return results

def analyze_competitive_landscape(company_data: Dict[str, Any], 
                                 competitors: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyzes the competitive landscape based on company and competitor data.
    
    Args:
        company_data: Dictionary with the company's data
        competitors: List of dictionaries with competitor data
    
    Returns:
        Dictionary with competitive analysis metrics
    """
    if not competitors:
        return {"error": "No competitor data provided"}
    
    analysis = {
        "total_competitors": len(competitors),
        "market_share": {},
        "relative_strengths": {},
        "relative_weaknesses": {},
        "positioning": {}
    }
    
    # Extract company metrics for comparison
    company_metrics = {}
    for metric in ["market_share", "growth_rate", "revenue", "customer_count", "product_rating"]:
        if metric in company_data:
            company_metrics[metric] = company_data[metric]
    
    # Calculate total market share
    total_market_share = sum(comp.get("market_share", 0) for comp in competitors)
    if "market_share" in company_metrics:
        total_market_share += company_metrics["market_share"]
    
    # Calculate market share percentages
    if total_market_share > 0:
        analysis["market_share"]["company"] = company_metrics.get("market_share", 0) / total_market_share * 100
        analysis["market_share"]["competitors"] = {}
        
        for comp in competitors:
            comp_name = comp.get("name", "Unknown")
            comp_share = comp.get("market_share", 0) / total_market_share * 100
            analysis["market_share"]["competitors"][comp_name] = comp_share
    
    # Identify relative strengths and weaknesses
    strengths = []
    weaknesses = []
    
    metrics_to_compare = ["growth_rate", "product_rating", "customer_satisfaction", "pricing"]
    for metric in metrics_to_compare:
        if metric in company_metrics:
            company_val = company_metrics[metric]
            comp_vals = [comp.get(metric, 0) for comp in competitors if metric in comp]
            
            if comp_vals:
                avg_val = sum(comp_vals) / len(comp_vals)
                
                if company_val > avg_val * 1.2:
                    strengths.append(f"Above average {metric}")
                elif company_val < avg_val * 0.8:
                    weaknesses.append(f"Below average {metric}")
    
    analysis["relative_strengths"] = strengths
    analysis["relative_weaknesses"] = weaknesses
    
    # Determine market positioning
    if "market_share" in company_metrics and "growth_rate" in company_metrics:
        ms = company_metrics["market_share"]
        gr = company_metrics["growth_rate"]
        
        avg_ms = sum(comp.get("market_share", 0) for comp in competitors) / len(competitors)
        avg_gr = sum(comp.get("growth_rate", 0) for comp in competitors if "growth_rate" in comp) / len([comp for comp in competitors if "growth_rate" in comp])
        
        if ms > avg_ms and gr > avg_gr:
            analysis["positioning"] = "Market Leader"
        elif ms <= avg_ms and gr > avg_gr:
            analysis["positioning"] = "Fast Grower"
        elif ms > avg_ms and gr <= avg_gr:
            analysis["positioning"] = "Established Player"
        else:
            analysis["positioning"] = "Challenger"
    
    return analysis

def generate_pitch_feedback(pitch_text: str) -> Dict[str, Any]:
    """
    Analyzes pitch text and generates feedback on structure, clarity, and key elements.
    
    Args:
        pitch_text: Text of the pitch to analyze
    
    Returns:
        Dictionary with pitch analysis and feedback
    """
    if not pitch_text or len(pitch_text) < 50:
        return {
            "overall_impression": "Too short to analyze properly",
            "score": 0,
            "strengths": [],
            "weaknesses": ["Pitch text is too short for meaningful analysis"]
        }
    
    # Initialize analysis dict
    analysis = {
        "length": len(pitch_text),
        "word_count": len(pitch_text.split()),
        "sections_identified": [],
        "missing_sections": [],
        "strengths": [],
        "weaknesses": [],
        "clarity_score": 0,
        "completeness_score": 0,
        "persuasiveness_score": 0,
        "overall_score": 0
    }
    
    # Define key pitch sections to check for
    key_sections = [
        "problem",
        "solution",
        "market",
        "business model",
        "competition",
        "team",
        "financials",
        "traction",
        "ask"
    ]
    
    # Check for the presence of key sections
    sections_found = []
    
    for section in key_sections:
        section_patterns = [
            re.compile(r'\b' + section + r'\b', re.IGNORECASE),
            re.compile(r'\b' + section.replace(' ', '') + r'\b', re.IGNORECASE)
        ]
        
        for pattern in section_patterns:
            if pattern.search(pitch_text):
                sections_found.append(section)
                break
    
    # Calculate section presence score (0-100)
    analysis["sections_identified"] = sections_found
    analysis["missing_sections"] = [s for s in key_sections if s not in sections_found]
    analysis["completeness_score"] = min(100, len(sections_found) / len(key_sections) * 100)
    
    # Assess clarity based on sentence length and structure
    sentences = re.split(r'[.!?]+', pitch_text)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    
    if valid_sentences:
        avg_sentence_length = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
        
        # Optimal sentence length range is ~15-20 words
        if 10 <= avg_sentence_length <= 25:
            analysis["clarity_score"] = 85
        elif 7 <= avg_sentence_length <= 30:
            analysis["clarity_score"] = 70
        else:
            analysis["clarity_score"] = 50
    
    # Check for quantitative statements (numbers, percentages, etc.)
    quant_patterns = [
        r'\d+%',
        r'\$\d+',
        r'\d+ million',
        r'\d+ billion',
        r'\d+x',
        r'grew by \d+'
    ]
    
    quant_count = 0
    for pattern in quant_patterns:
        quant_count += len(re.findall(pattern, pitch_text, re.IGNORECASE))
    
    # More quantitative statements = more persuasive
    if quant_count >= 5:
        analysis["persuasiveness_score"] = 90
    elif quant_count >= 3:
        analysis["persuasiveness_score"] = 75
    elif quant_count >= 1:
        analysis["persuasiveness_score"] = 60
    else:
        analysis["persuasiveness_score"] = 40
    
    # Calculate overall score (weighted average)
    analysis["overall_score"] = (
        analysis["completeness_score"] * 0.4 +
        analysis["clarity_score"] * 0.3 +
        analysis["persuasiveness_score"] * 0.3
    )
    
    # Generate strengths
    if analysis["completeness_score"] >= 70:
        analysis["strengths"].append("Comprehensive coverage of key pitch elements")
    
    if analysis["clarity_score"] >= 70:
        analysis["strengths"].append("Clear and concise communication")
    
    if analysis["persuasiveness_score"] >= 70:
        analysis["strengths"].append("Effective use of quantitative data to support claims")
    
    if "problem" in sections_found and "solution" in sections_found:
        analysis["strengths"].append("Clear problem-solution narrative")
    
    if "market" in sections_found and "competition" in sections_found:
        analysis["strengths"].append("Good market context and competitive positioning")
    
    # Generate weaknesses
    if analysis["completeness_score"] < 70:
        analysis["weaknesses"].append(f"Missing key sections: {', '.join(analysis['missing_sections'])}")
    
    if analysis["clarity_score"] < 70:
        if avg_sentence_length > 25:
            analysis["weaknesses"].append("Sentences are too long, reducing clarity")
        elif avg_sentence_length < 10:
            analysis["weaknesses"].append("Sentences are too short, potentially oversimplifying")
    
    if analysis["persuasiveness_score"] < 70:
        analysis["weaknesses"].append("Add more quantitative data to strengthen claims")
    
    # Set overall impression
    if analysis["overall_score"] >= 85:
        analysis["overall_impression"] = "Excellent pitch with strong structure and persuasive elements"
    elif analysis["overall_score"] >= 70:
        analysis["overall_impression"] = "Solid pitch with good structure but some areas for improvement"
    elif analysis["overall_score"] >= 50:
        analysis["overall_impression"] = "Adequate pitch that covers the basics but needs refinement"
    else:
        analysis["overall_impression"] = "Needs significant improvement in structure, clarity, and persuasiveness"
    
    return analysis

def run_monte_carlo_simulation(baseline_values: Dict[str, float], 
                               num_simulations: int = 1000,
                               output_metrics: List[str] = None) -> Dict[str, Any]:
    """
    Runs a Monte Carlo simulation for startup metrics with uncertainty.
    
    Args:
        baseline_values: Dictionary of baseline values for each input parameter
        num_simulations: Number of simulations to run
        output_metrics: List of output metrics to calculate (if None, uses default metrics)
    
    Returns:
        Dictionary with simulation results and statistics
    """
    import numpy as np
    
    # If no output metrics specified, use defaults
    if output_metrics is None:
        output_metrics = ["success_probability", "runway", "final_users", "ltv_cac_ratio"]
    
    # Define parameter ranges for simulation
    param_distributions = {}
    for param, value in baseline_values.items():
        if param == "churn_rate":
            # Beta distribution for rates (0-1)
            alpha = 2
            beta = 20 if value <= 0 or value >= 1 else (2 / value) - 2
            param_distributions[param] = {"type": "beta", "alpha": alpha, "beta": beta}
        
        elif param == "gross_margin":
            # Beta distribution for percentages (0-1)
            alpha = value * 10
            beta = (1 - value) * 10
            param_distributions[param] = {"type": "beta", "alpha": alpha, "beta": beta}
        
        elif param in ["user_growth_rate", "revenue_growth_rate"]:
            # Lognormal distribution for growth rates (positive, right-skewed)
            mu = np.log(max(0.01, value))
            sigma = 0.5
            param_distributions[param] = {"type": "lognormal", "mu": mu, "sigma": sigma}
        
        elif param in ["burn_rate", "monthly_revenue", "cac", "arpu"]:
            # Normal distribution for monetary values (symmetric around baseline)
            mu = value
            sigma = value * 0.2  # 20% standard deviation
            param_distributions[param] = {"type": "normal", "mu": mu, "sigma": sigma}
        
        else:
            # Normal distribution as default
            mu = value
            sigma = value * 0.2 if value != 0 else 1.0
            param_distributions[param] = {"type": "normal", "mu": mu, "sigma": sigma}
    
    # Generate samples for each parameter
    samples = {}
    for param, dist in param_distributions.items():
        if dist["type"] == "normal":
            samples[param] = np.random.normal(dist["mu"], dist["sigma"], num_simulations)
        elif dist["type"] == "lognormal":
            samples[param] = np.random.lognormal(dist["mu"], dist["sigma"], num_simulations)
        elif dist["type"] == "beta":
            samples[param] = np.random.beta(dist["alpha"], dist["beta"], num_simulations)
        elif dist["type"] == "uniform":
            samples[param] = np.random.uniform(dist["min"], dist["max"], num_simulations)
    
    # Calculate output metrics for each simulation
    results = {metric: np.zeros(num_simulations) for metric in output_metrics}
    
    for i in range(num_simulations):
        # Extract parameters for this simulation
        sim_params = {param: samples[param][i] for param in samples}
        
        # Calculate metrics using parameter combination
        for metric in output_metrics:
            if metric == "success_probability":
                # Example formula for success probability
                churn = sim_params.get("churn_rate", 0.05)
                growth = sim_params.get("user_growth_rate", 0.1)
                ltv_cac = sim_params.get("ltv", 0) / sim_params.get("cac", 1) if sim_params.get("cac", 0) > 0 else 2
                
                # Simple weighted formula
                sp = (
                    (1 - min(1, churn * 10)) * 0.3 +  # Lower churn is good
                    min(1, growth * 5) * 0.3 +  # Higher growth is good
                    min(1, ltv_cac / 3) * 0.4  # Higher LTV:CAC is good
                ) * 100
                
                results[metric][i] = sp
            
            elif metric == "runway":
                # Simplified runway calculation
                cash = sim_params.get("current_cash", 500000)
                burn = sim_params.get("burn_rate", 50000)
                revenue = sim_params.get("monthly_revenue", 10000)
                
                net_burn = max(0, burn - revenue)
                results[metric][i] = cash / net_burn if net_burn > 0 else float('inf')
            
            elif metric == "final_users":
                # Simplified user growth calculation
                users = sim_params.get("current_users", 1000)
                growth = sim_params.get("user_growth_rate", 0.1)
                churn = sim_params.get("churn_rate", 0.05)
                months = 12
                
                for _ in range(months):
                    users = users * (1 + growth - churn)
                
                results[metric][i] = users
            
            elif metric == "ltv_cac_ratio":
                # LTV:CAC calculation
                arpu = sim_params.get("arpu", 100)
                margin = sim_params.get("gross_margin", 0.7)
                churn = sim_params.get("churn_rate", 0.05)
                cac = sim_params.get("cac", 300)
                
                monthly_contribution = arpu * margin
                ltv = monthly_contribution / churn if churn > 0 else 0
                
                results[metric][i] = ltv / cac if cac > 0 else 0
    
    # Compile summary statistics
    summary = {}
    for metric in output_metrics:
        metric_results = results[metric]
        
        summary[metric] = {
            "mean": float(np.mean(metric_results)),
            "median": float(np.median(metric_results)),
            "std": float(np.std(metric_results)),
            "min": float(np.min(metric_results)),
            "max": float(np.max(metric_results)),
            "percentiles": {
                "10": float(np.percentile(metric_results, 10)),
                "25": float(np.percentile(metric_results, 25)),
                "50": float(np.percentile(metric_results, 50)),
                "75": float(np.percentile(metric_results, 75)),
                "90": float(np.percentile(metric_results, 90))
            }
        }
    
    # Calculate sensitivity/correlations
    sensitivity = {}
    for metric in output_metrics:
        sensitivity[metric] = {}
        for param in samples:
            corr = np.corrcoef(samples[param], results[metric])[0, 1]
            sensitivity[metric][param] = float(corr)
    
    return {
        "num_simulations": num_simulations,
        "summary": summary,
        "sensitivity": sensitivity,
        "baseline_values": baseline_values
    }