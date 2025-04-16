"""
FlashDNA Startup Analysis Platform
Main application entry point with all tab implementations
"""
import streamlit as st
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("flashdna_app")

# Set page configuration
st.set_page_config(
    page_title="FlashDNA Startup Analysis",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Card styling */
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24);
        margin-bottom: 1.5rem;
    }
    
    /* Animation for components */
    .animate-panel {
        animation: slideFadeIn 0.5s ease-out forwards;
        opacity: 0;
        transform: translateY(20px);
    }
    
    @keyframes slideFadeIn {
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    /* Tab container styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f1f5f9;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3A86FF;
        color: white;
    }
</style>
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
""", unsafe_allow_html=True)

def main():
    """Main application entry point"""
    # Display application header
    st.markdown("""
    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
        <h1 style="margin: 0; flex-grow: 1;">🧬 FlashDNA Startup Analysis</h1>
        <span style="color: #64748B; font-size: 0.8rem;">Comprehensive Startup Assessment Platform</span>
    </div>
    <hr style="margin-top: 0; margin-bottom: 2rem;">
    """, unsafe_allow_html=True)
    
    # Load all tab implementations
    try:
        # Add the current directory to path to ensure modules can be found
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        # Import tab loader module
        logger.info("Initializing tab loader")
        from tab_implementations.tab_loader import initialize_tabs
        
        # Initialize all tab implementations
        tab_results = initialize_tabs()
        
        # Log results
        for tab, status in tab_results.items():
            logger.info(f"Tab '{tab}' loaded: {status}")
        
        # Import the main analysis flow module
        logger.info("Importing analysis flow module")
        import analysis_flow
        
        # Start the analysis flow
        analysis_flow.main()
        
    except ImportError as ie:
        st.error(f"Error importing required modules: {str(ie)}")
        st.info("Please ensure all dependencies are installed and the directory structure is correct.")
        logger.error(f"Import error: {str(ie)}")
    except Exception as e:
        st.error(f"Error initializing application: {str(e)}")
        logger.error(f"Application error: {str(e)}", exc_info=True)
        
        # Provide a fallback experience
        _render_fallback_interface()

def _render_fallback_interface():
    """Render a fallback interface if the main app fails to load"""
    st.markdown("""
    ## ⚠️ Application Error
    
    The FlashDNA application encountered an error while loading. Please try the following:
    
    1. Check that all required modules are installed
    2. Verify that the application structure is intact
    3. Ensure that the analysis_flow.py file is in the correct location
    4. Check the logs for detailed error information
    
    ### Running Diagnostics
    
    You can run diagnostics to help identify the issue:
    """)
    
    if st.button("Run Diagnostics"):
        st.write("Checking environment...")
        
        # Check Python version
        import platform
        st.write(f"Python version: {platform.python_version()}")
        
        # Check directory structure
        st.write("Checking directory structure:")
        try:
            base_dir = Path(__file__).parent
            st.write(f"Base directory: {base_dir}")
            
            # Check key files
            files_to_check = [
                "analysis_flow.py",
                "tab_implementations/tab_loader.py",
                "benchmark_tab_implementation.py"
            ]
            
            for file_path in files_to_check:
                full_path = base_dir / file_path
                exists = full_path.exists()
                st.write(f"- {file_path}: {'✅ Found' if exists else '❌ Missing'}")
                
            # Check if modules can be imported
            st.write("Checking module imports:")
            modules_to_check = ["streamlit", "pandas", "numpy", "plotly"]
            
            for module in modules_to_check:
                try:
                    __import__(module)
                    st.write(f"- {module}: ✅ Imported successfully")
                except ImportError:
                    st.write(f"- {module}: ❌ Import failed")
        
        except Exception as e:
            st.error(f"Error during diagnostics: {str(e)}")

if __name__ == "__main__":
    main()
