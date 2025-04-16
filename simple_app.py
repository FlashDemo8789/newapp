import streamlit as st

def main():
    st.set_page_config(
        page_title="Flash DNA Simple Demo",
        page_icon="🧬",
        layout="wide"
    )
    
    st.title("Flash DNA - Simple Demo")
    st.write("This is a simple demonstration app to verify that Streamlit is working correctly.")
    
    # Add a sidebar with basic navigation
    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Select a page:", ["Home", "About", "Demo"])
    
    # Main page content
    if page == "Home":
        st.header("Welcome to Flash DNA")
        st.write("This is a simplified version of the Flash DNA application.")
        
        if st.button("Click Me!"):
            st.success("Button clicked successfully!")
            
    elif page == "About":
        st.header("About Flash DNA")
        st.write("Flash DNA is a data analysis tool for genomic data.")
        st.info("This is a simplified version without ML dependencies.")
        
    elif page == "Demo":
        st.header("Interactive Demo")
        
        # Simple interactive elements
        name = st.text_input("Enter your name:")
        if name:
            st.write(f"Hello, {name}!")
            
        number = st.slider("Select a number:", 0, 100, 50)
        st.write(f"You selected: {number}")
        
        option = st.selectbox("Choose an option:", ["Option 1", "Option 2", "Option 3"])
        st.write(f"You selected: {option}")

if __name__ == "__main__":
    main() 