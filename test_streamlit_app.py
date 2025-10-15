"""
Test Streamlit App - Simplified Version
======================================

A simplified version of the Streamlit app to test basic functionality
and identify any issues.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Test Crop Yield Predictor",
    page_icon="🌾",
    layout="wide"
)

def main():
    """
    Main function for the test app
    """
    st.title("🌾 Test Crop Yield Predictor")
    
    st.write("This is a test version of the crop yield predictor app.")
    
    # Test data loading
    try:
        df = pd.read_csv('preprocessed_agricultural_data.csv')
        st.success(f"✅ Data loaded successfully: {df.shape}")
        
        # Show basic info
        st.subheader("Dataset Information")
        st.write(f"- Total samples: {df.shape[0]}")
        st.write(f"- Features: {df.shape[1]}")
        st.write(f"- Target variable range: {df['Yield'].min():.2f} - {df['Yield'].max():.2f}")
        
        # Show first few rows
        st.subheader("First 5 rows of data")
        st.dataframe(df.head())
        
    except FileNotFoundError:
        st.error("❌ Error: preprocessed_agricultural_data.csv not found")
        st.write("Please ensure the data file exists in the current directory.")
    except Exception as e:
        st.error(f"❌ Error loading data: {str(e)}")
    
    # Test model imports
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        st.success("✅ Machine learning libraries imported successfully")
    except ImportError as e:
        st.error(f"❌ Error importing ML libraries: {str(e)}")
    
    # Test plotly
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        st.success("✅ Plotly imported successfully")
    except ImportError as e:
        st.error(f"❌ Error importing Plotly: {str(e)}")
    
    # Simple input form
    st.subheader("Test Input Form")
    
    col1, col2 = st.columns(2)
    
    with col1:
        temperature = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)
        humidity = st.slider("Humidity (%)", 20.0, 90.0, 60.0)
        rainfall = st.slider("Rainfall (mm)", 50.0, 400.0, 200.0)
    
    with col2:
        ph = st.number_input("pH Level", 4.0, 9.0, 6.5)
        nitrogen = st.slider("Nitrogen (kg/ha)", 20.0, 200.0, 100.0)
        phosphorus = st.slider("Phosphorus (kg/ha)", 5.0, 150.0, 50.0)
    
    # Test prediction button
    if st.button("Test Prediction"):
        st.write("### Test Results")
        st.write(f"Temperature: {temperature}°C")
        st.write(f"Humidity: {humidity}%")
        st.write(f"Rainfall: {rainfall}mm")
        st.write(f"pH: {ph}")
        st.write(f"Nitrogen: {nitrogen} kg/ha")
        st.write(f"Phosphorus: {phosphorus} kg/ha")
        
        # Simple mock prediction
        mock_prediction = (temperature * 0.1 + humidity * 0.05 + rainfall * 0.01 + 
                          ph * 0.5 + nitrogen * 0.02 + phosphorus * 0.03)
        st.success(f"Mock Prediction: {mock_prediction:.2f}")
    
    # Test visualization
    st.subheader("Test Visualization")
    
    try:
        # Create a simple chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=[1, 2, 3, 4, 5],
            y=[2, 3, 4, 5, 6],
            mode='lines+markers',
            name='Test Data'
        ))
        
        fig.update_layout(
            title="Test Chart",
            xaxis_title="X Axis",
            yaxis_title="Y Axis"
        )
        
        st.plotly_chart(fig)
        st.success("✅ Visualization created successfully")
        
    except Exception as e:
        st.error(f"❌ Error creating visualization: {str(e)}")
    
    st.write("---")
    st.write("If all tests pass, the main Streamlit app should work correctly.")

if __name__ == "__main__":
    main()
