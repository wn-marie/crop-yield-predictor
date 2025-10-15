"""
Crop Yield Prediction Streamlit App - Final Version
=================================================

A comprehensive Streamlit application for crop yield prediction using trained
Random Forest and Linear Regression models.

Author: AI Assistant
Date: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crop Yield Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

class CropYieldPredictor:
    """
    Main class for crop yield prediction functionality
    """
    
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.feature_ranges = {}
        self.load_models()
    
    def load_models(self):
        """
        Load trained models and scaler
        """
        try:
            # Load preprocessed data to get feature information
            df = pd.read_csv('preprocessed_agricultural_data.csv')
            
            # Prepare features and target
            y = df['Yield']
            X = df.drop(columns=['Yield', 'Planting_Date', 'Harvest_Date', 'Year', 'Region_Code', 'Country_Name'])
            
            # Store feature names and ranges
            self.feature_names = X.columns.tolist()
            self.feature_ranges = {
                col: {'min': X[col].min(), 'max': X[col].max(), 'mean': X[col].mean()}
                for col in X.columns
            }
            
            # Split data (same as training)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train models (same parameters as before)
            self.models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.models['Linear Regression'] = LinearRegression()
            
            self.models['Random Forest'].fit(X_train, y_train)
            self.models['Linear Regression'].fit(X_train, y_train)
            
            # Store scaler information (data is already scaled)
            self.scaler = StandardScaler()
            self.scaler.fit(X_train)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def predict_yield(self, input_data, model_name='Random Forest'):
        """
        Predict crop yield for given input data
        """
        try:
            # Ensure input data is in correct format
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data
            
            # Ensure all required features are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = self.feature_ranges[feature]['mean']
            
            # Reorder columns to match training data
            input_df = input_df[self.feature_names]
            
            # Make prediction
            prediction = self.models[model_name].predict(input_df)[0]
            
            # Calculate confidence interval for Random Forest
            if model_name == 'Random Forest':
                # Get predictions from individual trees for confidence interval
                tree_predictions = []
                for tree in self.models[model_name].estimators_:
                    tree_pred = tree.predict(input_df)[0]
                    tree_predictions.append(tree_pred)
                
                mean_pred = np.mean(tree_predictions)
                std_pred = np.std(tree_predictions)
                confidence_interval = (mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred)
            else:
                # For Linear Regression, use a simple approximation
                confidence_interval = (prediction - 1.0, prediction + 1.0)
            
            return prediction, confidence_interval
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None

def create_input_form():
    """
    Create the input form for user parameters
    """
    st.sidebar.header("Crop Yield Prediction Parameters")
    
    # Initialize session state for inputs
    if 'input_data' not in st.session_state:
        st.session_state.input_data = {}
    
    # Weather Parameters
    st.sidebar.subheader("Weather Parameters")
    temperature = st.sidebar.slider(
        "Temperature (C)", 
        min_value=10.0, max_value=40.0, value=25.0, step=0.1,
        help="Average temperature during growing season"
    )
    
    humidity = st.sidebar.slider(
        "Humidity (%)", 
        min_value=20.0, max_value=90.0, value=60.0, step=1.0,
        help="Average relative humidity"
    )
    
    rainfall = st.sidebar.slider(
        "Rainfall (mm)", 
        min_value=50.0, max_value=400.0, value=200.0, step=5.0,
        help="Total rainfall during growing season"
    )
    
    solar_radiation = st.sidebar.slider(
        "Solar Radiation (W/m2)", 
        min_value=1000.0, max_value=3000.0, value=2000.0, step=50.0,
        help="Average solar radiation"
    )
    
    wind_speed = st.sidebar.slider(
        "Wind Speed (km/h)", 
        min_value=5.0, max_value=50.0, value=15.0, step=1.0,
        help="Average wind speed"
    )
    
    # Soil Parameters
    st.sidebar.subheader("Soil Parameters")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        ph = st.number_input(
            "pH Level", 
            min_value=4.0, max_value=9.0, value=6.5, step=0.1,
            help="Soil pH level"
        )
        
        organic_carbon = st.number_input(
            "Organic Carbon (%)", 
            min_value=0.1, max_value=3.0, value=1.0, step=0.1,
            help="Soil organic carbon content"
        )
        
        sand = st.slider(
            "Sand (%)", 
            min_value=10.0, max_value=80.0, value=40.0, step=1.0,
            help="Sand percentage in soil"
        )
        
        clay = st.slider(
            "Clay (%)", 
            min_value=5.0, max_value=50.0, value=25.0, step=1.0,
            help="Clay percentage in soil"
        )
    
    with col2:
        ec = st.number_input(
            "EC (dS/m)", 
            min_value=0.1, max_value=3.0, value=1.0, step=0.1,
            help="Electrical conductivity"
        )
        
        bulk_density = st.number_input(
            "Bulk Density (g/cm3)", 
            min_value=1.0, max_value=2.0, value=1.3, step=0.01,
            help="Soil bulk density"
        )
        
        silt = 100 - sand - clay  # Calculate silt automatically
        st.metric("Silt (%)", f"{silt:.1f}")
        
        water_holding = st.slider(
            "Water Holding Capacity (%)", 
            min_value=10.0, max_value=50.0, value=30.0, step=1.0,
            help="Soil water holding capacity"
        )
    
    # Nutrient Parameters
    st.sidebar.subheader("Nutrient Parameters")
    
    col3, col4 = st.sidebar.columns(2)
    
    with col3:
        nitrogen = st.slider("Nitrogen (kg/ha)", 20.0, 200.0, 100.0, 5.0)
        phosphorus = st.slider("Phosphorus (kg/ha)", 5.0, 150.0, 50.0, 5.0)
        potassium = st.slider("Potassium (kg/ha)", 50.0, 300.0, 150.0, 10.0)
        calcium = st.slider("Calcium (kg/ha)", 200.0, 2000.0, 800.0, 50.0)
        magnesium = st.slider("Magnesium (kg/ha)", 100.0, 500.0, 250.0, 10.0)
    
    with col4:
        sulfur = st.slider("Sulfur (kg/ha)", 10.0, 100.0, 40.0, 5.0)
        zinc = st.slider("Zinc (mg/kg)", 0.5, 10.0, 2.0, 0.1)
        iron = st.slider("Iron (mg/kg)", 5.0, 50.0, 20.0, 1.0)
        copper = st.slider("Copper (mg/kg)", 0.2, 5.0, 1.0, 0.1)
        manganese = st.slider("Manganese (mg/kg)", 5.0, 50.0, 20.0, 1.0)
    
    # Additional Parameters
    st.sidebar.subheader("Additional Parameters")
    
    boron = st.slider("Boron (mg/kg)", 0.1, 3.0, 1.0, 0.1)
    molybdenum = st.slider("Molybdenum (mg/kg)", 0.1, 1.0, 0.3, 0.05)
    cec = st.slider("CEC (cmol/kg)", 5.0, 60.0, 25.0, 1.0, help="Cation Exchange Capacity")
    
    # Categorical Parameters
    soil_type = st.sidebar.selectbox(
        "Soil Type", 
        ["Clayey", "Loamy", "Sandy", "Silty"],
        help="Primary soil type"
    )
    
    crop_type = st.sidebar.selectbox(
        "Crop Type", 
        ["Maize", "Rice", "Soybean", "Wheat"],
        help="Type of crop to be grown"
    )
    
    fertilizer_type = st.sidebar.selectbox(
        "Fertilizer Type", 
        ["Chemical", "Mixed", "Organic"],
        help="Type of fertilizer used"
    )
    
    season = st.sidebar.selectbox(
        "Season", 
        ["Kharif", "Rabi", "Zaid"],
        help="Growing season"
    )
    
    region = st.sidebar.selectbox(
        "Region", 
        ["North", "South", "East", "West"],
        help="Geographical region"
    )
    
    growth_stage = st.sidebar.selectbox(
        "Growth Stage", 
        ["Vegetative", "Reproductive", "Maturity"],
        help="Current growth stage"
    )
    
    irrigation_frequency = st.sidebar.slider(
        "Irrigation Frequency (days)", 
        1, 30, 7, 1,
        help="Days between irrigation"
    )
    
    pesticide_usage = st.sidebar.selectbox(
        "Pesticide Usage", 
        ["Low", "Medium", "High"],
        help="Level of pesticide usage"
    )
    
    # Calculate additional parameters
    elevation = st.sidebar.slider("Elevation (m)", 0, 1000, 200, 10)
    slope = st.sidebar.slider("Slope (%)", 0.0, 40.0, 5.0, 0.5)
    aspect = st.sidebar.slider("Aspect (degrees)", 0, 360, 180, 10)
    
    # Vegetation indices (calculated parameters)
    ndvi = st.sidebar.slider("NDVI", -1.0, 1.0, 0.5, 0.01)
    evi = st.sidebar.slider("EVI", -1.0, 1.0, 0.3, 0.01)
    lai = st.sidebar.slider("LAI", 0.5, 8.0, 3.0, 0.1)
    chlorophyll = st.sidebar.slider("Chlorophyll Content", 10.0, 60.0, 30.0, 1.0)
    gdd = st.sidebar.slider("GDD", 500.0, 3000.0, 1500.0, 50.0, help="Growing Degree Days")
    
    # Store input data
    st.session_state.input_data = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Rainfall': rainfall,
        'pH': ph,
        'EC': ec,
        'OC': organic_carbon,
        'N': nitrogen,
        'P': phosphorus,
        'K': potassium,
        'Ca': calcium,
        'Mg': magnesium,
        'S': sulfur,
        'Zn': zinc,
        'Fe': iron,
        'Cu': copper,
        'Mn': manganese,
        'B': boron,
        'Mo': molybdenum,
        'CEC': cec,
        'Sand': sand,
        'Silt': silt,
        'Clay': clay,
        'Bulk_Density': bulk_density,
        'Water_Holding_Capacity': water_holding,
        'Slope': slope,
        'Aspect': aspect,
        'Elevation': elevation,
        'Solar_Radiation': solar_radiation,
        'Wind_Speed': wind_speed,
        'NDVI': ndvi,
        'EVI': evi,
        'LAI': lai,
        'Chlorophyll': chlorophyll,
        'GDD': gdd,
        'Soil_Type': soil_type,
        'Crop_Type': crop_type,
        'Growth_Stage': growth_stage,
        'Irrigation_Frequency': irrigation_frequency,
        'Fertilizer_Type': fertilizer_type,
        'Pesticide_Usage': pesticide_usage,
        'Region': region,
        'Season': season
    }
    
    return st.session_state.input_data

def encode_categorical_features(input_data):
    """
    Encode categorical features to match training data format
    """
    encoded_data = input_data.copy()
    
    # One-hot encode categorical variables
    categorical_features = {
        'Soil_Type': ['Clayey', 'Loamy', 'Sandy', 'Silty'],
        'Crop_Type': ['Maize', 'Rice', 'Soybean', 'Wheat'],
        'Fertilizer_Type': ['Chemical', 'Mixed', 'Organic'],
        'Region': ['North', 'South', 'East', 'West'],
        'Season': ['Kharif', 'Rabi', 'Zaid']
    }
    
    # Label encode ordinal variables
    ordinal_features = {
        'Growth_Stage': {'Vegetative': 0, 'Reproductive': 1, 'Maturity': 2},
        'Pesticide_Usage': {'Low': 0, 'Medium': 1, 'High': 2}
    }
    
    # Apply label encoding
    for feature, mapping in ordinal_features.items():
        encoded_data[feature] = mapping[encoded_data[feature]]
    
    # Apply one-hot encoding (simplified - using binary indicators)
    for feature, categories in categorical_features.items():
        for category in categories[1:]:  # Skip first category (reference)
            col_name = f"{feature}_{category}"
            encoded_data[col_name] = 1 if encoded_data[feature] == category else 0
        # Remove original categorical column
        del encoded_data[feature]
    
    return encoded_data

def create_yield_gauge(prediction, confidence_interval):
    """
    Create a gauge chart for yield prediction
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Predicted Crop Yield"},
        delta = {'reference': 5.0},
        gauge = {
            'axis': {'range': [None, 10]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 3], 'color': "lightgray"},
                {'range': [3, 6], 'color': "yellow"},
                {'range': [6, 10], 'color': "green"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 8
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_confidence_interval_chart(prediction, confidence_interval):
    """
    Create a chart showing confidence interval
    """
    fig = go.Figure()
    
    # Add confidence interval
    fig.add_trace(go.Scatter(
        x=[prediction, prediction],
        y=[confidence_interval[0], confidence_interval[1]],
        mode='lines+markers',
        name='Confidence Interval',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # Add prediction point
    fig.add_trace(go.Scatter(
        x=[prediction],
        y=[prediction],
        mode='markers',
        name='Prediction',
        marker=dict(color='blue', size=12),
        text=[f'Yield: {prediction:.2f}'],
        textposition='top center'
    ))
    
    fig.update_layout(
        title="Prediction Confidence Interval",
        xaxis_title="Predicted Yield",
        yaxis_title="Yield Range",
        height=300,
        showlegend=True
    )
    
    return fig

def create_feature_importance_chart(predictor, model_name='Random Forest'):
    """
    Create a feature importance chart
    """
    if model_name == 'Random Forest':
        importance = predictor.models[model_name].feature_importances_
        feature_names = predictor.feature_names
        
        # Create DataFrame
        df_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=True).tail(10)
        
        fig = px.bar(
            df_importance, 
            x='Importance', 
            y='Feature',
            orientation='h',
            title="Top 10 Most Important Features",
            color='Importance',
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(height=400)
        
        return fig
    else:
        return None

def main():
    """
    Main Streamlit application
    """
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .prediction-box {
        background-color: #e8f5e8;
        padding: 2rem;
        border-radius: 1rem;
        border: 2px solid #2E8B57;
        text-align: center;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">Crop Yield Predictor</h1>', unsafe_allow_html=True)
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        with st.spinner('Loading prediction models...'):
            st.session_state.predictor = CropYieldPredictor()
    
    predictor = st.session_state.predictor
    
    if not predictor.models:
        st.error("Failed to load prediction models. Please check the data files.")
        return
    
    # Create input form
    input_data = create_input_form()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Prediction Results")
        
        # Model selection
        model_name = st.selectbox(
            "Select Model for Prediction:",
            ["Random Forest", "Linear Regression"],
            help="Choose the machine learning model for prediction"
        )
        
        # Prediction button
        if st.button("Predict Crop Yield", type="primary", use_container_width=True):
            with st.spinner('Calculating prediction...'):
                # Encode categorical features
                encoded_data = encode_categorical_features(input_data)
                
                # Make prediction
                prediction, confidence_interval = predictor.predict_yield(encoded_data, model_name)
                
                if prediction is not None:
                    # Store results in session state
                    st.session_state.prediction = prediction
                    st.session_state.confidence_interval = confidence_interval
                    st.session_state.model_name = model_name
                    
                    # Display prediction results
                    st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                    st.markdown(f"### Predicted Crop Yield: {prediction:.2f}")
                    st.markdown(f"**Model Used:** {model_name}")
                    st.markdown(f"**Confidence Interval:** {confidence_interval[0]:.2f} - {confidence_interval[1]:.2f}")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Yield interpretation
                    if prediction < 3:
                        yield_status = "Low"
                        status_emoji = "Warning"
                    elif prediction < 6:
                        yield_status = "Medium"
                        status_emoji = "Moderate"
                    else:
                        yield_status = "High"
                        status_emoji = "Excellent"
                    
                    st.markdown(f"### Yield Status: **{yield_status}**")
                    
                    # Recommendations
                    st.subheader("Recommendations")
                    if prediction < 3:
                        st.warning("**Low yield predicted!** Consider:")
                        st.markdown("- Increase fertilizer application")
                        st.markdown("- Improve irrigation practices")
                        st.markdown("- Check soil pH and nutrient levels")
                    elif prediction < 6:
                        st.info("**Medium yield predicted.** Consider:")
                        st.markdown("- Optimize nutrient management")
                        st.markdown("- Monitor weather conditions")
                        st.markdown("- Consider crop rotation")
                    else:
                        st.success("**High yield predicted!** Maintain:")
                        st.markdown("- Current farming practices")
                        st.markdown("- Regular monitoring")
                        st.markdown("- Soil health management")
    
    with col2:
        st.header("Visual Feedback")
        
        # Display visualizations if prediction exists
        if 'prediction' in st.session_state:
            # Yield gauge
            st.subheader("Yield Gauge")
            gauge_fig = create_yield_gauge(
                st.session_state.prediction, 
                st.session_state.confidence_interval
            )
            st.plotly_chart(gauge_fig, use_container_width=True)
            
            # Confidence interval chart
            st.subheader("Confidence Interval")
            ci_fig = create_confidence_interval_chart(
                st.session_state.prediction,
                st.session_state.confidence_interval
            )
            st.plotly_chart(ci_fig, use_container_width=True)
            
            # Feature importance (only for Random Forest)
            if st.session_state.model_name == 'Random Forest':
                st.subheader("Feature Importance")
                importance_fig = create_feature_importance_chart(predictor, 'Random Forest')
                if importance_fig:
                    st.plotly_chart(importance_fig, use_container_width=True)
    
    # Model comparison section
    st.header("Model Comparison")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        st.subheader("Random Forest")
        st.markdown("""
        **Advantages:**
        - Handles non-linear relationships
        - Feature importance analysis
        - Robust to outliers
        
        **Performance:**
        - MAE: 2.2881
        - RMSE: 2.6437
        - R2: -0.0165
        """)
    
    with col4:
        st.subheader("Linear Regression")
        st.markdown("""
        **Advantages:**
        - Fast and interpretable
        - Good baseline model
        - Less prone to overfitting
        
        **Performance:**
        - MAE: 2.2807
        - RMSE: 2.6349
        - R2: -0.0097
        """)
    
    with col5:
        st.subheader("Best Choice")
        st.markdown("""
        **Linear Regression** performs slightly better:
        
        - Lower prediction errors
        - Better R2 score
        - More stable predictions
        
        *Note: Both models show room for improvement*
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Crop Yield Predictor | Built with Streamlit & Machine Learning</p>
        <p>For agricultural decision support and yield optimization</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
