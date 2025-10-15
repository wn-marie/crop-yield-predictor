"""
Crop Yield Prediction Streamlit App - Debug Version
==================================================

Debug version to identify and fix prediction accuracy issues.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Crop Yield Predictor - Debug",
    page_icon="🌾",
    layout="wide"
)

class CropYieldPredictor:
    def __init__(self):
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.feature_ranges = {}
        self.load_models()
    
    def load_models(self):
        try:
            # Load preprocessed data
            df = pd.read_csv('preprocessed_agricultural_data.csv')
            
            # Prepare features and target
            y = df['Yield']
            X = df.drop(columns=['Yield', 'Planting_Date', 'Harvest_Date', 'Year', 'Region_Code', 'Country_Name'])
            
            # Store feature information
            self.feature_names = X.columns.tolist()
            self.feature_ranges = {
                col: {'min': X[col].min(), 'max': X[col].max(), 'mean': X[col].mean()}
                for col in X.columns
            }
            
            # Split and train
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            self.models['Random Forest'] = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            self.models['Linear Regression'] = LinearRegression()
            
            self.models['Random Forest'].fit(X_train, y_train)
            self.models['Linear Regression'].fit(X_train, y_train)
            
            return True
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
            return False
    
    def predict_yield(self, input_data, model_name='Random Forest'):
        try:
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = input_data
            
            # Ensure all features are present
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = self.feature_ranges[feature]['mean']
            
            # Reorder columns
            input_df = input_df[self.feature_names]
            
            # Make prediction
            prediction = self.models[model_name].predict(input_df)[0]
            
            # Calculate confidence interval
            if model_name == 'Random Forest':
                tree_predictions = []
                for tree in self.models[model_name].estimators_:
                    tree_pred = tree.predict(input_df)[0]
                    tree_predictions.append(tree_pred)
                
                mean_pred = np.mean(tree_predictions)
                std_pred = np.std(tree_predictions)
                confidence_interval = (mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred)
            else:
                confidence_interval = (prediction - 1.0, prediction + 1.0)
            
            return prediction, confidence_interval, input_df
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None, None, None

def encode_categorical_features(input_data):
    """Encode categorical features properly"""
    encoded_data = input_data.copy()
    
    # Label encode ordinal variables
    ordinal_features = {
        'Growth_Stage': {'Vegetative': 0, 'Reproductive': 1, 'Maturity': 2},
        'Pesticide_Usage': {'Low': 0, 'Medium': 1, 'High': 2}
    }
    
    # Apply label encoding
    for feature, mapping in ordinal_features.items():
        if feature in encoded_data:
            encoded_data[feature] = mapping[encoded_data[feature]]
    
    # One-hot encode categorical variables
    categorical_features = {
        'Soil_Type': ['Clayey', 'Loamy', 'Sandy', 'Silty'],
        'Crop_Type': ['Maize', 'Rice', 'Soybean', 'Wheat'],
        'Fertilizer_Type': ['Chemical', 'Mixed', 'Organic'],
        'Region': ['North', 'South', 'East', 'West'],
        'Season': ['Kharif', 'Rabi', 'Zaid']
    }
    
    # Apply one-hot encoding
    for feature, categories in categorical_features.items():
        if feature in encoded_data:
            for category in categories[1:]:  # Skip first category (reference)
                col_name = f"{feature}_{category}"
                encoded_data[col_name] = 1 if encoded_data[feature] == category else 0
            # Remove original categorical column
            del encoded_data[feature]
    
    return encoded_data

def create_simple_input_form():
    """Simplified input form for debugging"""
    st.sidebar.header("Debug Input Parameters")
    
    # Key parameters that should affect predictions
    temperature = st.sidebar.slider("Temperature (C)", 10.0, 40.0, 25.0, 0.1)
    humidity = st.sidebar.slider("Humidity (%)", 20.0, 90.0, 60.0, 1.0)
    rainfall = st.sidebar.slider("Rainfall (mm)", 50.0, 400.0, 200.0, 5.0)
    ph = st.sidebar.number_input("pH Level", 4.0, 9.0, 6.5, 0.1)
    nitrogen = st.sidebar.slider("Nitrogen (kg/ha)", 20.0, 200.0, 100.0, 5.0)
    phosphorus = st.sidebar.slider("Phosphorus (kg/ha)", 5.0, 150.0, 50.0, 5.0)
    potassium = st.sidebar.slider("Potassium (kg/ha)", 50.0, 300.0, 150.0, 10.0)
    copper = st.sidebar.slider("Copper (mg/kg)", 0.2, 5.0, 1.0, 0.1)
    molybdenum = st.sidebar.slider("Molybdenum (mg/kg)", 0.1, 1.0, 0.3, 0.05)
    
    # Categorical variables
    soil_type = st.sidebar.selectbox("Soil Type", ["Clayey", "Loamy", "Sandy", "Silty"])
    crop_type = st.sidebar.selectbox("Crop Type", ["Maize", "Rice", "Soybean", "Wheat"])
    
    # Calculate other required parameters with reasonable defaults
    input_data = {
        'Temperature': temperature,
        'Humidity': humidity,
        'Rainfall': rainfall,
        'pH': ph,
        'EC': 1.0,  # Default
        'OC': 1.0,  # Default
        'N': nitrogen,
        'P': phosphorus,
        'K': potassium,
        'Ca': 800.0,  # Default
        'Mg': 250.0,  # Default
        'S': 40.0,    # Default
        'Zn': 2.0,    # Default
        'Fe': 20.0,   # Default
        'Cu': copper,
        'Mn': 20.0,   # Default
        'B': 1.0,     # Default
        'Mo': molybdenum,
        'CEC': 25.0,  # Default
        'Sand': 40.0, # Default
        'Silt': 35.0, # Default
        'Clay': 25.0, # Default
        'Bulk_Density': 1.3,  # Default
        'Water_Holding_Capacity': 30.0,  # Default
        'Slope': 5.0,  # Default
        'Aspect': 180.0,  # Default
        'Elevation': 200.0,  # Default
        'Solar_Radiation': 2000.0,  # Default
        'Wind_Speed': 15.0,  # Default
        'NDVI': 0.5,   # Default
        'EVI': 0.3,    # Default
        'LAI': 3.0,    # Default
        'Chlorophyll': 30.0,  # Default
        'GDD': 1500.0,  # Default
        'Soil_Type': soil_type,
        'Crop_Type': crop_type,
        'Growth_Stage': 'Reproductive',
        'Irrigation_Frequency': 7,
        'Fertilizer_Type': 'Mixed',
        'Pesticide_Usage': 'Medium',
        'Region': 'North',
        'Season': 'Kharif'
    }
    
    return input_data

def main():
    st.title("Crop Yield Predictor - Debug Version")
    
    # Initialize predictor
    if 'predictor' not in st.session_state:
        with st.spinner('Loading models...'):
            st.session_state.predictor = CropYieldPredictor()
    
    predictor = st.session_state.predictor
    
    if not predictor.models:
        st.error("Failed to load models")
        return
    
    # Create input form
    input_data = create_simple_input_form()
    
    # Model selection
    model_name = st.selectbox("Select Model:", ["Random Forest", "Linear Regression"])
    
    # Make predictions
    try:
        encoded_data = encode_categorical_features(input_data)
        prediction, confidence_interval, processed_input = predictor.predict_yield(encoded_data, model_name)
        
        if prediction is not None:
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Prediction Results")
                st.metric("Predicted Yield", f"{prediction:.4f}")
                st.metric("Confidence Interval", f"{confidence_interval[0]:.2f} - {confidence_interval[1]:.2f}")
                
                # Yield status
                if prediction < 3:
                    yield_status = "Low"
                    st.error(f"Yield Status: **{yield_status}**")
                    st.warning("Recommendations: Increase fertilizer, improve irrigation, check soil pH")
                elif prediction < 6:
                    yield_status = "Medium"
                    st.warning(f"Yield Status: **{yield_status}**")
                    st.info("Recommendations: Optimize nutrients, monitor weather, consider crop rotation")
                else:
                    yield_status = "High"
                    st.success(f"Yield Status: **{yield_status}**")
                    st.success("Recommendations: Maintain current practices, regular monitoring, soil health")
            
            with col2:
                # Simple gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': f"{model_name} Prediction"},
                    gauge = {
                        'axis': {'range': [None, 10]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 3], 'color': "red"},
                            {'range': [3, 6], 'color': "yellow"},
                            {'range': [6, 10], 'color': "green"}
                        ]
                    }
                ))
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Debug information
        with st.expander("Debug Information"):
            st.subheader("Input Data")
            st.json(input_data)
            
            st.subheader("Encoded Data")
            st.json(encoded_data)
            
            if processed_input is not None:
                st.subheader("Processed Input (first 10 features)")
                st.write(processed_input.iloc[0, :10].to_dict())
            
            st.subheader("Feature Names")
            st.write(predictor.feature_names[:10])  # Show first 10
            
            # Test with different values
            st.subheader("Sensitivity Test")
            test_values = [0.5, 1.0, 2.0, 5.0, 10.0]
            test_predictions = []
            
            for val in test_values:
                test_data = input_data.copy()
                test_data['Temperature'] = val  # Extreme values
                test_encoded = encode_categorical_features(test_data)
                test_pred, _, _ = predictor.predict_yield(test_encoded, model_name)
                test_predictions.append(test_pred)
            
            sensitivity_df = pd.DataFrame({
                'Temperature': test_values,
                'Prediction': test_predictions
            })
            st.dataframe(sensitivity_df)
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    main()
