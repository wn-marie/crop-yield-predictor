"""
Streamlit App Demo Script
========================

This script demonstrates the key features of the Crop Yield Predictor app
by showing example predictions with different parameter combinations.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def demo_predictions():
    """
    Demonstrate the app's prediction capabilities with example scenarios
    """
    print("Crop Yield Predictor - Demo Scenarios")
    print("=" * 50)
    
    # Load data and train models (simplified version)
    try:
        df = pd.read_csv('preprocessed_agricultural_data.csv')
        
        # Prepare data
        y = df['Yield']
        X = df.drop(columns=['Yield', 'Planting_Date', 'Harvest_Date', 'Year', 'Region_Code', 'Country_Name'])
        
        # Train models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        lr_model = LinearRegression()
        
        rf_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        
        print("[OK] Models trained successfully!")
        
        # Demo scenarios
        scenarios = [
            {
                "name": "Optimal Conditions",
                "description": "Perfect weather, rich soil, proper nutrients",
                "params": {
                    "Temperature": 0.5,  # Slightly above average
                    "Humidity": 0.5,
                    "Rainfall": 0.5,
                    "pH": 0.5,
                    "N": 0.7,  # High nitrogen
                    "P": 0.7,  # High phosphorus
                    "K": 0.7,  # High potassium
                    "Cu": 0.8,  # High copper (important feature)
                    "Mo": 0.8,  # High molybdenum (important feature)
                }
            },
            {
                "name": "Poor Conditions",
                "description": "Harsh weather, poor soil, nutrient deficiency",
                "params": {
                    "Temperature": -0.8,  # Below average
                    "Humidity": -0.5,
                    "Rainfall": -0.7,  # Drought conditions
                    "pH": -0.5,
                    "N": -0.7,  # Low nitrogen
                    "P": -0.7,  # Low phosphorus
                    "K": -0.7,  # Low potassium
                    "Cu": -0.8,  # Low copper
                    "Mo": -0.8,  # Low molybdenum
                }
            },
            {
                "name": "Average Conditions",
                "description": "Typical farming conditions with moderate parameters",
                "params": {
                    "Temperature": 0.0,
                    "Humidity": 0.0,
                    "Rainfall": 0.0,
                    "pH": 0.0,
                    "N": 0.0,
                    "P": 0.0,
                    "K": 0.0,
                    "Cu": 0.0,
                    "Mo": 0.0,
                }
            }
        ]
        
        print("\nDemo Predictions:")
        print("-" * 30)
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{i}. {scenario['name']}")
            print(f"   {scenario['description']}")
            
            # Create input data (fill missing features with mean values)
            input_data = {}
            for feature in X.columns:
                if feature in scenario['params']:
                    # Scale the parameter to the feature's range
                    feature_mean = X[feature].mean()
                    feature_std = X[feature].std()
                    input_data[feature] = feature_mean + scenario['params'][feature] * feature_std
                else:
                    input_data[feature] = X[feature].mean()
            
            # Make predictions
            input_df = pd.DataFrame([input_data])
            
            rf_pred = rf_model.predict(input_df)[0]
            lr_pred = lr_model.predict(input_df)[0]
            
            print(f"   Random Forest Prediction: {rf_pred:.2f}")
            print(f"   Linear Regression Prediction: {lr_pred:.2f}")
            
            # Determine yield status
            avg_pred = (rf_pred + lr_pred) / 2
            if avg_pred < 3:
                status = "Low Yield (Warning)"
            elif avg_pred < 6:
                status = "Medium Yield (Moderate)"
            else:
                status = "High Yield (Excellent)"
            
            print(f"   Average Prediction: {avg_pred:.2f} - {status}")
        
        print("\nKey Insights:")
        print("- Optimal conditions show higher yield predictions")
        print("- Poor conditions result in lower yield predictions")
        print("- Both models show similar trends but different magnitudes")
        print("- Feature importance: Cu (Copper) and Mo (Molybdenum) are crucial")
        
        print("\nTo try the interactive app:")
        print("   Run: streamlit run streamlit_crop_yield_app.py")
        print("   Then open: http://localhost:8501")
        
    except FileNotFoundError:
        print("[ERROR] preprocessed_agricultural_data.csv not found")
        print("   Please run the preprocessing pipeline first")
    except Exception as e:
        print(f"[ERROR] {str(e)}")

def show_app_features():
    """
    Display the key features of the Streamlit app
    """
    print("\nStreamlit App Features:")
    print("=" * 30)
    
    features = [
        "Interactive Parameter Input",
        "   - Weather conditions (temperature, humidity, rainfall)",
        "   - Soil properties (pH, nutrients, composition)",
        "   - Farming practices (crop type, fertilizer, irrigation)",
        "",
        "Machine Learning Models",
        "   - Random Forest Regressor (100 trees)",
        "   - Linear Regression",
        "   - Real-time model training and prediction",
        "",
        "Visual Feedback",
        "   - Interactive yield gauge with color coding",
        "   - Confidence interval visualization",
        "   - Feature importance charts",
        "   - Performance metrics comparison",
        "",
        "Smart Recommendations",
        "   - Yield status interpretation (Low/Medium/High)",
        "   - Actionable improvement suggestions",
        "   - Context-aware farming advice",
        "",
        "Professional Interface",
        "   - Clean, modern design",
        "   - Responsive layout for all devices",
        "   - Agricultural-themed styling",
        "   - Real-time updates and feedback"
    ]
    
    for feature in features:
        print(feature)

if __name__ == "__main__":
    demo_predictions()
    show_app_features()
