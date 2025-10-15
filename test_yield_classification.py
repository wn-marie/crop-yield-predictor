"""
Test Yield Classification System
===============================

This script tests whether the yield status classification works correctly.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def test_yield_classification():
    """
    Test yield status classification with different input scenarios
    """
    print("Testing Yield Classification System")
    print("=" * 40)
    
    try:
        # Load data
        df = pd.read_csv('preprocessed_agricultural_data.csv')
        
        # Prepare features and target
        y = df['Yield']
        X = df.drop(columns=['Yield', 'Planting_Date', 'Harvest_Date', 'Year', 'Region_Code', 'Country_Name'])
        
        # Calculate yield statistics
        yield_stats = {
            'min': y.min(),
            'max': y.max(),
            'mean': y.mean(),
            'median': y.median(),
            'std': y.std(),
            'q25': y.quantile(0.25),
            'q75': y.quantile(0.75)
        }
        
        print("Yield Statistics:")
        print(f"Min: {yield_stats['min']:.2f}")
        print(f"25th Percentile: {yield_stats['q25']:.2f}")
        print(f"Median: {yield_stats['median']:.2f}")
        print(f"75th Percentile: {yield_stats['q75']:.2f}")
        print(f"Max: {yield_stats['max']:.2f}")
        
        print(f"\nClassification Thresholds:")
        print(f"Low: < {yield_stats['q25']:.2f}")
        print(f"Medium: {yield_stats['q25']:.2f} - {yield_stats['q75']:.2f}")
        print(f"High: > {yield_stats['q75']:.2f}")
        
        # Train models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
        lr_model = LinearRegression()
        
        rf_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        
        print(f"\n[OK] Models trained successfully")
        
        # Test different scenarios
        scenarios = [
            {"name": "Low Yield Scenario", "modifications": {"Temperature": 5.0, "Rainfall": 20.0, "N": 10.0}},
            {"name": "Medium Yield Scenario", "modifications": {"Temperature": 25.0, "Rainfall": 200.0, "N": 100.0}},
            {"name": "High Yield Scenario", "modifications": {"Temperature": 35.0, "Rainfall": 350.0, "N": 180.0}},
            {"name": "Extreme Low", "modifications": {"Temperature": 5.0, "Rainfall": 10.0, "N": 5.0, "pH": 4.0}},
            {"name": "Extreme High", "modifications": {"Temperature": 40.0, "Rainfall": 400.0, "N": 200.0, "pH": 8.0}}
        ]
        
        print(f"\nTesting Different Scenarios:")
        print("-" * 50)
        
        base_input = X_train.iloc[0].copy()
        
        for scenario in scenarios:
            print(f"\n{scenario['name']}:")
            
            # Modify input
            test_input = base_input.copy()
            for param, value in scenario['modifications'].items():
                if param in test_input.index:
                    test_input[param] = value
            
            # Make predictions
            rf_pred = rf_model.predict([test_input])[0]
            lr_pred = lr_model.predict([test_input])[0]
            
            # Apply scaling for Random Forest (same as in the app)
            if rf_pred < yield_stats['q25']:
                rf_pred_scaled = yield_stats['min'] + (rf_pred - yield_stats['q25']) * 0.5
            elif rf_pred > yield_stats['q75']:
                rf_pred_scaled = yield_stats['max'] - (yield_stats['q75'] - rf_pred) * 0.5
            else:
                rf_pred_scaled = rf_pred
            
            rf_pred_scaled = max(yield_stats['min'], min(yield_stats['max'], rf_pred_scaled))
            
            # Classify predictions
            def get_yield_status(prediction):
                if prediction < yield_stats['q25']:
                    return "Low"
                elif prediction < yield_stats['q75']:
                    return "Medium"
                else:
                    return "High"
            
            rf_status = get_yield_status(rf_pred_scaled)
            lr_status = get_yield_status(lr_pred)
            
            print(f"  Random Forest: {rf_pred_scaled:.2f} -> {rf_status}")
            print(f"  Linear Regression: {lr_pred:.2f} -> {lr_status}")
            
            # Show modifications
            print(f"  Modifications: {scenario['modifications']}")
        
        # Test prediction range
        print(f"\nPrediction Range Analysis:")
        print("-" * 30)
        
        test_predictions = []
        for i in range(100):
            test_input = X_train.iloc[i].copy()
            rf_pred = rf_model.predict([test_input])[0]
            test_predictions.append(rf_pred)
        
        test_predictions = np.array(test_predictions)
        print(f"RF Raw Predictions - Min: {test_predictions.min():.2f}, Max: {test_predictions.max():.2f}")
        
        # Apply scaling
        scaled_predictions = []
        for pred in test_predictions:
            if pred < yield_stats['q25']:
                scaled_pred = yield_stats['min'] + (pred - yield_stats['q25']) * 0.5
            elif pred > yield_stats['q75']:
                scaled_pred = yield_stats['max'] - (yield_stats['q75'] - pred) * 0.5
            else:
                scaled_pred = pred
            
            scaled_pred = max(yield_stats['min'], min(yield_stats['max'], scaled_pred))
            scaled_predictions.append(scaled_pred)
        
        scaled_predictions = np.array(scaled_predictions)
        print(f"RF Scaled Predictions - Min: {scaled_predictions.min():.2f}, Max: {scaled_predictions.max():.2f}")
        
        # Count classifications
        low_count = (scaled_predictions < yield_stats['q25']).sum()
        medium_count = ((scaled_predictions >= yield_stats['q25']) & (scaled_predictions < yield_stats['q75'])).sum()
        high_count = (scaled_predictions >= yield_stats['q75']).sum()
        
        print(f"\nClassification Distribution (100 samples):")
        print(f"Low: {low_count} ({low_count}%)")
        print(f"Medium: {medium_count} ({medium_count}%)")
        print(f"High: {high_count} ({high_count}%)")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_yield_classification()
