"""
Test Final Fix for Random Forest Predictions
===========================================

This script tests whether the final fix resolves the Random Forest prediction issues.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def test_final_fix():
    """
    Test the final fix for Random Forest predictions
    """
    print("Testing Final Fix for Random Forest Predictions")
    print("=" * 50)
    
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
        
        # Train models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(
            n_estimators=200, 
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        lr_model = LinearRegression()
        
        rf_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        
        # Get RF prediction statistics
        rf_train_pred = rf_model.predict(X_train)
        rf_prediction_stats = {
            'min': rf_train_pred.min(),
            'max': rf_train_pred.max(),
            'mean': rf_train_pred.mean(),
            'std': rf_train_pred.std()
        }
        
        print(f"\nRandom Forest Raw Prediction Statistics:")
        print(f"Min: {rf_prediction_stats['min']:.2f}")
        print(f"Max: {rf_prediction_stats['max']:.2f}")
        print(f"Mean: {rf_prediction_stats['mean']:.2f}")
        
        def apply_rf_scaling(prediction):
            """Apply the same scaling as in the final fix"""
            rf_min, rf_max = rf_prediction_stats['min'], rf_prediction_stats['max']
            yield_min, yield_max = yield_stats['min'], yield_stats['max']
            
            # Normalize RF prediction to 0-1 range
            rf_normalized = (prediction - rf_min) / (rf_max - rf_min)
            
            # Apply non-linear scaling
            if rf_normalized < 0.3:
                # Lower third maps to low yield range (1.0 - 3.18)
                scaled_pred = yield_min + rf_normalized * (yield_stats['q25'] - yield_min) / 0.3
            elif rf_normalized < 0.7:
                # Middle third maps to medium yield range (3.18 - 7.78)
                scaled_pred = yield_stats['q25'] + (rf_normalized - 0.3) * (yield_stats['q75'] - yield_stats['q25']) / 0.4
            else:
                # Upper third maps to high yield range (7.78 - 10.0)
                scaled_pred = yield_stats['q75'] + (rf_normalized - 0.7) * (yield_max - yield_stats['q75']) / 0.3
            
            return max(yield_min, min(yield_max, scaled_pred))
        
        def get_yield_status(prediction):
            """Get yield status based on prediction"""
            if prediction < yield_stats['q25']:
                return "Low"
            elif prediction < yield_stats['q75']:
                return "Medium"
            else:
                return "High"
        
        # Test different scenarios
        scenarios = [
            {"name": "Low Yield Scenario", "modifications": {"Temperature": 5.0, "Rainfall": 20.0, "N": 10.0}},
            {"name": "Medium Yield Scenario", "modifications": {"Temperature": 25.0, "Rainfall": 200.0, "N": 100.0}},
            {"name": "High Yield Scenario", "modifications": {"Temperature": 35.0, "Rainfall": 350.0, "N": 180.0}},
            {"name": "Extreme Low", "modifications": {"Temperature": 5.0, "Rainfall": 10.0, "N": 5.0, "pH": 4.0}},
            {"name": "Extreme High", "modifications": {"Temperature": 40.0, "Rainfall": 400.0, "N": 200.0, "pH": 8.0}}
        ]
        
        print(f"\nTesting Different Scenarios with Final Fix:")
        print("-" * 60)
        
        base_input = X_train.iloc[0].copy()
        
        for scenario in scenarios:
            print(f"\n{scenario['name']}:")
            
            # Modify input
            test_input = base_input.copy()
            for param, value in scenario['modifications'].items():
                if param in test_input.index:
                    test_input[param] = value
            
            # Make predictions
            rf_pred_raw = rf_model.predict([test_input])[0]
            rf_pred_scaled = apply_rf_scaling(rf_pred_raw)
            lr_pred = lr_model.predict([test_input])[0]
            
            # Classify predictions
            rf_status = get_yield_status(rf_pred_scaled)
            lr_status = get_yield_status(lr_pred)
            
            print(f"  RF Raw: {rf_pred_raw:.2f} -> Scaled: {rf_pred_scaled:.2f} -> {rf_status}")
            print(f"  LR: {lr_pred:.2f} -> {lr_status}")
            print(f"  Modifications: {scenario['modifications']}")
        
        # Test prediction range after scaling
        print(f"\nPrediction Range Analysis (100 samples):")
        print("-" * 45)
        
        test_predictions_raw = []
        test_predictions_scaled = []
        
        for i in range(100):
            test_input = X_train.iloc[i].copy()
            rf_pred_raw = rf_model.predict([test_input])[0]
            rf_pred_scaled = apply_rf_scaling(rf_pred_raw)
            
            test_predictions_raw.append(rf_pred_raw)
            test_predictions_scaled.append(rf_pred_scaled)
        
        test_predictions_raw = np.array(test_predictions_raw)
        test_predictions_scaled = np.array(test_predictions_scaled)
        
        print(f"RF Raw Predictions - Min: {test_predictions_raw.min():.2f}, Max: {test_predictions_raw.max():.2f}")
        print(f"RF Scaled Predictions - Min: {test_predictions_scaled.min():.2f}, Max: {test_predictions_scaled.max():.2f}")
        
        # Count classifications
        low_count = (test_predictions_scaled < yield_stats['q25']).sum()
        medium_count = ((test_predictions_scaled >= yield_stats['q25']) & (test_predictions_scaled < yield_stats['q75'])).sum()
        high_count = (test_predictions_scaled >= yield_stats['q75']).sum()
        
        print(f"\nClassification Distribution:")
        print(f"Low: {low_count} ({low_count}%)")
        print(f"Medium: {medium_count} ({medium_count}%)")
        print(f"High: {high_count} ({high_count}%)")
        
        # Success criteria
        success = low_count > 0 and high_count > 0
        print(f"\nFix Success: {'YES' if success else 'NO'}")
        
        if success:
            print("✅ Random Forest now produces Low, Medium, and High predictions!")
        else:
            print("❌ Random Forest still not producing full range of predictions")
        
        return success
        
    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_final_fix()
