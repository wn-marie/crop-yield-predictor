"""
Test Prediction Responsiveness
=============================

This script tests whether the prediction function responds correctly to input changes.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def test_prediction_responsiveness():
    """
    Test if predictions change when input values change
    """
    print("Testing Prediction Responsiveness")
    print("=" * 40)
    
    try:
        # Load data
        df = pd.read_csv('preprocessed_agricultural_data.csv')
        
        # Prepare features and target
        y = df['Yield']
        X = df.drop(columns=['Yield', 'Planting_Date', 'Harvest_Date', 'Year', 'Region_Code', 'Country_Name'])
        
        # Train models
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        lr_model = LinearRegression()
        
        rf_model.fit(X_train, y_train)
        lr_model.fit(X_train, y_train)
        
        print("[OK] Models trained successfully")
        
        # Test with different input values
        base_input = X_train.iloc[0].copy()  # Use first training sample as base
        
        print("\nTesting Random Forest Model:")
        print("-" * 30)
        
        # Test 1: Base prediction
        pred1 = rf_model.predict([base_input])[0]
        print(f"Base prediction: {pred1:.4f}")
        
        # Test 2: Change temperature
        modified_input = base_input.copy()
        modified_input['Temperature'] = modified_input['Temperature'] + 2.0  # Increase temperature
        pred2 = rf_model.predict([modified_input])[0]
        print(f"Temperature +2.0: {pred2:.4f} (difference: {pred2-pred1:.4f})")
        
        # Test 3: Change rainfall
        modified_input = base_input.copy()
        modified_input['Rainfall'] = modified_input['Rainfall'] + 50.0  # Increase rainfall
        pred3 = rf_model.predict([modified_input])[0]
        print(f"Rainfall +50.0: {pred3:.4f} (difference: {pred3-pred1:.4f})")
        
        # Test 4: Change nitrogen
        modified_input = base_input.copy()
        modified_input['N'] = modified_input['N'] + 20.0  # Increase nitrogen
        pred4 = rf_model.predict([modified_input])[0]
        print(f"Nitrogen +20.0: {pred4:.4f} (difference: {pred4-pred1:.4f})")
        
        # Test 5: Change copper (important feature)
        modified_input = base_input.copy()
        modified_input['Cu'] = modified_input['Cu'] + 1.0  # Increase copper
        pred5 = rf_model.predict([modified_input])[0]
        print(f"Copper +1.0: {pred5:.4f} (difference: {pred5-pred1:.4f})")
        
        print("\nTesting Linear Regression Model:")
        print("-" * 30)
        
        # Test Linear Regression
        lr_pred1 = lr_model.predict([base_input])[0]
        print(f"Base prediction: {lr_pred1:.4f}")
        
        modified_input = base_input.copy()
        modified_input['Temperature'] = modified_input['Temperature'] + 2.0
        lr_pred2 = lr_model.predict([modified_input])[0]
        print(f"Temperature +2.0: {lr_pred2:.4f} (difference: {lr_pred2-lr_pred1:.4f})")
        
        # Analyze results
        print("\nAnalysis:")
        print("-" * 20)
        
        rf_changes = [abs(pred2-pred1), abs(pred3-pred1), abs(pred4-pred1), abs(pred5-pred1)]
        lr_changes = [abs(lr_pred2-lr_pred1)]
        
        print(f"Random Forest changes: {rf_changes}")
        print(f"Linear Regression changes: {lr_changes}")
        
        if any(change > 0.001 for change in rf_changes):
            print("[OK] Random Forest predictions ARE responsive to input changes")
        else:
            print("[WARNING] Random Forest predictions show minimal change")
            
        if any(change > 0.001 for change in lr_changes):
            print("[OK] Linear Regression predictions ARE responsive to input changes")
        else:
            print("[WARNING] Linear Regression predictions show minimal change")
        
        # Feature importance check
        print("\nFeature Importance (Random Forest):")
        print("-" * 35)
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)
        
        for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
            print(f"{i:2d}. {row['Feature']:<25} {row['Importance']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_prediction_responsiveness()
