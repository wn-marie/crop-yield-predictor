"""
Model Performance Visualization
==============================

This script creates visualizations to better understand the model performance
for crop yield prediction.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the preprocessed data
df = pd.read_csv('preprocessed_agricultural_data.csv')

# Prepare data (same as in ML pipeline)
y = df['Yield']
X = df.drop(columns=['Yield', 'Planting_Date', 'Harvest_Date', 'Year', 'Region_Code', 'Country_Name'])

# Split data (same random state as ML pipeline)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models (same as ML pipeline)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
rf_pred = rf_model.predict(X_test)
lr_pred = lr_model.predict(X_test)

# Create visualizations
plt.style.use('default')
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Actual vs Predicted - Random Forest
axes[0, 0].scatter(y_test, rf_pred, alpha=0.6, color='blue')
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Yield')
axes[0, 0].set_ylabel('Predicted Yield')
axes[0, 0].set_title('Random Forest: Actual vs Predicted')
axes[0, 0].grid(True, alpha=0.3)

# Add R² score
rf_r2 = r2_score(y_test, rf_pred)
axes[0, 0].text(0.05, 0.95, f'R² = {rf_r2:.4f}', transform=axes[0, 0].transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2. Actual vs Predicted - Linear Regression
axes[0, 1].scatter(y_test, lr_pred, alpha=0.6, color='green')
axes[0, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 1].set_xlabel('Actual Yield')
axes[0, 1].set_ylabel('Predicted Yield')
axes[0, 1].set_title('Linear Regression: Actual vs Predicted')
axes[0, 1].grid(True, alpha=0.3)

# Add R² score
lr_r2 = r2_score(y_test, lr_pred)
axes[0, 1].text(0.05, 0.95, f'R² = {lr_r2:.4f}', transform=axes[0, 1].transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 3. Residuals - Random Forest
rf_residuals = y_test - rf_pred
axes[1, 0].scatter(rf_pred, rf_residuals, alpha=0.6, color='blue')
axes[1, 0].axhline(y=0, color='r', linestyle='--')
axes[1, 0].set_xlabel('Predicted Yield')
axes[1, 0].set_ylabel('Residuals')
axes[1, 0].set_title('Random Forest: Residuals vs Predicted')
axes[1, 0].grid(True, alpha=0.3)

# 4. Residuals - Linear Regression
lr_residuals = y_test - lr_pred
axes[1, 1].scatter(lr_pred, lr_residuals, alpha=0.6, color='green')
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_xlabel('Predicted Yield')
axes[1, 1].set_ylabel('Residuals')
axes[1, 1].set_title('Linear Regression: Residuals vs Predicted')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('model_performance_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# Feature importance visualization
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(15)
plt.barh(range(len(top_features)), top_features['importance'])
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Feature Importance')
plt.title('Top 15 Most Important Features (Random Forest)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualizations saved as:")
print("- model_performance_visualization.png")
print("- feature_importance_visualization.png")
