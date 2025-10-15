"""
Enhanced Crop Yield Prediction Visualizations
============================================

This script creates comprehensive visualizations for crop yield prediction models:
1. Actual vs Predicted scatter plots for both models
2. Feature importance chart for Random Forest model
3. Enhanced styling and annotations using Matplotlib and Seaborn

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

class CropYieldVisualizer:
    """
    A class for creating comprehensive visualizations of crop yield prediction models
    """
    
    def __init__(self):
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.rf_model = None
        self.lr_model = None
        self.rf_predictions = None
        self.lr_predictions = None
        self.feature_importance = None
        
    def load_and_prepare_data(self, file_path='preprocessed_agricultural_data.csv'):
        """
        Load and prepare data for visualization
        """
        print("Loading and preparing data for visualization...")
        
        # Load data
        self.df = pd.read_csv(file_path)
        
        # Prepare features and target
        y = self.df['Yield']
        X = self.df.drop(columns=['Yield', 'Planting_Date', 'Harvest_Date', 'Year', 'Region_Code', 'Country_Name'])
        
        # Split data (same as ML pipeline)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"[OK] Data loaded: {self.df.shape}")
        print(f"[OK] Features: {X.shape[1]}, Training samples: {self.X_train.shape[0]}, Test samples: {self.X_test.shape[0]}")
        
        return X, y
    
    def train_models(self):
        """
        Train both models for visualization
        """
        print("Training models for visualization...")
        
        # Train Random Forest
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        self.rf_model.fit(self.X_train, self.y_train)
        
        # Train Linear Regression
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train, self.y_train)
        
        # Make predictions
        self.rf_predictions = self.rf_model.predict(self.X_test)
        self.lr_predictions = self.lr_model.predict(self.X_test)
        
        print("[OK] Models trained and predictions made")
        
        return self.rf_predictions, self.lr_predictions
    
    def create_actual_vs_predicted_plots(self):
        """
        Create enhanced actual vs predicted scatter plots for both models
        """
        print("Creating actual vs predicted scatter plots...")
        
        # Calculate metrics
        rf_mae = mean_absolute_error(self.y_test, self.rf_predictions)
        rf_rmse = np.sqrt(mean_squared_error(self.y_test, self.rf_predictions))
        rf_r2 = r2_score(self.y_test, self.rf_predictions)
        
        lr_mae = mean_absolute_error(self.y_test, self.lr_predictions)
        lr_rmse = np.sqrt(mean_squared_error(self.y_test, self.lr_predictions))
        lr_r2 = r2_score(self.y_test, self.lr_predictions)
        
        # Create figure with subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Random Forest plot
        ax1 = axes[0]
        scatter1 = ax1.scatter(self.y_test, self.rf_predictions, alpha=0.6, c=self.rf_predictions, 
                              cmap='Blues', s=50, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), self.rf_predictions.min())
        max_val = max(self.y_test.max(), self.rf_predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, alpha=0.8, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Yield', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Predicted Yield', fontsize=12, fontweight='bold')
        ax1.set_title('Random Forest: Actual vs Predicted Yield', fontsize=14, fontweight='bold', pad=20)
        ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.legend(fontsize=10)
        
        # Add metrics text box
        metrics_text = f'MAE: {rf_mae:.4f}\nRMSE: {rf_rmse:.4f}\nR²: {rf_r2:.4f}'
        ax1.text(0.05, 0.95, metrics_text, transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
                verticalalignment='top')
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1)
        cbar1.set_label('Predicted Yield', fontsize=10)
        
        # Linear Regression plot
        ax2 = axes[1]
        scatter2 = ax2.scatter(self.y_test, self.lr_predictions, alpha=0.6, c=self.lr_predictions,
                              cmap='Greens', s=50, edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=3, alpha=0.8, label='Perfect Prediction')
        
        ax2.set_xlabel('Actual Yield', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Predicted Yield', fontsize=12, fontweight='bold')
        ax2.set_title('Linear Regression: Actual vs Predicted Yield', fontsize=14, fontweight='bold', pad=20)
        ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax2.legend(fontsize=10)
        
        # Add metrics text box
        metrics_text = f'MAE: {lr_mae:.4f}\nRMSE: {lr_rmse:.4f}\nR²: {lr_r2:.4f}'
        ax2.text(0.05, 0.95, metrics_text, transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                verticalalignment='top')
        
        # Add colorbar
        cbar2 = plt.colorbar(scatter2, ax=ax2)
        cbar2.set_label('Predicted Yield', fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig('enhanced_actual_vs_predicted.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print("[OK] Actual vs predicted plots created and saved")
        
        return fig
    
    def create_feature_importance_chart(self, top_n=20):
        """
        Create enhanced feature importance chart for Random Forest model
        """
        print(f"Creating feature importance chart (top {top_n} features)...")
        
        # Get feature importance
        feature_names = self.X_train.columns
        importances = self.rf_model.feature_importances_
        
        # Create DataFrame
        self.feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=True)
        
        # Get top N features
        top_features = self.feature_importance.tail(top_n)
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create color map based on importance values
        colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
        
        # Create horizontal bar plot
        bars = ax.barh(range(len(top_features)), top_features['Importance'], 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Customize y-axis
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'], fontsize=11)
        ax.invert_yaxis()
        
        # Customize x-axis
        ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Most Important Features\n(Random Forest Model)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, alpha=0.3, axis='x', linestyle='-', linewidth=0.5)
        
        # Add value labels on bars
        for i, (bar, value) in enumerate(zip(bars, top_features['Importance'])):
            ax.text(value + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{value:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Add summary statistics
        total_importance = top_features['Importance'].sum()
        ax.text(0.02, 0.98, f'Total Importance: {total_importance:.4f}', 
               transform=ax.transAxes, fontsize=10,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8),
               verticalalignment='top')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        plt.savefig('enhanced_feature_importance.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print("[OK] Feature importance chart created and saved")
        
        return fig
    
    def create_comprehensive_comparison_plot(self):
        """
        Create a comprehensive comparison plot showing both models side by side
        """
        print("Creating comprehensive comparison plot...")
        
        # Calculate residuals
        rf_residuals = self.y_test - self.rf_predictions
        lr_residuals = self.y_test - self.lr_predictions
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Actual vs Predicted - Random Forest
        ax1 = axes[0, 0]
        ax1.scatter(self.y_test, self.rf_predictions, alpha=0.6, color='steelblue', s=50, edgecolors='black', linewidth=0.5)
        min_val = min(self.y_test.min(), self.rf_predictions.min())
        max_val = max(self.y_test.max(), self.rf_predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8)
        ax1.set_xlabel('Actual Yield', fontweight='bold')
        ax1.set_ylabel('Predicted Yield', fontweight='bold')
        ax1.set_title('Random Forest: Actual vs Predicted', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add R² score
        rf_r2 = r2_score(self.y_test, self.rf_predictions)
        ax1.text(0.05, 0.95, f'R² = {rf_r2:.4f}', transform=ax1.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 2. Actual vs Predicted - Linear Regression
        ax2 = axes[0, 1]
        ax2.scatter(self.y_test, self.lr_predictions, alpha=0.6, color='darkgreen', s=50, edgecolors='black', linewidth=0.5)
        ax2.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, alpha=0.8)
        ax2.set_xlabel('Actual Yield', fontweight='bold')
        ax2.set_ylabel('Predicted Yield', fontweight='bold')
        ax2.set_title('Linear Regression: Actual vs Predicted', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add R² score
        lr_r2 = r2_score(self.y_test, self.lr_predictions)
        ax2.text(0.05, 0.95, f'R² = {lr_r2:.4f}', transform=ax2.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # 3. Residuals - Random Forest
        ax3 = axes[1, 0]
        ax3.scatter(self.rf_predictions, rf_residuals, alpha=0.6, color='steelblue', s=50, edgecolors='black', linewidth=0.5)
        ax3.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax3.set_xlabel('Predicted Yield', fontweight='bold')
        ax3.set_ylabel('Residuals', fontweight='bold')
        ax3.set_title('Random Forest: Residuals vs Predicted', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Residuals - Linear Regression
        ax4 = axes[1, 1]
        ax4.scatter(self.lr_predictions, lr_residuals, alpha=0.6, color='darkgreen', s=50, edgecolors='black', linewidth=0.5)
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.8)
        ax4.set_xlabel('Predicted Yield', fontweight='bold')
        ax4.set_ylabel('Residuals', fontweight='bold')
        ax4.set_title('Linear Regression: Residuals vs Predicted', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add main title
        fig.suptitle('Crop Yield Prediction Model Performance Comparison', fontsize=16, fontweight='bold', y=0.98)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save plot
        plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print("[OK] Comprehensive comparison plot created and saved")
        
        return fig
    
    def create_performance_metrics_chart(self):
        """
        Create a bar chart comparing performance metrics
        """
        print("Creating performance metrics comparison chart...")
        
        # Calculate metrics
        rf_mae = mean_absolute_error(self.y_test, self.rf_predictions)
        rf_rmse = np.sqrt(mean_squared_error(self.y_test, self.rf_predictions))
        rf_r2 = r2_score(self.y_test, self.rf_predictions)
        
        lr_mae = mean_absolute_error(self.y_test, self.lr_predictions)
        lr_rmse = np.sqrt(mean_squared_error(self.y_test, self.lr_predictions))
        lr_r2 = r2_score(self.y_test, self.lr_predictions)
        
        # Create data for plotting
        metrics = ['MAE', 'RMSE', 'R²']
        rf_values = [rf_mae, rf_rmse, rf_r2]
        lr_values = [lr_mae, lr_rmse, lr_r2]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        bars1 = ax.bar(x - width/2, rf_values, width, label='Random Forest', 
                      color='steelblue', alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, lr_values, width, label='Linear Regression',
                      color='darkgreen', alpha=0.8, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.set_ylabel('Values', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('performance_metrics_comparison.png', dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.show()
        
        print("[OK] Performance metrics chart created and saved")
        
        return fig
    
    def run_complete_visualization(self):
        """
        Run the complete visualization pipeline
        """
        print("="*70)
        print("ENHANCED CROP YIELD PREDICTION VISUALIZATIONS")
        print("="*70)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Train models
        self.train_models()
        
        # Create all visualizations
        self.create_actual_vs_predicted_plots()
        self.create_feature_importance_chart()
        self.create_comprehensive_comparison_plot()
        self.create_performance_metrics_chart()
        
        print("\n" + "="*70)
        print("ALL VISUALIZATIONS COMPLETED SUCCESSFULLY!")
        print("="*70)
        print("\nGenerated files:")
        print("- enhanced_actual_vs_predicted.png")
        print("- enhanced_feature_importance.png")
        print("- comprehensive_model_comparison.png")
        print("- performance_metrics_comparison.png")
        
        return True

def main():
    """
    Main function to run the enhanced visualization pipeline
    """
    # Create visualizer instance
    visualizer = CropYieldVisualizer()
    
    # Run complete visualization pipeline
    success = visualizer.run_complete_visualization()
    
    if success:
        print("\nEnhanced visualizations completed successfully!")
        print("All plots have been saved as high-quality PNG files.")
    else:
        print("Error in visualization pipeline.")

if __name__ == "__main__":
    main()
