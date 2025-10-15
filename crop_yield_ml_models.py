"""
Crop Yield Prediction Machine Learning Models
============================================

This script trains and evaluates Random Forest and Linear Regression models
for crop yield prediction using the preprocessed agricultural dataset.

Features:
- Data splitting (80/20 train/test)
- Random Forest Regressor training
- Linear Regression training
- Performance evaluation (MAE, RMSE, R²)
- Model comparison and analysis
- Feature importance analysis

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class CropYieldMLPipeline:
    """
    A comprehensive class for training and evaluating crop yield prediction models
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.rf_model = None
        self.lr_model = None
        self.rf_predictions = None
        self.lr_predictions = None
        self.results = {}
        
    def load_preprocessed_data(self, file_path='preprocessed_agricultural_data.csv'):
        """
        Load the preprocessed agricultural dataset
        
        Args:
            file_path (str): Path to the preprocessed CSV file
            
        Returns:
            pd.DataFrame: Loaded dataset
        """
        print("Loading preprocessed agricultural dataset...")
        
        try:
            df = pd.read_csv(file_path)
            print(f"[OK] Dataset loaded: {df.shape}")
            print(f"[INFO] Features: {df.shape[1] - 1}, Samples: {df.shape[0]}")
            return df
        except FileNotFoundError:
            print(f"[ERROR] File not found: {file_path}")
            print("[INFO] Please run the preprocessing pipeline first")
            return None
    
    def prepare_features_and_target(self, df):
        """
        Prepare features and target variable for modeling
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            tuple: (X, y, feature_names)
        """
        print("Preparing features and target variable...")
        
        # Identify target variable
        target_column = 'Yield'
        if target_column not in df.columns:
            print(f"[ERROR] Target column '{target_column}' not found")
            return None, None, None
        
        # Separate features and target
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        # Remove non-feature columns (dates, identifiers, etc.)
        columns_to_drop = ['Planting_Date', 'Harvest_Date', 'Year', 'Region_Code', 'Country_Name']
        X = X.drop(columns=[col for col in columns_to_drop if col in X.columns])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"[OK] Features prepared: {X.shape[1]} features, {X.shape[0]} samples")
        print(f"[OK] Target variable range: {y.min():.2f} - {y.max():.2f}")
        
        return X, y, self.feature_names
    
    def split_data(self, X, y, test_size=0.2):
        """
        Split data into training and testing sets
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        print(f"Splitting data into {int((1-test_size)*100)}/{int(test_size*100)} train/test sets...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=None  # No stratification for regression
        )
        
        print(f"[OK] Training set: {self.X_train.shape}")
        print(f"[OK] Testing set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_random_forest(self, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        """
        Train Random Forest Regressor
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of trees
            min_samples_split (int): Minimum samples required to split
            min_samples_leaf (int): Minimum samples required at leaf node
            
        Returns:
            RandomForestRegressor: Trained model
        """
        print("Training Random Forest Regressor...")
        
        self.rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1  # Use all available cores
        )
        
        self.rf_model.fit(self.X_train, self.y_train)
        print(f"[OK] Random Forest trained with {n_estimators} trees")
        
        return self.rf_model
    
    def train_linear_regression(self):
        """
        Train Linear Regression model
        
        Returns:
            LinearRegression: Trained model
        """
        print("Training Linear Regression model...")
        
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train, self.y_train)
        print("[OK] Linear Regression trained")
        
        return self.lr_model
    
    def make_predictions(self):
        """
        Make predictions using both trained models
        
        Returns:
            tuple: (rf_predictions, lr_predictions)
        """
        print("Making predictions...")
        
        if self.rf_model is None or self.lr_model is None:
            print("[ERROR] Models not trained yet")
            return None, None
        
        self.rf_predictions = self.rf_model.predict(self.X_test)
        self.lr_predictions = self.lr_model.predict(self.X_test)
        
        print("[OK] Predictions made for both models")
        
        return self.rf_predictions, self.lr_predictions
    
    def evaluate_model(self, predictions, model_name):
        """
        Evaluate model performance using multiple metrics
        
        Args:
            predictions (np.array): Model predictions
            model_name (str): Name of the model
            
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        mae = mean_absolute_error(self.y_test, predictions)
        rmse = np.sqrt(mean_squared_error(self.y_test, predictions))
        r2 = r2_score(self.y_test, predictions)
        
        metrics = {
            'Model': model_name,
            'MAE': mae,
            'RMSE': rmse,
            'R²': r2
        }
        
        print(f"\n{model_name} Performance:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        return metrics
    
    def compare_models(self):
        """
        Compare performance of both models
        
        Returns:
            pd.DataFrame: Comparison results
        """
        print("\n" + "="*60)
        print("MODEL PERFORMANCE COMPARISON")
        print("="*60)
        
        # Evaluate both models
        rf_metrics = self.evaluate_model(self.rf_predictions, "Random Forest")
        lr_metrics = self.evaluate_model(self.lr_predictions, "Linear Regression")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame([rf_metrics, lr_metrics])
        
        # Store results
        self.results['comparison'] = comparison_df
        self.results['rf_metrics'] = rf_metrics
        self.results['lr_metrics'] = lr_metrics
        
        # Display comparison
        print("\nComparison Summary:")
        print(comparison_df.round(4))
        
        # Determine best model
        best_mae = comparison_df.loc[comparison_df['MAE'].idxmin(), 'Model']
        best_rmse = comparison_df.loc[comparison_df['RMSE'].idxmin(), 'Model']
        best_r2 = comparison_df.loc[comparison_df['R²'].idxmax(), 'Model']
        
        print(f"\nBest Model by Metric:")
        print(f"  Best MAE:  {best_mae}")
        print(f"  Best RMSE: {best_rmse}")
        print(f"  Best R²:   {best_r2}")
        
        return comparison_df
    
    def analyze_feature_importance(self, top_n=15):
        """
        Analyze feature importance from Random Forest model
        
        Args:
            top_n (int): Number of top features to display
            
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if self.rf_model is None:
            print("[ERROR] Random Forest model not trained yet")
            return None
        
        print(f"\nTop {top_n} Most Important Features (Random Forest):")
        print("-" * 50)
        
        # Get feature importance
        importance = self.rf_model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        }).sort_values('Importance', ascending=False)
        
        # Display top features
        top_features = feature_importance_df.head(top_n)
        for i, (_, row) in enumerate(top_features.iterrows(), 1):
            print(f"{i:2d}. {row['Feature']:<25} {row['Importance']:.4f}")
        
        # Store results
        self.results['feature_importance'] = feature_importance_df
        
        return feature_importance_df
    
    def analyze_residuals(self):
        """
        Analyze prediction residuals for both models
        """
        print("\nResidual Analysis:")
        print("-" * 30)
        
        # Calculate residuals
        rf_residuals = self.y_test - self.rf_predictions
        lr_residuals = self.y_test - self.lr_predictions
        
        # Residual statistics
        print(f"Random Forest Residuals:")
        print(f"  Mean: {rf_residuals.mean():.4f}")
        print(f"  Std:  {rf_residuals.std():.4f}")
        print(f"  Min:  {rf_residuals.min():.4f}")
        print(f"  Max:  {rf_residuals.max():.4f}")
        
        print(f"\nLinear Regression Residuals:")
        print(f"  Mean: {lr_residuals.mean():.4f}")
        print(f"  Std:  {lr_residuals.std():.4f}")
        print(f"  Min:  {lr_residuals.min():.4f}")
        print(f"  Max:  {lr_residuals.max():.4f}")
        
        # Store results
        self.results['rf_residuals'] = rf_residuals
        self.results['lr_residuals'] = lr_residuals
    
    def create_predictions_comparison(self):
        """
        Create a comparison of actual vs predicted values
        """
        print("\nPrediction Comparison (First 10 test samples):")
        print("-" * 55)
        
        comparison_df = pd.DataFrame({
            'Actual': self.y_test.iloc[:10].values,
            'RF_Predicted': self.rf_predictions[:10],
            'LR_Predicted': self.lr_predictions[:10],
            'RF_Error': np.abs(self.y_test.iloc[:10].values - self.rf_predictions[:10]),
            'LR_Error': np.abs(self.y_test.iloc[:10].values - self.lr_predictions[:10])
        })
        
        print(comparison_df.round(4))
        
        # Store results
        self.results['predictions_comparison'] = comparison_df
        
        return comparison_df
    
    def run_complete_pipeline(self, data_path='preprocessed_agricultural_data.csv'):
        """
        Run the complete machine learning pipeline
        
        Args:
            data_path (str): Path to the preprocessed dataset
            
        Returns:
            dict: Complete results dictionary
        """
        print("="*70)
        print("CROP YIELD PREDICTION MACHINE LEARNING PIPELINE")
        print("="*70)
        
        # Step 1: Load data
        df = self.load_preprocessed_data(data_path)
        if df is None:
            return None
        
        # Step 2: Prepare features and target
        X, y, feature_names = self.prepare_features_and_target(df)
        if X is None:
            return None
        
        # Step 3: Split data
        self.split_data(X, y)
        
        # Step 4: Train models
        self.train_random_forest()
        self.train_linear_regression()
        
        # Step 5: Make predictions
        self.make_predictions()
        
        # Step 6: Compare models
        comparison_df = self.compare_models()
        
        # Step 7: Feature importance analysis
        feature_importance = self.analyze_feature_importance()
        
        # Step 8: Residual analysis
        self.analyze_residuals()
        
        # Step 9: Prediction comparison
        predictions_comparison = self.create_predictions_comparison()
        
        print("\n" + "="*70)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        return self.results
    
    def save_results(self, filename='ml_results_summary.txt'):
        """
        Save results to a text file
        
        Args:
            filename (str): Output filename
        """
        with open(filename, 'w') as f:
            f.write("CROP YIELD PREDICTION - MODEL RESULTS\n")
            f.write("="*50 + "\n\n")
            
            # Model comparison
            f.write("MODEL PERFORMANCE COMPARISON:\n")
            f.write("-" * 30 + "\n")
            f.write(self.results['comparison'].to_string(index=False))
            f.write("\n\n")
            
            # Feature importance
            f.write("TOP 15 MOST IMPORTANT FEATURES:\n")
            f.write("-" * 30 + "\n")
            top_features = self.results['feature_importance'].head(15)
            for i, (_, row) in enumerate(top_features.iterrows(), 1):
                f.write(f"{i:2d}. {row['Feature']:<25} {row['Importance']:.4f}\n")
            
        print(f"[OK] Results saved to: {filename}")

def main():
    """
    Main function to demonstrate the machine learning pipeline
    """
    # Create ML pipeline instance
    ml_pipeline = CropYieldMLPipeline(random_state=42)
    
    # Run the complete pipeline
    results = ml_pipeline.run_complete_pipeline()
    
    if results is not None:
        # Save results
        ml_pipeline.save_results()
        
        # Additional analysis
        print("\nAdditional Analysis:")
        print("-" * 20)
        
        # Model performance summary
        rf_r2 = results['rf_metrics']['R²']
        lr_r2 = results['lr_metrics']['R²']
        
        if rf_r2 > lr_r2:
            print(f"Random Forest performs better with R² = {rf_r2:.4f} vs Linear Regression R² = {lr_r2:.4f}")
        else:
            print(f"Linear Regression performs better with R² = {lr_r2:.4f} vs Random Forest R² = {rf_r2:.4f}")
        
        # Feature importance insights
        top_feature = results['feature_importance'].iloc[0]
        print(f"\nMost important feature: {top_feature['Feature']} (importance: {top_feature['Importance']:.4f})")
        
        print("\nMachine learning pipeline completed successfully!")

if __name__ == "__main__":
    main()
