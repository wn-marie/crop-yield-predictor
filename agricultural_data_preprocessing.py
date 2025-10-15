"""
Agricultural Data Preprocessing Pipeline
=======================================

This script loads multiple agricultural datasets from CSV files, merges them,
and performs comprehensive preprocessing including:
- Missing value handling
- Numerical feature normalization
- Categorical variable encoding

Datasets:
1. Agri-yield-prediction.csv - Main yield prediction dataset
2. Agricultural-land (%oflandarea).csv - Land area percentage data
3. crop-production.csv - Crop production index data
4. Fertilizer-consumption.csv - Fertilizer consumption data

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class AgriculturalDataPreprocessor:
    """
    A comprehensive class for preprocessing agricultural datasets
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)
        self.numerical_imputer = KNNImputer(n_neighbors=5)
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.preprocessed_data = None
        self.feature_names = None
        
    def load_datasets(self, data_path='data/'):
        """
        Load all agricultural datasets from CSV files
        
        Args:
            data_path (str): Path to the data directory
            
        Returns:
            dict: Dictionary containing all loaded datasets
        """
        print("Loading agricultural datasets...")
        
        datasets = {}
        
        try:
            # Load main yield prediction dataset
            datasets['yield_prediction'] = pd.read_csv(f'{data_path}Agri-yield-prediction.csv')
            print(f"[OK] Loaded yield prediction dataset: {datasets['yield_prediction'].shape}")
            
            # Load agricultural land percentage dataset
            datasets['agricultural_land'] = pd.read_csv(f'{data_path}Agricultural-land (%oflandarea).csv')
            print(f"[OK] Loaded agricultural land dataset: {datasets['agricultural_land'].shape}")
            
            # Load crop production dataset
            datasets['crop_production'] = pd.read_csv(f'{data_path}crop-production.csv')
            print(f"[OK] Loaded crop production dataset: {datasets['crop_production'].shape}")
            
            # Load fertilizer consumption dataset
            datasets['fertilizer_consumption'] = pd.read_csv(f'{data_path}Fertilizer-consumption.csv')
            print(f"[OK] Loaded fertilizer consumption dataset: {datasets['fertilizer_consumption'].shape}")
            
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}")
            return None
            
        return datasets
    
    def preprocess_yield_dataset(self, df):
        """
        Preprocess the main yield prediction dataset
        
        Args:
            df (pd.DataFrame): Main yield prediction dataset
            
        Returns:
            pd.DataFrame: Preprocessed yield dataset
        """
        print("Preprocessing yield prediction dataset...")
        
        # Convert date columns to datetime
        df['Planting_Date'] = pd.to_datetime(df['Planting_Date'], errors='coerce')
        df['Harvest_Date'] = pd.to_datetime(df['Harvest_Date'], errors='coerce')
        
        # Extract year from dates for merging
        df['Year'] = df['Planting_Date'].dt.year.fillna(df['Year'])
        
        # Create region mapping (since the dataset uses directions like North, South, East, West)
        region_mapping = {
            'North': 'North',
            'South': 'South', 
            'East': 'East',
            'West': 'West'
        }
        df['Region_Code'] = df['Region'].map(region_mapping)
        
        print(f"[OK] Preprocessed yield dataset: {df.shape}")
        return df
    
    def preprocess_country_datasets(self, datasets):
        """
        Preprocess country-level datasets (land, production, fertilizer)
        
        Args:
            datasets (dict): Dictionary containing country-level datasets
            
        Returns:
            dict: Dictionary containing preprocessed country datasets
        """
        print("Preprocessing country-level datasets...")
        
        processed_datasets = {}
        
        for name, df in datasets.items():
            if name == 'yield_prediction':
                continue
                
            print(f"Processing {name}...")
            
            # Melt the dataset to convert year columns to rows
            year_columns = [col for col in df.columns if col.startswith(('199', '200', '201', '202'))]
            
            if year_columns:
                df_melted = pd.melt(
                    df, 
                    id_vars=['Country Name', 'Country Code', 'Series Name', 'Series Code'],
                    value_vars=year_columns,
                    var_name='Year_Str',
                    value_name='Value'
                )
                
                # Extract year from column names
                df_melted['Year'] = df_melted['Year_Str'].str.extract(r'(\d{4})').astype(float)
                df_melted = df_melted.drop('Year_Str', axis=1)
                
                # Clean up the data
                df_melted = df_melted.dropna(subset=['Year'])
                df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')
                
                # Create a simplified version for merging
                df_simplified = df_melted.groupby(['Country Name', 'Year'])['Value'].mean().reset_index()
                df_simplified.columns = ['Country_Name', 'Year', f'{name}_value']
                
                processed_datasets[name] = df_simplified
                print(f"[OK] Processed {name}: {df_simplified.shape}")
        
        return processed_datasets
    
    def merge_datasets(self, yield_df, country_datasets):
        """
        Merge all datasets based on region/country and year
        
        Args:
            yield_df (pd.DataFrame): Preprocessed yield dataset
            country_datasets (dict): Dictionary of preprocessed country datasets
            
        Returns:
            pd.DataFrame: Merged dataset
        """
        print("Merging datasets...")
        
        # Start with the yield prediction dataset
        merged_df = yield_df.copy()
        
        # Create a country-region mapping for merging
        # Since yield dataset uses regions (North, South, East, West) and country datasets use country names,
        # we'll create a mapping or use the existing country information
        
        # For this example, let's assume we can map regions to representative countries
        # In a real scenario, you might have more specific geographic data
        region_country_mapping = {
            'North': 'India',  # Assuming this represents North India
            'South': 'India',  # Assuming this represents South India  
            'East': 'India',   # Assuming this represents East India
            'West': 'India'    # Assuming this represents West India
        }
        
        merged_df['Country_Name'] = merged_df['Region'].map(region_country_mapping)
        
        # Merge with country datasets
        for name, df in country_datasets.items():
            if not df.empty:
                merged_df = pd.merge(
                    merged_df, 
                    df, 
                    left_on=['Country_Name', 'Year'], 
                    right_on=['Country_Name', 'Year'],
                    how='left'
                )
                print(f"[OK] Merged with {name}")
        
        print(f"[OK] Final merged dataset shape: {merged_df.shape}")
        return merged_df
    
    def identify_feature_types(self, df):
        """
        Identify numerical and categorical features
        
        Args:
            df (pd.DataFrame): Input dataset
            
        Returns:
            tuple: (numerical_features, categorical_features)
        """
        numerical_features = []
        categorical_features = []
        
        for col in df.columns:
            if col in ['Yield']:  # Skip target variable
                continue
            elif df[col].dtype in ['int64', 'float64']:
                numerical_features.append(col)
            else:
                categorical_features.append(col)
        
        print(f"Numerical features: {len(numerical_features)}")
        print(f"Categorical features: {len(categorical_features)}")
        
        return numerical_features, categorical_features
    
    def handle_missing_values(self, df, numerical_features, categorical_features):
        """
        Handle missing values in the dataset
        
        Args:
            df (pd.DataFrame): Input dataset
            numerical_features (list): List of numerical feature names
            categorical_features (list): List of categorical feature names
            
        Returns:
            pd.DataFrame: Dataset with missing values handled
        """
        print("Handling missing values...")
        
        df_clean = df.copy()
        
        # Handle numerical missing values using KNN imputation
        if numerical_features:
            print(f"Imputing {len(numerical_features)} numerical features...")
            df_clean[numerical_features] = self.numerical_imputer.fit_transform(df_clean[numerical_features])
        
        # Handle categorical missing values using mode imputation
        if categorical_features:
            print(f"Imputing {len(categorical_features)} categorical features...")
            df_clean[categorical_features] = self.categorical_imputer.fit_transform(df_clean[categorical_features])
        
        print(f"[OK] Missing values handled. Remaining missing values: {df_clean.isnull().sum().sum()}")
        return df_clean
    
    def normalize_numerical_features(self, df, numerical_features):
        """
        Normalize numerical features using StandardScaler
        
        Args:
            df (pd.DataFrame): Input dataset
            numerical_features (list): List of numerical feature names
            
        Returns:
            pd.DataFrame: Dataset with normalized numerical features
        """
        print("Normalizing numerical features...")
        
        df_normalized = df.copy()
        
        if numerical_features:
            # Fit and transform numerical features
            df_normalized[numerical_features] = self.scaler.fit_transform(df_normalized[numerical_features])
            print(f"[OK] Normalized {len(numerical_features)} numerical features")
        
        return df_normalized
    
    def encode_categorical_features(self, df, categorical_features):
        """
        Encode categorical features using Label Encoding and One-Hot Encoding
        
        Args:
            df (pd.DataFrame): Input dataset
            categorical_features (list): List of categorical feature names
            
        Returns:
            pd.DataFrame: Dataset with encoded categorical features
        """
        print("Encoding categorical features...")
        
        df_encoded = df.copy()
        
        for feature in categorical_features:
            if feature in df_encoded.columns:
                # Use Label Encoding for ordinal categorical variables
                if feature in ['Growth_Stage', 'Irrigation_Frequency', 'Pesticide_Usage']:
                    if feature not in self.label_encoders:
                        self.label_encoders[feature] = LabelEncoder()
                    df_encoded[feature] = self.label_encoders[feature].fit_transform(df_encoded[feature].astype(str))
                    print(f"[OK] Label encoded: {feature}")
                
                # Use One-Hot Encoding for nominal categorical variables
                elif feature in ['Soil_Type', 'Crop_Type', 'Fertilizer_Type', 'Season', 'Region']:
                    # Create dummy variables
                    dummies = pd.get_dummies(df_encoded[feature], prefix=feature, drop_first=True)
                    df_encoded = pd.concat([df_encoded, dummies], axis=1)
                    df_encoded = df_encoded.drop(feature, axis=1)
                    print(f"[OK] One-hot encoded: {feature}")
        
        print(f"[OK] Encoded categorical features. New shape: {df_encoded.shape}")
        return df_encoded
    
    def create_preprocessing_pipeline(self, data_path='data/'):
        """
        Create the complete preprocessing pipeline
        
        Args:
            data_path (str): Path to the data directory
            
        Returns:
            pd.DataFrame: Fully preprocessed dataset
        """
        print("=" * 60)
        print("AGRICULTURAL DATA PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Step 1: Load datasets
        datasets = self.load_datasets(data_path)
        if datasets is None:
            return None
        
        # Step 2: Preprocess yield prediction dataset
        yield_df = self.preprocess_yield_dataset(datasets['yield_prediction'])
        
        # Step 3: Preprocess country-level datasets
        country_datasets = self.preprocess_country_datasets(datasets)
        
        # Step 4: Merge datasets
        merged_df = self.merge_datasets(yield_df, country_datasets)
        
        # Step 5: Identify feature types
        numerical_features, categorical_features = self.identify_feature_types(merged_df)
        
        # Step 6: Handle missing values
        df_clean = self.handle_missing_values(merged_df, numerical_features, categorical_features)
        
        # Step 7: Normalize numerical features
        df_normalized = self.normalize_numerical_features(df_clean, numerical_features)
        
        # Step 8: Encode categorical features
        df_final = self.encode_categorical_features(df_normalized, categorical_features)
        
        # Store the preprocessed data
        self.preprocessed_data = df_final
        self.feature_names = [col for col in df_final.columns if col != 'Yield']
        
        print("=" * 60)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Final dataset shape: {df_final.shape}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Target variable: Yield")
        print(f"Missing values: {df_final.isnull().sum().sum()}")
        
        return df_final
    
    def get_data_summary(self):
        """
        Get summary statistics of the preprocessed data
        
        Returns:
            pd.DataFrame: Summary statistics
        """
        if self.preprocessed_data is None:
            print("No preprocessed data available. Run create_preprocessing_pipeline() first.")
            return None
        
        print("\nDATA SUMMARY:")
        print("-" * 40)
        print(f"Dataset shape: {self.preprocessed_data.shape}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Target variable range: {self.preprocessed_data['Yield'].min():.2f} - {self.preprocessed_data['Yield'].max():.2f}")
        
        # Show first few rows
        print("\nFirst 5 rows:")
        print(self.preprocessed_data.head())
        
        # Show data types
        print("\nData types:")
        print(self.preprocessed_data.dtypes.value_counts())
        
        return self.preprocessed_data.describe()

def main():
    """
    Main function to demonstrate the preprocessing pipeline
    """
    # Create preprocessor instance
    preprocessor = AgriculturalDataPreprocessor()
    
    # Run the complete preprocessing pipeline
    processed_data = preprocessor.create_preprocessing_pipeline()
    
    if processed_data is not None:
        # Display summary
        summary = preprocessor.get_data_summary()
        
        # Save the preprocessed data
        output_file = 'preprocessed_agricultural_data.csv'
        processed_data.to_csv(output_file, index=False)
        print(f"\n[OK] Preprocessed data saved to: {output_file}")
        
        # Display feature importance (correlation with target)
        if 'Yield' in processed_data.columns:
            # Select only numerical columns for correlation
            numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
            correlations = processed_data[numerical_cols].corr()['Yield'].abs().sort_values(ascending=False)
            print("\nTop 10 features most correlated with Yield:")
            print(correlations.head(11)[1:])  # Skip Yield itself

if __name__ == "__main__":
    main()
