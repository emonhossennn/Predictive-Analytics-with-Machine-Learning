"""
Data Preprocessing Module

This module contains functions for data cleaning, feature engineering,
and preparation for machine learning models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from typing import Tuple, List, Optional, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    A comprehensive data preprocessing class for machine learning workflows.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
        self.feature_names = []
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from various file formats.
        
        Args:
            file_path: Path to the data file
            
        Returns:
            Loaded DataFrame
        """
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            elif file_path.endswith('.xlsx'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get basic information about the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary with basic dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'duplicates': df.duplicated().sum()
        }
        
        return info
    
    def handle_missing_values(self, df: pd.DataFrame, 
                            strategy: str = 'mean',
                            columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
            columns: Specific columns to impute (if None, impute all numeric columns)
            
        Returns:
            DataFrame with imputed values
        """
        df_copy = df.copy()
        
        if columns is None:
            numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = df_copy.select_dtypes(include=['object']).columns.tolist()
        else:
            numeric_columns = [col for col in columns if df_copy[col].dtype in ['int64', 'float64']]
            categorical_columns = [col for col in columns if df_copy[col].dtype == 'object']
        
        # Impute numeric columns
        if numeric_columns:
            if 'numeric' not in self.imputers:
                self.imputers['numeric'] = SimpleImputer(strategy=strategy)
                df_copy[numeric_columns] = self.imputers['numeric'].fit_transform(df_copy[numeric_columns])
            else:
                df_copy[numeric_columns] = self.imputers['numeric'].transform(df_copy[numeric_columns])
        
        # Impute categorical columns
        if categorical_columns:
            if 'categorical' not in self.imputers:
                self.imputers['categorical'] = SimpleImputer(strategy='most_frequent')
                df_copy[categorical_columns] = self.imputers['categorical'].fit_transform(df_copy[categorical_columns])
            else:
                df_copy[categorical_columns] = self.imputers['categorical'].transform(df_copy[categorical_columns])
        
        logger.info(f"Missing values handled using {strategy} strategy")
        return df_copy
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                  columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df: Input DataFrame
            columns: Specific columns to encode (if None, encode all object columns)
            
        Returns:
            DataFrame with encoded features
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=['object']).columns.tolist()
        
        for column in columns:
            if column in df_copy.columns:
                if column not in self.label_encoders:
                    self.label_encoders[column] = LabelEncoder()
                    df_copy[column] = self.label_encoders[column].fit_transform(df_copy[column].astype(str))
                else:
                    df_copy[column] = self.label_encoders[column].transform(df_copy[column].astype(str))
        
        logger.info(f"Categorical features encoded: {columns}")
        return df_copy
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        df_copy = df.copy()
        
        # Example feature engineering (customize based on your data)
        numeric_columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        # Create polynomial features for numeric columns (degree 2)
        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                df_copy[f'{col1}_{col2}_interaction'] = df_copy[col1] * df_copy[col2]
        
        # Create aggregated features
        if len(numeric_columns) > 1:
            df_copy['numeric_mean'] = df_copy[numeric_columns].mean(axis=1)
            df_copy['numeric_std'] = df_copy[numeric_columns].std(axis=1)
            df_copy['numeric_sum'] = df_copy[numeric_columns].sum(axis=1)
        
        logger.info(f"Created {len(df_copy.columns) - len(df.columns)} new features")
        return df_copy
    
    def scale_features(self, df: pd.DataFrame, 
                      columns: Optional[List[str]] = None,
                      fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame
            columns: Specific columns to scale (if None, scale all numeric columns)
            fit: Whether to fit the scaler or just transform
            
        Returns:
            DataFrame with scaled features
        """
        df_copy = df.copy()
        
        if columns is None:
            columns = df_copy.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            df_copy[columns] = self.scaler.fit_transform(df_copy[columns])
        else:
            df_copy[columns] = self.scaler.transform(df_copy[columns])
        
        logger.info(f"Features scaled: {columns}")
        return df_copy
    
    def split_data(self, df: pd.DataFrame, 
                   target_column: str,
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and testing sets.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            test_size: Proportion of test set
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def process_data(self, file_path: str, 
                    target_column: str,
                    test_size: float = 0.2,
                    handle_missing: bool = True,
                    encode_categorical: bool = True,
                    create_new_features: bool = True,
                    scale_features: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Complete data preprocessing pipeline.
        
        Args:
            file_path: Path to the data file
            target_column: Name of the target column
            test_size: Proportion of test set
            handle_missing: Whether to handle missing values
            encode_categorical: Whether to encode categorical features
            create_new_features: Whether to create new features
            scale_features: Whether to scale features
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Starting data preprocessing pipeline...")
        
        # Load data
        df = self.load_data(file_path)
        
        # Basic info
        info = self.basic_info(df)
        logger.info(f"Dataset info: {info}")
        
        # Handle missing values
        if handle_missing:
            df = self.handle_missing_values(df)
        
        # Encode categorical features
        if encode_categorical:
            df = self.encode_categorical_features(df)
        
        # Create new features
        if create_new_features:
            df = self.create_features(df)
        
        # Split data before scaling to avoid data leakage
        X_train, X_test, y_train, y_test = self.split_data(df, target_column, test_size)
        
        # Scale features
        if scale_features:
            X_train = self.scale_features(X_train, fit=True)
            X_test = self.scale_features(X_test, fit=False)
        
        # Store feature names
        self.feature_names = X_train.columns.tolist()
        
        logger.info("Data preprocessing pipeline completed successfully")
        return X_train, X_test, y_train, y_test


def main():
    """
    Example usage of the DataPreprocessor class.
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Example: Process your data
    # X_train, X_test, y_train, y_test = preprocessor.process_data(
    #     file_path='data/raw/your_data.csv',
    #     target_column='target'
    # )
    
    print("DataPreprocessor initialized successfully!")
    print("Use process_data() method to preprocess your dataset.")


if __name__ == "__main__":
    main()
