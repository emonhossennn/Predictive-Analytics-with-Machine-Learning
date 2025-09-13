"""
Prediction Module

This module contains functions for making predictions using trained models
and handling prediction workflows for predictive analytics.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import Union, List, Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Predictor:
    """
    A comprehensive prediction class for making forecasts with trained models.
    """
    
    def __init__(self, model_path: Optional[str] = None, model: Optional[Any] = None):
        """
        Initialize the predictor with either a model path or a model object.
        
        Args:
            model_path: Path to the saved model file
            model: Pre-trained model object
        """
        self.model = None
        self.model_path = model_path
        self.feature_names = []
        self.preprocessing_pipeline = None
        
        if model_path:
            self.load_model(model_path)
        elif model is not None:
            self.model = model
            
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.model = joblib.load(model_path)
            self.model_path = model_path
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {str(e)}")
            return False
    
    def save_model(self, save_path: str) -> bool:
        """
        Save the current model to disk.
        
        Args:
            save_path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No model to save")
            return False
            
        try:
            joblib.dump(self.model, save_path)
            logger.info(f"Model saved successfully to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {str(e)}")
            return False
    
    def predict(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """
        Make predictions using the loaded model.
        
        Args:
            X: Input features for prediction
            
        Returns:
            Predictions as numpy array
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            # Convert input to appropriate format
            if isinstance(X, list):
                X = np.array(X)
            elif isinstance(X, pd.DataFrame):
                X = X.values
            
            # Reshape if single prediction
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            predictions = self.model.predict(X)
            logger.info(f"Made predictions for {len(predictions)} samples")
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise
    
    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray, List]) -> np.ndarray:
        """
        Make probability predictions (for models that support it).
        
        Args:
            X: Input features for prediction
            
        Returns:
            Probability predictions as numpy array
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError("Model does not support probability predictions")
        
        try:
            # Convert input to appropriate format
            if isinstance(X, list):
                X = np.array(X)
            elif isinstance(X, pd.DataFrame):
                X = X.values
            
            # Reshape if single prediction
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            probabilities = self.model.predict_proba(X)
            logger.info(f"Made probability predictions for {len(probabilities)} samples")
            
            return probabilities
            
        except Exception as e:
            logger.error(f"Error making probability predictions: {str(e)}")
            raise
    
    def predict_single(self, features: Union[List, Dict]) -> float:
        """
        Make a prediction for a single sample.
        
        Args:
            features: Feature values as list or dictionary
            
        Returns:
            Single prediction value
        """
        try:
            if isinstance(features, dict):
                # Ensure features are in the correct order
                if self.feature_names:
                    features = [features.get(name, 0) for name in self.feature_names]
                else:
                    features = list(features.values())
            
            prediction = self.predict([features])
            return float(prediction[0])
            
        except Exception as e:
            logger.error(f"Error making single prediction: {str(e)}")
            raise
    
    def predict_batch(self, data: pd.DataFrame, 
                     output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Make predictions for a batch of data.
        
        Args:
            data: Input DataFrame with features
            output_path: Optional path to save predictions
            
        Returns:
            DataFrame with original data and predictions
        """
        try:
            predictions = self.predict(data)
            
            # Create results DataFrame
            results_df = data.copy()
            results_df['predictions'] = predictions
            
            # Add prediction metadata
            results_df['prediction_timestamp'] = pd.Timestamp.now()
            
            if output_path:
                results_df.to_csv(output_path, index=False)
                logger.info(f"Batch predictions saved to {output_path}")
            
            logger.info(f"Made batch predictions for {len(results_df)} samples")
            return results_df
            
        except Exception as e:
            logger.error(f"Error making batch predictions: {str(e)}")
            raise
    
    def predict_with_confidence(self, X: Union[pd.DataFrame, np.ndarray], 
                               confidence_level: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with confidence intervals (for applicable models).
        
        Args:
            X: Input features
            confidence_level: Confidence level for intervals
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            predictions = self.predict(X)
            
            # For models that support prediction intervals
            if hasattr(self.model, 'predict') and hasattr(self.model, 'estimators_'):
                # Bootstrap approach for ensemble models
                all_predictions = []
                for estimator in self.model.estimators_:
                    est_pred = estimator.predict(X)
                    all_predictions.append(est_pred)
                
                all_predictions = np.array(all_predictions)
                
                # Calculate confidence intervals
                alpha = 1 - confidence_level
                lower_percentile = (alpha / 2) * 100
                upper_percentile = (1 - alpha / 2) * 100
                
                lower_bounds = np.percentile(all_predictions, lower_percentile, axis=0)
                upper_bounds = np.percentile(all_predictions, upper_percentile, axis=0)
                
                logger.info(f"Made predictions with {confidence_level*100}% confidence intervals")
                return predictions, lower_bounds, upper_bounds
            else:
                # Simple standard deviation based approach
                std_pred = np.std(predictions) if len(predictions) > 1 else np.std([predictions])
                z_score = 1.96 if confidence_level == 0.95 else 2.576  # Simplified
                
                margin = z_score * std_pred
                lower_bounds = predictions - margin
                upper_bounds = predictions + margin
                
                logger.info(f"Made predictions with approximate {confidence_level*100}% confidence intervals")
                return predictions, lower_bounds, upper_bounds
                
        except Exception as e:
            logger.error(f"Error making predictions with confidence: {str(e)}")
            raise
    
    def feature_importance_prediction(self, X: Union[pd.DataFrame, np.ndarray],
                                    feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Make predictions and analyze feature importance for the prediction.
        
        Args:
            X: Input features
            feature_names: Names of features
            
        Returns:
            Dictionary with predictions and feature analysis
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        try:
            predictions = self.predict(X)
            
            result = {
                'predictions': predictions,
                'feature_analysis': {}
            }
            
            # Feature importance for tree-based models
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                if feature_names:
                    feature_importance = dict(zip(feature_names, importances))
                    # Sort by importance
                    feature_importance = dict(sorted(feature_importance.items(), 
                                                   key=lambda x: x[1], reverse=True))
                    result['feature_analysis']['feature_importance'] = feature_importance
            
            # Coefficients for linear models
            if hasattr(self.model, 'coef_'):
                coefs = self.model.coef_
                if feature_names and len(feature_names) == len(coefs):
                    feature_coefs = dict(zip(feature_names, coefs))
                    result['feature_analysis']['feature_coefficients'] = feature_coefs
            
            return result
            
        except Exception as e:
            logger.error(f"Error in feature importance prediction: {str(e)}")
            raise
    
    def predict_time_series(self, data: pd.DataFrame, 
                           time_column: str, 
                           periods_ahead: int = 1) -> pd.DataFrame:
        """
        Make time series predictions.
        
        Args:
            data: Historical data
            time_column: Name of the time column
            periods_ahead: Number of periods to predict ahead
            
        Returns:
            DataFrame with predictions
        """
        try:
            # Sort by time column
            data_sorted = data.sort_values(time_column).copy()
            
            predictions = []
            current_data = data_sorted.copy()
            
            for i in range(periods_ahead):
                # Use last row for prediction
                last_features = current_data.drop(columns=[time_column]).iloc[-1:].values
                pred = self.predict(last_features)[0]
                
                # Create new time point
                if pd.api.types.is_datetime64_any_dtype(current_data[time_column]):
                    # For datetime, add one period (assuming daily frequency)
                    next_time = current_data[time_column].iloc[-1] + pd.Timedelta(days=1)
                else:
                    # For numeric time, add one unit
                    next_time = current_data[time_column].iloc[-1] + 1
                
                predictions.append({
                    time_column: next_time,
                    'prediction': pred,
                    'period_ahead': i + 1
                })
                
                # Add prediction to data for next iteration (if making multi-step predictions)
                new_row = current_data.iloc[-1:].copy()
                new_row[time_column] = next_time
                # You might want to update other features based on the prediction
                
            predictions_df = pd.DataFrame(predictions)
            logger.info(f"Made time series predictions for {periods_ahead} periods ahead")
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error in time series prediction: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.model is None:
            return {"error": "No model loaded"}
        
        info = {
            "model_type": type(self.model).__name__,
            "model_path": self.model_path,
        }
        
        # Add model-specific information
        if hasattr(self.model, 'n_features_in_'):
            info["n_features"] = self.model.n_features_in_
        
        if hasattr(self.model, 'feature_names_in_'):
            info["feature_names"] = list(self.model.feature_names_in_)
        
        if hasattr(self.model, 'score'):
            info["has_score_method"] = True
        
        if hasattr(self.model, 'predict_proba'):
            info["supports_probabilities"] = True
        
        if hasattr(self.model, 'feature_importances_'):
            info["supports_feature_importance"] = True
        
        return info


def create_prediction_pipeline(preprocessor, model, save_path: Optional[str] = None):
    """
    Create a complete prediction pipeline combining preprocessing and model.
    
    Args:
        preprocessor: Fitted preprocessing object
        model: Trained model
        save_path: Path to save the pipeline
        
    Returns:
        Pipeline object
    """
    from sklearn.pipeline import Pipeline
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    if save_path:
        joblib.dump(pipeline, save_path)
        logger.info(f"Pipeline saved to {save_path}")
    
    return pipeline


def main():
    """
    Example usage of the Predictor class.
    """
    # Initialize predictor (without loading a model)
    predictor = Predictor()
    
    print("Predictor initialized successfully!")
    print("Available methods:")
    print("- load_model()")
    print("- predict()")
    print("- predict_single()")
    print("- predict_batch()")
    print("- predict_with_confidence()")
    print("- feature_importance_prediction()")
    print("- predict_time_series()")
    print("- get_model_info()")
    
    print("\nTo use the predictor:")
    print("1. Load a model: predictor.load_model('path/to/model.pkl')")
    print("2. Make predictions: predictions = predictor.predict(X_test)")


if __name__ == "__main__":
    main()
