"""
Model Training Module

This module contains functions for training various machine learning models
for predictive analytics and regression tasks.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    A comprehensive model training class for regression tasks.
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.best_model = None
        self.best_score = float('-inf')
        
    def initialize_models(self) -> Dict[str, Any]:
        """
        Initialize various regression models with default parameters.
        
        Returns:
            Dictionary of initialized models
        """
        models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(random_state=self.random_state),
            'lasso': Lasso(random_state=self.random_state),
            'elastic_net': ElasticNet(random_state=self.random_state),
            'random_forest': RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100,
                random_state=self.random_state
            ),
            'decision_tree': DecisionTreeRegressor(random_state=self.random_state),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'svr': SVR(kernel='rbf')
        }
        
        self.models = models
        logger.info(f"Initialized {len(models)} models")
        return models
    
    def train_single_model(self, model: Any, X_train: pd.DataFrame, 
                          y_train: pd.Series, model_name: str) -> Any:
        """
        Train a single model.
        
        Args:
            model: The model to train
            X_train: Training features
            y_train: Training target
            model_name: Name of the model
            
        Returns:
            Trained model
        """
        try:
            logger.info(f"Training {model_name}...")
            model.fit(X_train, y_train)
            self.trained_models[model_name] = model
            logger.info(f"Successfully trained {model_name}")
            return model
        except Exception as e:
            logger.error(f"Error training {model_name}: {str(e)}")
            return None
    
    def train_multiple_models(self, X_train: pd.DataFrame, 
                            y_train: pd.Series) -> Dict[str, Any]:
        """
        Train multiple models and return them.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        if not self.models:
            self.initialize_models()
        
        logger.info("Starting training for multiple models...")
        
        for model_name, model in self.models.items():
            self.train_single_model(model, X_train, y_train, model_name)
        
        logger.info(f"Training completed for {len(self.trained_models)} models")
        return self.trained_models
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, 
                      y_test: pd.Series) -> Dict[str, float]:
        """
        Evaluate a single model using various metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            y_pred = model.predict(X_test)
            
            metrics = {
                'mse': mean_squared_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'mae': mean_absolute_error(y_test, y_pred),
                'r2': r2_score(y_test, y_pred)
            }
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return {}
    
    def cross_validate_model(self, model: Any, X: pd.DataFrame, 
                           y: pd.Series, cv: int = 5,
                           scoring: str = 'neg_mean_squared_error') -> Dict[str, float]:
        """
        Perform cross-validation on a model.
        
        Args:
            model: Model to cross-validate
            X: Features
            y: Target
            cv: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Cross-validation results
        """
        try:
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
            
            cv_results = {
                'mean_score': scores.mean(),
                'std_score': scores.std(),
                'scores': scores.tolist()
            }
            
            return cv_results
        except Exception as e:
            logger.error(f"Error in cross-validation: {str(e)}")
            return {}
    
    def hyperparameter_tuning(self, model_name: str, X_train: pd.DataFrame,
                            y_train: pd.Series, param_grid: Dict[str, List],
                            search_type: str = 'grid',
                            cv: int = 5, n_iter: int = 100) -> Any:
        """
        Perform hyperparameter tuning for a specific model.
        
        Args:
            model_name: Name of the model to tune
            X_train: Training features
            y_train: Training target
            param_grid: Parameter grid for tuning
            search_type: Type of search ('grid' or 'random')
            cv: Number of cross-validation folds
            n_iter: Number of iterations for random search
            
        Returns:
            Best model after tuning
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not found")
            return None
        
        model = self.models[model_name]
        
        try:
            logger.info(f"Starting hyperparameter tuning for {model_name}...")
            
            if search_type == 'grid':
                search = GridSearchCV(
                    model, param_grid, cv=cv, 
                    scoring='neg_mean_squared_error',
                    n_jobs=-1
                )
            else:
                search = RandomizedSearchCV(
                    model, param_grid, cv=cv,
                    n_iter=n_iter, scoring='neg_mean_squared_error',
                    n_jobs=-1, random_state=self.random_state
                )
            
            search.fit(X_train, y_train)
            
            best_model = search.best_estimator_
            self.trained_models[f"{model_name}_tuned"] = best_model
            
            logger.info(f"Best parameters for {model_name}: {search.best_params_}")
            logger.info(f"Best score for {model_name}: {search.best_score_}")
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error in hyperparameter tuning for {model_name}: {str(e)}")
            return None
    
    def get_default_param_grids(self) -> Dict[str, Dict[str, List]]:
        """
        Get default parameter grids for hyperparameter tuning.
        
        Returns:
            Dictionary of parameter grids for each model
        """
        param_grids = {
            'ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'lasso': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'elastic_net': {
                'alpha': [0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.5, 0.7, 0.9]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0]
            },
            'decision_tree': {
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance']
            },
            'svr': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
        }
        
        return param_grids
    
    def find_best_model(self, X_test: pd.DataFrame, y_test: pd.Series,
                       metric: str = 'r2') -> Tuple[str, Any, float]:
        """
        Find the best performing model based on a specific metric.
        
        Args:
            X_test: Test features
            y_test: Test target
            metric: Metric to optimize ('r2', 'rmse', 'mae')
            
        Returns:
            Tuple of (best_model_name, best_model, best_score)
        """
        if not self.trained_models:
            logger.error("No trained models found")
            return None, None, None
        
        best_model_name = None
        best_model = None
        best_score = float('-inf') if metric == 'r2' else float('inf')
        
        logger.info(f"Comparing models based on {metric}...")
        
        for model_name, model in self.trained_models.items():
            try:
                metrics = self.evaluate_model(model, X_test, y_test)
                score = metrics.get(metric, None)
                
                if score is not None:
                    if metric == 'r2':
                        if score > best_score:
                            best_score = score
                            best_model = model
                            best_model_name = model_name
                    else:  # For rmse, mae (lower is better)
                        if score < best_score:
                            best_score = score
                            best_model = model
                            best_model_name = model_name
                
                logger.info(f"{model_name} - {metric}: {score:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        self.best_model = best_model
        self.best_score = best_score
        
        logger.info(f"Best model: {best_model_name} with {metric}: {best_score:.4f}")
        return best_model_name, best_model, best_score
    
    def save_model(self, model: Any, filepath: str) -> bool:
        """
        Save a trained model to disk.
        
        Args:
            model: Trained model to save
            filepath: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            joblib.dump(model, filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, filepath: str) -> Any:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        try:
            model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None


def main():
    """
    Example usage of the ModelTrainer class.
    """
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Initialize models
    models = trainer.initialize_models()
    
    # Example: Train models (you need to provide your data)
    # X_train, X_test, y_train, y_test = your_preprocessed_data
    # trained_models = trainer.train_multiple_models(X_train, y_train)
    # best_model_name, best_model, best_score = trainer.find_best_model(X_test, y_test)
    
    print("ModelTrainer initialized successfully!")
    print(f"Available models: {list(models.keys())}")
    print("Use train_multiple_models() to train all models.")


if __name__ == "__main__":
    main()
