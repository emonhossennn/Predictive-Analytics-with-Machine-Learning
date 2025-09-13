"""
Model Evaluation Module

This module contains functions for evaluating machine learning models
with various metrics and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error, mean_absolute_percentage_error
)
from sklearn.model_selection import learning_curve, validation_curve
import logging
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelEvaluator:
    """
    A comprehensive model evaluation class for regression tasks.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.evaluation_results = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'explained_variance': explained_variance_score(y_true, y_pred),
                'max_error': max_error(y_true, y_pred),
                'mape': mean_absolute_percentage_error(y_true, y_pred) * 100
            }
            
            # Additional custom metrics
            metrics['mean_residual'] = np.mean(y_true - y_pred)
            metrics['std_residual'] = np.std(y_true - y_pred)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            return {}
    
    def evaluate_single_model(self, model: Any, X_test: pd.DataFrame, 
                            y_test: pd.Series, model_name: str = "Model") -> Dict[str, Any]:
        """
        Evaluate a single model with comprehensive metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary containing evaluation results
        """
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self.calculate_metrics(y_test.values, y_pred)
            
            # Store results
            result = {
                'model_name': model_name,
                'metrics': metrics,
                'predictions': y_pred,
                'actual': y_test.values,
                'residuals': y_test.values - y_pred
            }
            
            self.evaluation_results[model_name] = result
            
            logger.info(f"Evaluation completed for {model_name}")
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
            return {}
    
    def evaluate_multiple_models(self, models: Dict[str, Any], 
                                X_test: pd.DataFrame, 
                                y_test: pd.Series) -> pd.DataFrame:
        """
        Evaluate multiple models and return comparison dataframe.
        
        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with comparison metrics
        """
        results = []
        
        for model_name, model in models.items():
            try:
                result = self.evaluate_single_model(model, X_test, y_test, model_name)
                if result:
                    metric_row = {'model': model_name}
                    metric_row.update(result['metrics'])
                    results.append(metric_row)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        if results:
            comparison_df = pd.DataFrame(results).set_index('model')
            logger.info(f"Evaluated {len(results)} models")
            return comparison_df
        else:
            logger.error("No models successfully evaluated")
            return pd.DataFrame()
    
    def plot_predictions_vs_actual(self, model_name: str, 
                                  save_path: Optional[str] = None) -> None:
        """
        Plot predictions vs actual values.
        
        Args:
            model_name: Name of the model to plot
            save_path: Path to save the plot (optional)
        """
        if model_name not in self.evaluation_results:
            logger.error(f"No evaluation results found for {model_name}")
            return
        
        result = self.evaluation_results[model_name]
        y_true = result['actual']
        y_pred = result['predictions']
        
        plt.figure(figsize=self.figsize)
        
        # Scatter plot
        plt.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Labels and title
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title(f'Predictions vs Actual - {model_name}', fontsize=14, fontweight='bold')
        plt.legend()
        
        # Add R² score to the plot
        r2 = result['metrics']['r2']
        plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
                fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_residuals(self, model_name: str, 
                      save_path: Optional[str] = None) -> None:
        """
        Plot residuals analysis.
        
        Args:
            model_name: Name of the model to plot
            save_path: Path to save the plot (optional)
        """
        if model_name not in self.evaluation_results:
            logger.error(f"No evaluation results found for {model_name}")
            return
        
        result = self.evaluation_results[model_name]
        y_pred = result['predictions']
        residuals = result['residuals']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Residuals vs Predictions
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='r', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predictions')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[0, 1].hist(residuals, bins=30, density=True, alpha=0.7, color='skyblue')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Distribution of Residuals')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot of Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Residuals vs Index (time series check)
        axes[1, 1].plot(residuals, marker='o', markersize=3, alpha=0.6)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Index')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Residuals Analysis - {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, save_path: Optional[str] = None) -> None:
        """
        Plot comparison of multiple models.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        if not self.evaluation_results:
            logger.error("No evaluation results found")
            return
        
        # Prepare data for plotting
        models = list(self.evaluation_results.keys())
        metrics_data = {
            'R²': [self.evaluation_results[model]['metrics']['r2'] for model in models],
            'RMSE': [self.evaluation_results[model]['metrics']['rmse'] for model in models],
            'MAE': [self.evaluation_results[model]['metrics']['mae'] for model in models],
            'MAPE': [self.evaluation_results[model]['metrics']['mape'] for model in models]
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # R² Score (higher is better)
        axes[0, 0].bar(models, metrics_data['R²'], color='skyblue')
        axes[0, 0].set_title('R² Score (Higher is Better)', fontweight='bold')
        axes[0, 0].set_ylabel('R² Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # RMSE (lower is better)
        axes[0, 1].bar(models, metrics_data['RMSE'], color='lightcoral')
        axes[0, 1].set_title('Root Mean Squared Error (Lower is Better)', fontweight='bold')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # MAE (lower is better)
        axes[1, 0].bar(models, metrics_data['MAE'], color='lightgreen')
        axes[1, 0].set_title('Mean Absolute Error (Lower is Better)', fontweight='bold')
        axes[1, 0].set_ylabel('MAE')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(axis='y', alpha=0.3)
        
        # MAPE (lower is better)
        axes[1, 1].bar(models, metrics_data['MAPE'], color='gold')
        axes[1, 1].set_title('Mean Absolute Percentage Error (Lower is Better)', fontweight='bold')
        axes[1, 1].set_ylabel('MAPE (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        plt.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()
    
    def plot_learning_curve(self, model: Any, X: pd.DataFrame, y: pd.Series,
                          model_name: str = "Model", cv: int = 5,
                          save_path: Optional[str] = None) -> None:
        """
        Plot learning curve for a model.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            model_name: Name of the model
            cv: Number of cross-validation folds
            save_path: Path to save the plot (optional)
        """
        try:
            train_sizes, train_scores, val_scores = learning_curve(
                model, X, y, cv=cv, n_jobs=-1, 
                train_sizes=np.linspace(0.1, 1.0, 10),
                scoring='r2'
            )
            
            train_mean = np.mean(train_scores, axis=1)
            train_std = np.std(train_scores, axis=1)
            val_mean = np.mean(val_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            plt.figure(figsize=self.figsize)
            
            plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
            plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                           alpha=0.1, color='blue')
            
            plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
            plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                           alpha=0.1, color='red')
            
            plt.xlabel('Training Set Size')
            plt.ylabel('R² Score')
            plt.title(f'Learning Curve - {model_name}', fontweight='bold')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting learning curve: {str(e)}")
    
    def feature_importance_plot(self, model: Any, feature_names: List[str],
                              model_name: str = "Model", top_n: int = 20,
                              save_path: Optional[str] = None) -> None:
        """
        Plot feature importance (for models that support it).
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            model_name: Name of the model
            top_n: Number of top features to show
            save_path: Path to save the plot (optional)
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:top_n]
                
                plt.figure(figsize=self.figsize)
                plt.bar(range(len(indices)), importances[indices])
                plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
                plt.xlabel('Features')
                plt.ylabel('Importance')
                plt.title(f'Feature Importance - {model_name}', fontweight='bold')
                plt.tight_layout()
                
                if save_path:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight')
                    logger.info(f"Plot saved to {save_path}")
                
                plt.show()
            else:
                logger.warning(f"Model {model_name} doesn't support feature importance")
                
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
    
    def generate_evaluation_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            save_path: Path to save the report (optional)
            
        Returns:
            Evaluation report as string
        """
        if not self.evaluation_results:
            logger.error("No evaluation results found")
            return ""
        
        report = "=== MODEL EVALUATION REPORT ===\n\n"
        
        for model_name, result in self.evaluation_results.items():
            report += f"Model: {model_name}\n"
            report += "-" * 30 + "\n"
            
            metrics = result['metrics']
            report += f"R² Score: {metrics['r2']:.4f}\n"
            report += f"RMSE: {metrics['rmse']:.4f}\n"
            report += f"MAE: {metrics['mae']:.4f}\n"
            report += f"MAPE: {metrics['mape']:.2f}%\n"
            report += f"Max Error: {metrics['max_error']:.4f}\n"
            report += f"Explained Variance: {metrics['explained_variance']:.4f}\n"
            report += f"Mean Residual: {metrics['mean_residual']:.4f}\n"
            report += f"Std Residual: {metrics['std_residual']:.4f}\n"
            report += "\n"
        
        # Best model summary
        best_r2_model = max(self.evaluation_results.keys(), 
                           key=lambda x: self.evaluation_results[x]['metrics']['r2'])
        best_rmse_model = min(self.evaluation_results.keys(), 
                            key=lambda x: self.evaluation_results[x]['metrics']['rmse'])
        
        report += "=== BEST MODELS ===\n"
        report += f"Best R² Score: {best_r2_model} ({self.evaluation_results[best_r2_model]['metrics']['r2']:.4f})\n"
        report += f"Best RMSE: {best_rmse_model} ({self.evaluation_results[best_rmse_model]['metrics']['rmse']:.4f})\n"
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        
        return report


def main():
    """
    Example usage of the ModelEvaluator class.
    """
    evaluator = ModelEvaluator()
    
    print("ModelEvaluator initialized successfully!")
    print("Available methods:")
    print("- evaluate_single_model()")
    print("- evaluate_multiple_models()")
    print("- plot_predictions_vs_actual()")
    print("- plot_residuals()")
    print("- plot_model_comparison()")
    print("- generate_evaluation_report()")


if __name__ == "__main__":
    main()
