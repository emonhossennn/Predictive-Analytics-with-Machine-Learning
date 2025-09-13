"""
Configuration Module

This module contains all configuration settings, hyperparameters,
and constants used throughout the predictive analytics project.
"""

import os
from typing import Dict, Any, List

# ============================================================================
# PROJECT SETTINGS
# ============================================================================

# Project information
PROJECT_NAME = "Predictive Analytics with Machine Learning"
VERSION = "1.0.0"
AUTHOR = "Your Name"
DESCRIPTION = "A comprehensive ML project for predictive analytics and forecasting"

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
EXTERNAL_DATA_DIR = os.path.join(DATA_DIR, "external")
MODELS_DIR = os.path.join(BASE_DIR, "models")
NOTEBOOKS_DIR = os.path.join(BASE_DIR, "notebooks")
SRC_DIR = os.path.join(BASE_DIR, "src")
TESTS_DIR = os.path.join(BASE_DIR, "tests")

# Ensure directories exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
                 EXTERNAL_DATA_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# DATA SETTINGS
# ============================================================================

# Data loading settings
SUPPORTED_FORMATS = ['.csv', '.json', '.xlsx', '.parquet']
DEFAULT_ENCODING = 'utf-8'
DEFAULT_SEPARATOR = ','

# Data validation settings
MAX_MISSING_PERCENTAGE = 0.3  # 30% missing values threshold
MIN_SAMPLES = 50  # Minimum number of samples required
MAX_FEATURES = 1000  # Maximum number of features

# Data preprocessing settings
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VALIDATION_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VALIDATION_FOLDS = 5

# Feature engineering settings
FEATURE_SELECTION_K = 20  # Number of top features to select
POLYNOMIAL_DEGREE = 2
INTERACTION_THRESHOLD = 0.1  # Minimum correlation for interaction features

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Training settings
N_JOBS = -1  # Use all available cores
VERBOSE = 1  # Verbosity level

# Model evaluation settings
SCORING_METRICS = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
PRIMARY_METRIC = 'r2'

# Model selection settings
MODEL_SELECTION_CV = 5
HYPERPARAMETER_SEARCH_CV = 3
HYPERPARAMETER_SEARCH_ITER = 100  # For RandomizedSearchCV

# ============================================================================
# HYPERPARAMETER GRIDS
# ============================================================================

# Linear models
LINEAR_REGRESSION_PARAMS = {}

RIDGE_PARAMS = {
    'alpha': [0.1, 1.0, 10.0, 100.0, 1000.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg']
}

LASSO_PARAMS = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'max_iter': [1000, 2000, 5000]
}

ELASTIC_NET_PARAMS = {
    'alpha': [0.01, 0.1, 1.0, 10.0],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    'max_iter': [1000, 2000, 5000]
}

# Tree-based models
DECISION_TREE_PARAMS = {
    'max_depth': [None, 5, 10, 15, 20, 25],
    'min_samples_split': [2, 5, 10, 15],
    'min_samples_leaf': [1, 2, 4, 6],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error']
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None, 0.3, 0.5],
    'bootstrap': [True, False]
}

GRADIENT_BOOSTING_PARAMS = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'max_depth': [3, 4, 5, 6, 7],
    'subsample': [0.8, 0.9, 1.0],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Other models
KNN_PARAMS = {
    'n_neighbors': [3, 5, 7, 9, 11, 15, 20],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
}

SVR_PARAMS = {
    'C': [0.01, 0.1, 1, 10, 100, 1000],
    'epsilon': [0.01, 0.1, 0.2, 0.5],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
}

# Combined parameter grids
HYPERPARAMETER_GRIDS = {
    'linear_regression': LINEAR_REGRESSION_PARAMS,
    'ridge': RIDGE_PARAMS,
    'lasso': LASSO_PARAMS,
    'elastic_net': ELASTIC_NET_PARAMS,
    'decision_tree': DECISION_TREE_PARAMS,
    'random_forest': RANDOM_FOREST_PARAMS,
    'gradient_boosting': GRADIENT_BOOSTING_PARAMS,
    'knn': KNN_PARAMS,
    'svr': SVR_PARAMS
}

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

# Plotting settings
FIGURE_SIZE = (12, 8)
DPI = 300
PLOT_STYLE = 'seaborn-v0_8'
COLOR_PALETTE = 'husl'

# Report settings
DECIMAL_PLACES = 4
TOP_FEATURES_TO_SHOW = 20

# Validation settings
CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_SAMPLES = 1000

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# Log file settings
LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'predictive_analytics.log')

# ============================================================================
# PREDICTION SETTINGS
# ============================================================================

# Batch prediction settings
BATCH_SIZE = 1000
MAX_PREDICTION_SIZE = 100000

# Time series settings
DEFAULT_FORECAST_PERIODS = 30
TIME_SERIES_FREQUENCY = 'D'  # Daily frequency

# Confidence interval settings
PREDICTION_INTERVALS = [0.80, 0.90, 0.95, 0.99]

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Memory settings
MAX_MEMORY_USAGE = '4GB'
CHUNK_SIZE = 10000  # For processing large datasets

# Parallel processing
MAX_WORKERS = None  # Use all available cores
BACKEND = 'loky'  # Joblib backend

# ============================================================================
# ENVIRONMENT SETTINGS
# ============================================================================

# Environment variables
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')
DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'

# Database settings (if needed)
DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///predictive_analytics.db')

# API settings (if needed)
API_KEY = os.getenv('API_KEY', '')
API_BASE_URL = os.getenv('API_BASE_URL', 'http://localhost:8000')

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_params(model_name: str) -> Dict[str, Any]:
    """
    Get hyperparameter grid for a specific model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Parameter grid dictionary
    """
    return HYPERPARAMETER_GRIDS.get(model_name, {})

def get_all_paths() -> Dict[str, str]:
    """
    Get all project paths.
    
    Returns:
        Dictionary of project paths
    """
    return {
        'base': BASE_DIR,
        'data': DATA_DIR,
        'raw_data': RAW_DATA_DIR,
        'processed_data': PROCESSED_DATA_DIR,
        'external_data': EXTERNAL_DATA_DIR,
        'models': MODELS_DIR,
        'notebooks': NOTEBOOKS_DIR,
        'src': SRC_DIR,
        'tests': TESTS_DIR,
        'logs': LOG_DIR
    }

def get_project_info() -> Dict[str, str]:
    """
    Get project information.
    
    Returns:
        Dictionary with project metadata
    """
    return {
        'name': PROJECT_NAME,
        'version': VERSION,
        'author': AUTHOR,
        'description': DESCRIPTION,
        'environment': ENVIRONMENT
    }

def validate_config() -> bool:
    """
    Validate configuration settings.
    
    Returns:
        True if configuration is valid
    """
    try:
        # Check if required directories exist
        required_dirs = [DATA_DIR, MODELS_DIR]
        for directory in required_dirs:
            if not os.path.exists(directory):
                raise ValueError(f"Required directory does not exist: {directory}")
        
        # Check if test size is valid
        if not 0 < DEFAULT_TEST_SIZE < 1:
            raise ValueError("DEFAULT_TEST_SIZE must be between 0 and 1")
        
        # Check if cross-validation folds is valid
        if CROSS_VALIDATION_FOLDS < 2:
            raise ValueError("CROSS_VALIDATION_FOLDS must be at least 2")
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {str(e)}")
        return False

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

if __name__ == "__main__":
    # Validate configuration on import
    if validate_config():
        print("Configuration loaded and validated successfully!")
        print(f"Project: {PROJECT_NAME} v{VERSION}")
        print(f"Environment: {ENVIRONMENT}")
        print(f"Base directory: {BASE_DIR}")
    else:
        print("Configuration validation failed!")
