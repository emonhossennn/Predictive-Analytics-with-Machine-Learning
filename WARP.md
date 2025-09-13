# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a comprehensive machine learning project focused on predictive analytics and regression modeling. The project implements end-to-end ML workflows including data preprocessing, model training, evaluation, and prediction capabilities with support for multiple regression algorithms.

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows PowerShell
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### Running Code
```python
# Data preprocessing workflow
python -c "from src.data_preprocessing import DataPreprocessor; dp = DataPreprocessor(); print('DataPreprocessor ready')"

# Model training workflow  
python -c "from src.model_training import ModelTrainer; mt = ModelTrainer(); models = mt.initialize_models(); print(f'Models initialized: {list(models.keys())}')"

# Model evaluation
python -c "from src.model_evaluation import ModelEvaluator; me = ModelEvaluator(); print('ModelEvaluator ready')"

# Making predictions
python -c "from src.prediction import Predictor; p = Predictor(); print('Predictor ready')"
```

### Testing
```bash
# Run all tests (when implemented)
pytest tests/

# Run specific test module
pytest tests/test_data_preprocessing.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

### Code Quality
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Jupyter Development
```bash
# Launch Jupyter Lab for experimentation
jupyter lab

# Access notebooks directory
cd notebooks
```

## Core Architecture

### Configuration-Driven Design
The project uses a centralized configuration system in `config/config.py` that defines:
- **Hyperparameter grids** for all supported models (Linear, Ridge, Lasso, ElasticNet, RandomForest, GradientBoosting, DecisionTree, KNN, SVR)
- **Data processing settings** (test/validation splits, feature engineering parameters)
- **Evaluation metrics and plotting configurations**
- **Directory structure and file paths**

### Four-Module Architecture
The `src/` directory contains four main modules that form the ML pipeline:

1. **`data_preprocessing.py`**: DataPreprocessor class handling:
   - Multi-format data loading (CSV, JSON, Excel, Parquet)
   - Missing value imputation with configurable strategies
   - Categorical encoding and feature scaling
   - Automated feature engineering (polynomial features, interactions, aggregations)

2. **`model_training.py`**: ModelTrainer class providing:
   - Support for 9 different regression algorithms
   - Automated hyperparameter tuning via GridSearchCV/RandomizedSearchCV
   - Cross-validation and model comparison
   - Best model selection based on configurable metrics

3. **`model_evaluation.py`**: ModelEvaluator class offering:
   - Comprehensive regression metrics (MSE, RMSE, MAE, R², MAPE, etc.)
   - Visualization suite (predictions vs actual, residuals analysis, learning curves)
   - Multi-model comparison plotting
   - Feature importance analysis for tree-based models

4. **`prediction.py`**: Predictor class enabling:
   - Single and batch predictions
   - Confidence interval estimation for ensemble models
   - Time series forecasting capabilities
   - Feature importance analysis during prediction

### Data Flow Pattern
The typical workflow follows this pattern:
```
Raw Data → DataPreprocessor → ModelTrainer → ModelEvaluator → Predictor
```

Each module maintains its own state (scalers, encoders, trained models) and can be used independently or in sequence.

### Key Design Patterns

**Strategy Pattern**: Multiple imputation strategies, scaling methods, and evaluation metrics are configurable
**Pipeline Pattern**: Each module outputs data in the format expected by the next module
**Factory Pattern**: ModelTrainer initializes multiple model types from configuration
**State Management**: Preprocessing transformers and trained models are preserved for consistent predictions

## Important Implementation Details

### Model Training Strategy
- All models are trained with the same random_state (42) for reproducible results
- Hyperparameter tuning uses negative MSE as the default scoring metric
- Cross-validation defaults to 5 folds but is configurable
- The system automatically identifies the best model based on R² score by default

### Feature Engineering Approach
- Automatic creation of polynomial interaction features for numeric columns
- Aggregated features (mean, std, sum) across numeric columns
- Label encoding for categorical features with state preservation
- Standard scaling applied to numeric features with fit/transform pattern

### Evaluation Philosophy  
- Comprehensive metric calculation including custom residual statistics
- Visual diagnostics prioritized with residual plots, Q-Q plots, and learning curves
- Multi-model comparison through standardized metric comparison
- Feature importance available for tree-based models

### Prediction Capabilities
- Supports single predictions, batch processing, and time series forecasting
- Confidence intervals estimated using ensemble model variance or bootstrap approaches
- Feature importance analysis integrated into prediction workflow
- Automatic handling of data format conversions (DataFrame, numpy arrays, lists)

## Directory Structure Context

- `data/raw/`: Place original datasets here
- `data/processed/`: Cleaned and preprocessed data outputs  
- `data/external/`: External datasets for enrichment
- `models/`: Trained model artifacts saved as .pkl files
- `notebooks/`: Jupyter notebooks for exploration and experimentation
- `config/config.py`: Central configuration management
- `src/`: Core ML pipeline modules

## Configuration Access Patterns

Use the config module to access settings:
```python
from config.config import HYPERPARAMETER_GRIDS, get_model_params
from config.config import DEFAULT_TEST_SIZE, CROSS_VALIDATION_FOLDS

# Get hyperparameters for specific model
rf_params = get_model_params('random_forest')

# Access project paths
from config.config import get_all_paths
paths = get_all_paths()
```

The configuration system includes validation functions and can be run standalone to verify setup.
