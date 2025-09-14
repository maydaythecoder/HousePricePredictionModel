# House Prices: Advanced Regression Techniques

This project implements a machine learning pipeline to predict house prices using the Kaggle House Prices dataset. The solution includes comprehensive data exploration, feature engineering, and model comparison with hyperparameter tuning.

## Overview

The goal is to predict house sale prices based on various features including lot size, house characteristics, location, and quality ratings. This is a regression problem where we aim to minimize the Root Mean Square Error (RMSE) between predicted and actual prices.

## Dataset

- **Training Data**: 1,460 houses with 79 explanatory variables
- **Test Data**: 1,459 houses for prediction submission
- **Target Variable**: SalePrice (house sale prices)
- **Source**: [Kaggle House Prices Competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

## Project Structure

``` text
Houses/
├── README.md
├── Houses.ipynb                                    # Jupyter notebook analysis
├── submission.csv                                  # Final predictions
├── anaconda_projects/                              # Anaconda project files
│   └── db/
│       └── project_filebrowser.db                 # Project database
└── house-prices-advanced-regression-techniques/   # Kaggle dataset
    ├── data_description.txt                        # Feature descriptions
    ├── sample_submission.csv                       # Submission format example
    ├── test.csv                                   # Test dataset
    └── train.csv                                  # Training dataset
```

## Installation & Setup

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn xgboost
```

### Kaggle API Authentication

1. Download your `kaggle.json` from <https://www.kaggle.com/account>
2. Place it in `~/.kaggle/kaggle.json`
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Download Dataset

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques
unzip house-prices-advanced-regression-techniques.zip
```

## Analysis Pipeline

### 1. Data Exploration & Visualization

- **Target Analysis**: Distribution analysis with log transformation
- **Correlation Matrix**: Identify highly correlated numerical features
- **Missing Data**: Comprehensive missing value analysis and visualization
- **Feature Distributions**: Skewness analysis for numerical features

### 2. Data Preprocessing

#### Missing Value Handling Strategy

- **None Strategy**: Categorical features where missing = "None" (e.g., PoolQC, Fence)
- **Zero Strategy**: Numerical features where missing = 0 (e.g., GarageArea, BsmtFinSF1)
- **Mode Strategy**: Categorical features filled with most frequent value
- **Neighborhood-based**: LotFrontage filled using neighborhood median

#### Feature Engineering

- **Log Transformation**: Applied to target variable (SalePrice) for normality
- **Box-Cox Transformation**: Applied to skewed features (skewness > 0.75)
- **New Features Created**:
  - `TotalSF`: Total square footage (basement + 1st + 2nd floor)
  - `TotalBath`: Total bathroom count (full + 0.5*half baths)
  - `Age`: House age at time of sale
  - `RemodAge`: Years since last remodel

### 3. Model Implementation

#### Models Tested

1. **Linear Regression**: Baseline model
2. **Ridge Regression**: L2 regularization (α=10)
3. **Lasso Regression**: L1 regularization (α=0.001)
4. **Random Forest**: Ensemble method (100 trees)
5. **XGBoost**: Gradient boosting (1000 estimators, lr=0.05)
6. **Neural Network**: Multi-layer perceptron (100, 50 hidden units)

#### Evaluation Metrics

- Training RMSE
- Validation RMSE (20% holdout)
- 5-Fold Cross-Validation RMSE

### 4. Hyperparameter Tuning

**XGBoost Grid Search Parameters:**

- `n_estimators`: [500, 1000, 1500]
- `learning_rate`: [0.01, 0.05, 0.1]
- `max_depth`: [3, 4, 5]
- `subsample`: [0.8, 0.9, 1.0]
- `colsample_bytree`: [0.8, 0.9, 1.0]

## Key Features

### Data Quality

- ✅ Zero missing values after preprocessing
- ✅ Normalized target variable distribution
- ✅ Handled feature skewness with Box-Cox transformation
- ✅ One-hot encoded categorical variables

### Model Performance

- 📊 Comprehensive model comparison
- 🎯 Cross-validation for robust evaluation
- 🔧 Hyperparameter optimization
- 📈 Feature importance analysis

### Visualizations

- Distribution plots with statistical fitting
- Correlation heatmaps
- Missing data analysis charts
- Model performance comparisons
- Top feature importance rankings

## Usage

1 **Run the complete analysis:**

 ```python
 python house_prices_analysis.py
 ```

2 **Key outputs:**

- Model performance comparison
- Optimized hyperparameters
- `submission.csv` for Kaggle submission
- Feature importance rankings

## Results

The pipeline automatically:

- Compares 6 different models
- Identifies the best performing model
- Generates optimized predictions
- Creates submission file
- Provides feature importance insights

## Model Insights

### Most Important Features (typical)

- OverallQual: Overall material and finish quality
- GrLivArea: Above ground living area
- TotalSF: Total square footage (engineered feature)
- GarageCars: Size of garage in car capacity
- ExterQual: Exterior material quality

### Engineering Impact

Custom features like `TotalSF` and `TotalBath` often rank among top predictors, demonstrating the value of domain knowledge in feature engineering.

## Next Steps

- **Feature Selection**: Implement recursive feature elimination
- **Advanced Models**: Try ensemble methods (stacking, blending)
- **External Data**: Incorporate neighborhood economic indicators
- **Time Series**: Analyze seasonal price trends

## Notes

- All predictions are inverse log-transformed for submission
- Cross-validation prevents overfitting
- Feature scaling applied for neural networks
- Random state set for reproducibility

---

**Competition**: [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)

**Evaluation Metric**: Root Mean Squared Error (RMSE) between predicted and actual log sale prices
