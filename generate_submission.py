#!/usr/bin/env python3
"""
Enhanced House Prices Competition Submission Generator
Optimized for Kaggle House Prices: Advanced Regression Techniques
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, skew
from scipy.special import boxcox1p

# Machine learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor

import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the datasets"""
    print("Loading datasets...")
    
    # Load the datasets
    train = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
    test = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
    
    print(f"Training set shape: {train.shape}")
    print(f"Test set shape: {test.shape}")
    
    # Save the 'Id' column for submission
    train_id = train['Id']
    test_id = test['Id']
    
    # Drop Id column as it's not needed for prediction
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    
    return train, test, train_id, test_id

def handle_missing_values(train, test):
    """Handle missing values based on data description"""
    print("Handling missing values...")
    
    # Fill with 'None' for these categorical features
    none_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                    'MasVnrType']

    for feature in none_features:
        train[feature].fillna('None', inplace=True)
        test[feature].fillna('None', inplace=True)

    # Fill with 0 for these numerical features
    zero_features = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
                    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
                    'BsmtHalfBath', 'MasVnrArea']

    for feature in zero_features:
        train[feature].fillna(0, inplace=True)
        test[feature].fillna(0, inplace=True)

    # Fill with mode for these features
    mode_features = ['Electrical', 'MSZoning', 'Utilities', 'Exterior1st', 
                    'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']

    for feature in mode_features:
        train[feature].fillna(train[feature].mode()[0], inplace=True)
        test[feature].fillna(test[feature].mode()[0], inplace=True)

    # For LotFrontage, fill with median by neighborhood
    train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))
    test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))
    
    print(f"Remaining missing values in train: {train.isnull().sum().sum()}")
    print(f"Remaining missing values in test: {test.isnull().sum().sum()}")
    
    return train, test

def feature_engineering(train, test):
    """Apply feature engineering"""
    print("Applying feature engineering...")
    
    # Apply log transformation to the target variable
    train["SalePrice"] = np.log1p(train["SalePrice"])
    
    # Check for skewed numerical features
    numeric_features = train.dtypes[train.dtypes != 'object'].index
    skewed_features = train[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_features})
    skewness = skewness[abs(skewness) > 0.75]
    
    print(f"Number of skewed features: {skewness.shape[0]}")
    
    # Apply Box-Cox transformation to highly skewed features
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        if feat != 'SalePrice':  # Don't transform the target again
            train[feat] = boxcox1p(train[feat], lam)
            test[feat] = boxcox1p(test[feat], lam)
    
    # Create new features
    train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
    test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']

    train['TotalBath'] = train['FullBath'] + (0.5 * train['HalfBath']) + train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath'])
    test['TotalBath'] = test['FullBath'] + (0.5 * test['HalfBath']) + test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath'])

    train['Age'] = train['YrSold'] - train['YearBuilt']
    test['Age'] = test['YrSold'] - test['YearBuilt']

    train['RemodAge'] = train['YrSold'] - train['YearRemodAdd']
    test['RemodAge'] = test['YrSold'] - test['YearRemodAdd']
    
    return train, test

def create_advanced_features(df):
    """Create advanced features for better model performance"""
    df_new = df.copy()
    
    # Quality score combinations
    df_new['OverallQual_GrLivArea'] = df_new['OverallQual'] * df_new['GrLivArea']
    df_new['OverallQual_TotalSF'] = df_new['OverallQual'] * df_new['TotalSF']
    
    # Age-related features
    df_new['HasBeenRemodeled'] = (df_new['YearRemodAdd'] != df_new['YearBuilt']).astype(int)
    df_new['RecentRemodel'] = (df_new['YrSold'] - df_new['YearRemodAdd'] <= 5).astype(int)
    
    # Garage and basement features
    df_new['HasGarage'] = (df_new['GarageArea'] > 0).astype(int)
    df_new['HasBasement'] = (df_new['TotalBsmtSF'] > 0).astype(int)
    df_new['HasPool'] = (df_new['PoolArea'] > 0).astype(int)
    df_new['HasFireplace'] = (df_new['Fireplaces'] > 0).astype(int)
    
    # Lot and exterior features
    df_new['LotArea_per_SF'] = df_new['LotArea'] / df_new['TotalSF']
    df_new['GarageArea_per_Car'] = df_new['GarageArea'] / (df_new['GarageCars'] + 1)
    
    # Neighborhood-based features (check if neighborhood columns exist)
    neighborhood_cols = [col for col in df_new.columns if col.startswith('Neighborhood_')]
    if neighborhood_cols:
        high_end_neighborhoods = ['Neighborhood_NridgHt', 'Neighborhood_StoneBr', 
                                 'Neighborhood_Timber', 'Neighborhood_Veenker']
        df_new['HighEndNeighborhood'] = 0
        for col in high_end_neighborhoods:
            if col in df_new.columns:
                df_new['HighEndNeighborhood'] += df_new[col]
    
    return df_new

def prepare_final_data(train, test):
    """Prepare final data for modeling"""
    print("Preparing final data...")
    
    # Combine train and test for one-hot encoding
    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)

    # One-hot encode categorical variables
    all_data = pd.get_dummies(all_data)

    # Split back into train and test
    train_processed = all_data[:ntrain]
    test_processed = all_data[ntrain:]

    print(f"Processed training set shape: {train_processed.shape}")
    print(f"Processed test set shape: {test_processed.shape}")
    
    return train_processed, test_processed, y_train

def evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    """Evaluate model performance"""
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    
    # Calculate RMSE
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, 
                               scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores.mean())
    
    return {
        'model': model_name,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'cv_rmse': cv_rmse,
        'model_obj': model
    }

def main():
    """Main execution function"""
    print("=" * 60)
    print("HOUSE PRICES COMPETITION - ENHANCED SUBMISSION GENERATOR")
    print("=" * 60)
    
    # Load and prepare data
    train, test, train_id, test_id = load_and_prepare_data()
    train, test = handle_missing_values(train, test)
    train, test = feature_engineering(train, test)
    
    # Prepare final data
    train_processed, test_processed, y_train = prepare_final_data(train, test)
    
    # Apply advanced feature engineering
    print("Creating advanced features...")
    train_advanced = create_advanced_features(train_processed)
    test_advanced = create_advanced_features(test_processed)
    
    print(f"Original features: {train_processed.shape[1]}")
    print(f"With advanced features: {train_advanced.shape[1]}")
    print(f"Added {train_advanced.shape[1] - train_processed.shape[1]} new features")
    
    # Split for validation
    X_train, X_val, y_train_split, y_val = train_test_split(
        train_advanced, y_train, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    test_scaled = scaler.transform(test_advanced)
    
    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    
    # Initialize models
    models = [
        LinearRegression(),
        Ridge(alpha=10),
        Lasso(alpha=0.001),
        RandomForestRegressor(n_estimators=100, random_state=42),
        XGBRegressor(n_estimators=1000, learning_rate=0.05, random_state=42),
        MLPRegressor(hidden_layer_sizes=(100, 50), early_stopping=True, random_state=42)
    ]

    model_names = ['Linear Regression', 'Ridge', 'Lasso', 'Random Forest', 'XGBoost', 'Neural Network']

    # Evaluate all models
    print("\nEvaluating models...")
    results = []
    for model, name in zip(models, model_names):
        result = evaluate_model(model, X_train_scaled, y_train_split, X_val_scaled, y_val, name)
        results.append(result)
        print(f"{name}: Validation RMSE = {result['val_rmse']:.4f}, CV RMSE = {result['cv_rmse']:.4f}")
    
    # Find best model
    best_result = min(results, key=lambda x: x['val_rmse'])
    print(f"\nBest model: {best_result['model']} with RMSE = {best_result['val_rmse']:.4f}")
    
    # Hyperparameter tuning for XGBoost
    print("\nTuning XGBoost hyperparameters...")
    xgb = XGBRegressor(random_state=42)
    
    param_grid = {
        'n_estimators': [500, 1000, 1500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
                              cv=3, scoring='neg_mean_squared_error', 
                              n_jobs=-1, verbose=0)
    
    grid_search.fit(X_train_scaled, y_train_split)
    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")
    
    # Create ensemble of best performing models
    print("\nCreating ensemble model...")
    ensemble_models = [
        ('ridge', Ridge(alpha=10)),
        ('lasso', Lasso(alpha=0.001)),
        ('xgb', XGBRegressor(**best_params, random_state=42))
    ]
    
    ensemble = VotingRegressor(estimators=ensemble_models)
    ensemble.fit(X_train_scaled, y_train_split)
    
    # Evaluate ensemble
    ensemble_pred_val = ensemble.predict(X_val_scaled)
    ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred_val))
    print(f"Ensemble Validation RMSE: {ensemble_rmse:.4f}")
    
    # Train final model on all training data
    print("\nTraining final model on all data...")
    final_ensemble = VotingRegressor(estimators=ensemble_models)
    final_ensemble.fit(X_train_scaled, y_train_split)
    
    # Make predictions on test set
    test_pred = final_ensemble.predict(test_scaled)
    test_pred = np.expm1(test_pred)  # Reverse log transformation
    
    # Create submission file
    submission = pd.DataFrame({
        'Id': test_id,
        'SalePrice': test_pred
    })
    
    # Save submission file
    submission.to_csv('final_optimized_submission.csv', index=False)
    print("Final optimized submission created!")
    
    # Show prediction statistics
    print(f"\nFinal submission statistics:")
    print(f"Min price: ${test_pred.min():,.0f}")
    print(f"Max price: ${test_pred.max():,.0f}")
    print(f"Mean price: ${test_pred.mean():,.0f}")
    print(f"Median price: ${np.median(test_pred):,.0f}")
    
    # Validate submission
    print(f"\nSubmission validation:")
    print(f"Shape: {submission.shape}")
    print(f"Missing values: {submission['SalePrice'].isnull().sum()}")
    print(f"Price range: ${test_pred.min():,.0f} - ${test_pred.max():,.0f}")
    
    print("\n" + "=" * 60)
    print("SUBMISSION GENERATION COMPLETE!")
    print("=" * 60)
    print("Files created:")
    print("- final_optimized_submission.csv (RECOMMENDED for Kaggle)")
    print("\nNext steps:")
    print("1. Submit 'final_optimized_submission.csv' to Kaggle")
    print("2. Monitor your leaderboard position")
    print("3. Consider additional feature engineering if needed")

if __name__ == "__main__":
    main()
