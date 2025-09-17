#!/usr/bin/env python3
"""
Quick optimized submission using the best performing model (Lasso)
"""

import pandas as pd
import numpy as np
from scipy.stats import skew
from scipy.special import boxcox1p
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

def main():
    print("Creating optimized submission...")
    
    # Load data
    train = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
    test = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
    
    # Save IDs
    test_id = test['Id']
    
    # Drop IDs
    train.drop('Id', axis=1, inplace=True)
    test.drop('Id', axis=1, inplace=True)
    
    # Handle missing values
    none_features = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 
                    'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                    'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                    'MasVnrType']
    
    for feature in none_features:
        train[feature].fillna('None', inplace=True)
        test[feature].fillna('None', inplace=True)
    
    zero_features = ['GarageYrBlt', 'GarageArea', 'GarageCars', 'BsmtFinSF1', 
                    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 
                    'BsmtHalfBath', 'MasVnrArea']
    
    for feature in zero_features:
        train[feature].fillna(0, inplace=True)
        test[feature].fillna(0, inplace=True)
    
    mode_features = ['Electrical', 'MSZoning', 'Utilities', 'Exterior1st', 
                    'Exterior2nd', 'KitchenQual', 'Functional', 'SaleType']
    
    for feature in mode_features:
        train[feature].fillna(train[feature].mode()[0], inplace=True)
        test[feature].fillna(test[feature].mode()[0], inplace=True)
    
    # LotFrontage by neighborhood
    train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))
    test['LotFrontage'] = test.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))
    
    # Log transform target
    train["SalePrice"] = np.log1p(train["SalePrice"])
    
    # Box-Cox transform skewed features
    numeric_features = train.dtypes[train.dtypes != 'object'].index
    skewed_features = train[numeric_features].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew': skewed_features})
    skewness = skewness[abs(skewness) > 0.75]
    
    skewed_features = skewness.index
    lam = 0.15
    for feat in skewed_features:
        if feat != 'SalePrice':
            train[feat] = boxcox1p(train[feat], lam)
            test[feat] = boxcox1p(test[feat], lam)
    
    # Feature engineering
    train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
    test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']
    
    train['TotalBath'] = train['FullBath'] + (0.5 * train['HalfBath']) + train['BsmtFullBath'] + (0.5 * train['BsmtHalfBath'])
    test['TotalBath'] = test['FullBath'] + (0.5 * test['HalfBath']) + test['BsmtFullBath'] + (0.5 * test['BsmtHalfBath'])
    
    train['Age'] = train['YrSold'] - train['YearBuilt']
    test['Age'] = test['YrSold'] - test['YearBuilt']
    
    train['RemodAge'] = train['YrSold'] - train['YearRemodAdd']
    test['RemodAge'] = test['YrSold'] - test['YearRemodAdd']
    
    # Advanced features
    train['OverallQual_GrLivArea'] = train['OverallQual'] * train['GrLivArea']
    test['OverallQual_GrLivArea'] = test['OverallQual'] * test['GrLivArea']
    
    train['HasGarage'] = (train['GarageArea'] > 0).astype(int)
    test['HasGarage'] = (test['GarageArea'] > 0).astype(int)
    
    train['HasBasement'] = (train['TotalBsmtSF'] > 0).astype(int)
    test['HasBasement'] = (test['TotalBsmtSF'] > 0).astype(int)
    
    # One-hot encoding
    ntrain = train.shape[0]
    ntest = test.shape[0]
    y_train = train.SalePrice.values
    all_data = pd.concat((train, test)).reset_index(drop=True)
    all_data.drop(['SalePrice'], axis=1, inplace=True)
    all_data = pd.get_dummies(all_data)
    
    train_processed = all_data[:ntrain]
    test_processed = all_data[ntrain:]
    
    # Split and scale
    X_train, X_val, y_train_split, y_val = train_test_split(
        train_processed, y_train, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    test_scaled = scaler.transform(test_processed)
    
    # Create ensemble of best models
    ensemble = VotingRegressor([
        ('lasso', Lasso(alpha=0.001)),
        ('ridge', Ridge(alpha=10))
    ])
    
    ensemble.fit(X_train_scaled, y_train_split)
    
    # Evaluate
    val_pred = ensemble.predict(X_val_scaled)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Validation RMSE: {rmse:.4f}")
    
    # Final predictions
    test_pred = ensemble.predict(test_scaled)
    test_pred = np.expm1(test_pred)
    
    # Create submission
    submission = pd.DataFrame({
        'Id': test_id,
        'SalePrice': test_pred
    })
    
    submission.to_csv('optimized_submission.csv', index=False)
    print("Optimized submission created!")
    print(f"Price range: ${test_pred.min():,.0f} - ${test_pred.max():,.0f}")
    print(f"Mean price: ${test_pred.mean():,.0f}")

if __name__ == "__main__":
    main()
