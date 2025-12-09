#!/usr/bin/env python
"""
Diagnostic script to identify why model performance is poor.
Checks for feature mismatches, NaN handling, and data consistency.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def load_model():
    """Load the trained model."""
    model_path = Path("models/xgb_rating_model.json")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    model = XGBRegressor()
    model.load_model(str(model_path))
    return model

def load_training_data():
    """Load training data."""
    data_path = Path("data/training_team_week_2019_2024.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Training data not found: {data_path}")
    
    return pd.read_csv(data_path)

def get_feature_names():
    """Get expected feature names."""
    return [
        "games_played",
        "off_ppa_cum", "off_success_cum", "off_explosive_cum",
        "def_ppa_cum", "def_success_cum", "def_explosive_cum",
        "off_pass_ppa_cum", "off_pass_success_cum",
        "off_rush_ppa_cum", "off_rush_success_cum",
        "def_pass_ppa_cum", "def_pass_success_cum",
        "def_rush_ppa_cum", "def_rush_success_cum",
        "sos_opp_rating_mean_cum", "sos_opp_rating_min_cum",
        "sos_opp_rating_max_cum", "sos_games_cum",
        "talent", "rp_total", "rp_offense", "rp_defense",
        "off_ppa_adj_lagged_cum", "def_ppa_adj_lagged_cum",
        "margin_cum", "wins_cum", "losses_cum", "win_pct_cum",
        "wins_top25_cum", "wins_top10_cum", "road_wins_cum",
        "neutral_wins_cum", "opp_rating_game_mean_cum",
    ]

def main():
    print("="*70)
    print("MODEL PERFORMANCE DIAGNOSTIC")
    print("="*70)
    
    # Load data
    print("\n1. Loading data...")
    df = load_training_data()
    model = load_model()
    feature_names = get_feature_names()
    
    print(f"   Training data shape: {df.shape}")
    print(f"   Model expects {len(feature_names)} features")
    print(f"   Model has {len(model.feature_importances_)} features")
    
    # Check feature existence
    print("\n2. Checking feature columns...")
    missing_features = [f for f in feature_names if f not in df.columns]
    if missing_features:
        print(f"   ❌ MISSING FEATURES: {missing_features}")
    else:
        print("   ✅ All features present")
    
    # Check for NaN values
    print("\n3. Checking for NaN values in features...")
    feature_df = df[feature_names]
    nan_counts = feature_df.isna().sum()
    nan_features = nan_counts[nan_counts > 0]
    
    if len(nan_features) > 0:
        print(f"   ⚠️  Features with NaN values:")
        for feat, count in nan_features.items():
            pct = (count / len(df)) * 100
            print(f"      {feat}: {count} ({pct:.1f}%)")
    else:
        print("   ✅ No NaN values in features")
    
    # Check target variable
    print("\n4. Checking target variable...")
    if "rating_target" not in df.columns:
        print("   ❌ rating_target column missing!")
    else:
        target = df["rating_target"]
        print(f"   Target stats:")
        print(f"      Mean: {target.mean():.3f}")
        print(f"      Std: {target.std():.3f}")
        print(f"      Min: {target.min():.3f}")
        print(f"      Max: {target.max():.3f}")
        print(f"      NaN count: {target.isna().sum()}")
    
    # Split data
    print("\n5. Splitting data...")
    train_seasons = [2019, 2021, 2022]
    val_seasons = [2023]
    test_seasons = [2024]
    
    train = df[df["season"].isin(train_seasons)].copy()
    val = df[df["season"].isin(val_seasons)].copy()
    test = df[df["season"].isin(test_seasons)].copy()
    
    print(f"   Train: {len(train)} samples")
    print(f"   Val: {len(val)} samples")
    print(f"   Test: {len(test)} samples")
    
    # Prepare features (handle NaNs like XGBoost does - fill with 0 or median)
    print("\n6. Preparing features (filling NaNs)...")
    
    def prepare_X(df_split):
        X = df_split[feature_names].copy()
        # Fill NaNs with median (XGBoost default behavior)
        for col in X.columns:
            if X[col].isna().any():
                median_val = X[col].median()
                if pd.isna(median_val):
                    X[col] = X[col].fillna(0.0)
                else:
                    X[col] = X[col].fillna(median_val)
        return X.to_numpy(dtype=float)
    
    X_train = prepare_X(train)
    X_val = prepare_X(val)
    X_test = prepare_X(test)
    
    y_train = train["rating_target"].to_numpy(dtype=float)
    y_val = val["rating_target"].to_numpy(dtype=float)
    y_test = test["rating_target"].to_numpy(dtype=float)
    
    # Evaluate model
    print("\n7. Evaluating model...")
    
    for split_name, X, y in [("Train", X_train, y_train), 
                             ("Val", X_val, y_val), 
                             ("Test", X_test, y_test)]:
        y_pred = model.predict(X)
        
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        mae = np.mean(np.abs(y - y_pred))
        mean_error = np.mean(y_pred - y)
        
        # Baseline (predicting mean)
        baseline_pred = np.full_like(y, y.mean())
        baseline_rmse = np.sqrt(mean_squared_error(y, baseline_pred))
        baseline_r2 = r2_score(y, baseline_pred)
        
        print(f"\n   {split_name} Set:")
        print(f"      RMSE: {rmse:.3f} (baseline: {baseline_rmse:.3f})")
        print(f"      R²: {r2:.3f} (baseline: {baseline_r2:.3f})")
        print(f"      MAE: {mae:.3f}")
        print(f"      Mean Error (bias): {mean_error:.3f}")
        
        if r2 < 0:
            print(f"      ⚠️  Negative R² means model is worse than baseline!")
    
    # Check prediction distribution
    print("\n8. Checking prediction distribution...")
    y_pred_all = model.predict(X_train)
    print(f"   Train predictions:")
    print(f"      Mean: {y_pred_all.mean():.3f}")
    print(f"      Std: {y_pred_all.std():.3f}")
    print(f"      Min: {y_pred_all.min():.3f}")
    print(f"      Max: {y_pred_all.max():.3f}")
    
    print(f"\n   Train targets:")
    print(f"      Mean: {y_train.mean():.3f}")
    print(f"      Std: {y_train.std():.3f}")
    print(f"      Min: {y_train.min():.3f}")
    print(f"      Max: {y_train.max():.3f}")
    
    # Check if model was trained on same data
    print("\n9. Recommendations:")
    print("   - If R² is negative, the model may need retraining")
    print("   - Check if NaN handling matches training time")
    print("   - Verify model was trained on same feature set")
    print("   - Consider retraining with updated data")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()