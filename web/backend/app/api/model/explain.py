from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import os

router = APIRouter()

# Feature descriptions for display
FEATURE_DESCRIPTIONS = {
    "games_played": "Number of games played through this week",
    
    # Overall offense/defense
    "off_ppa_cum": "Cumulative offensive EPA (Expected Points Added) per play",
    "off_success_cum": "Cumulative offensive success rate (percentage of successful plays)",
    "off_explosive_cum": "Cumulative offensive explosiveness (big play ability)",
    "def_ppa_cum": "Cumulative defensive EPA allowed per play",
    "def_success_cum": "Cumulative defensive success rate (percentage of plays stopped)",
    "def_explosive_cum": "Cumulative defensive explosiveness allowed",
    
    # Offense pass/rush splits
    "off_pass_ppa_cum": "Cumulative passing EPA per play",
    "off_pass_success_cum": "Cumulative passing success rate",
    "off_rush_ppa_cum": "Cumulative rushing EPA per play",
    "off_rush_success_cum": "Cumulative rushing success rate",
    
    # Defense pass/rush splits
    "def_pass_ppa_cum": "Cumulative passing defense EPA allowed",
    "def_pass_success_cum": "Cumulative passing defense success rate",
    "def_rush_ppa_cum": "Cumulative rushing defense EPA allowed",
    "def_rush_success_cum": "Cumulative rushing defense success rate",
    
    # Strength of schedule
    "sos_opp_rating_mean_cum": "Mean opponent rating faced (strength of schedule)",
    "sos_opp_rating_min_cum": "Weakest opponent rating faced",
    "sos_opp_rating_max_cum": "Strongest opponent rating faced",
    "sos_games_cum": "Number of games with opponent ratings",
    
    # Preseason priors
    "talent": "247 Sports composite talent rating",
    "rp_total": "Returning production percentage (total team)",
    "rp_offense": "Returning production percentage (offense)",
    "rp_defense": "Returning production percentage (defense)",
    
    # Opponent-adjusted
    "off_ppa_adj_lagged_cum": "Offensive EPA adjusted by opponent defensive strength (previous week)",
    "def_ppa_adj_lagged_cum": "Defensive EPA adjusted by opponent offensive strength (previous week)",
    
    # Resume features
    "margin_cum": "Cumulative point margin (points for - points against)",
    "wins_cum": "Cumulative wins",
    "losses_cum": "Cumulative losses",
    "win_pct_cum": "Cumulative win percentage",
    "wins_top25_cum": "Wins against top 25 teams",
    "wins_top10_cum": "Wins against top 10 teams",
    "road_wins_cum": "Road wins",
    "neutral_wins_cum": "Neutral site wins",
    "opp_rating_game_mean_cum": "Mean opponent rating at time of game",
}

FEATURE_CATEGORIES = {
    "Meta": ["games_played"],
    "Overall Offense/Defense": [
        "off_ppa_cum", "off_success_cum", "off_explosive_cum",
        "def_ppa_cum", "def_success_cum", "def_explosive_cum"
    ],
    "Offense Pass/Rush Splits": [
        "off_pass_ppa_cum", "off_pass_success_cum",
        "off_rush_ppa_cum", "off_rush_success_cum"
    ],
    "Defense Pass/Rush Splits": [
        "def_pass_ppa_cum", "def_pass_success_cum",
        "def_rush_ppa_cum", "def_rush_success_cum"
    ],
    "Strength of Schedule": [
        "sos_opp_rating_mean_cum", "sos_opp_rating_min_cum",
        "sos_opp_rating_max_cum", "sos_games_cum"
    ],
    "Preseason Priors": [
        "talent", "rp_total", "rp_offense", "rp_defense"
    ],
    "Opponent-Adjusted Stats": [
        "off_ppa_adj_lagged_cum", "def_ppa_adj_lagged_cum"
    ],
    "Résumé Features": [
        "margin_cum", "wins_cum", "losses_cum", "win_pct_cum",
        "wins_top25_cum", "wins_top10_cum", "road_wins_cum",
        "neutral_wins_cum", "opp_rating_game_mean_cum"
    ]
}

def load_model() -> XGBRegressor:
    """Load the trained XGBoost model."""
    if os.path.exists("/app"):
        # Railway deployment - /app is now the repo root
        PROJECT_ROOT = Path("/app")
    elif os.path.exists(Path(__file__).parent.parent.parent.parent.parent.parent):
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
    else:
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
    
    model_path = PROJECT_ROOT / "models" / "xgb_rating_model.json"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = XGBRegressor()
    model.load_model(str(model_path))
    return model

def load_training_data():
    """Load the training dataset."""
    if os.path.exists("/app"):
        # Railway deployment - /app is now the repo root
        PROJECT_ROOT = Path("/app")
    elif os.path.exists(Path(__file__).parent.parent.parent.parent.parent.parent):
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
    else:
        PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
    
    training_path = PROJECT_ROOT / "data" / "training_team_week_2019_2024.csv"
    
    if not training_path.exists():
        raise FileNotFoundError(f"Training data not found: {training_path}")
    
    return pd.read_csv(training_path)

def calculate_model_metrics(model: XGBRegressor, feature_names: List[str]) -> Dict[str, Any]:
    """
    Calculate model performance metrics on train/val/test splits.
    """
    try:
        df = load_training_data()
        
        # Split by season (matching training script)
        train_seasons = [2019, 2021, 2022]
        val_seasons = [2023]
        test_seasons = [2024]
        
        train = df[df["season"].isin(train_seasons)].copy()
        val = df[df["season"].isin(val_seasons)].copy()
        test = df[df["season"].isin(test_seasons)].copy()
        
        def extract_X_y(df_split: pd.DataFrame):
            missing = [c for c in feature_names if c not in df_split.columns]
            if missing:
                raise ValueError(f"Missing feature columns: {missing}")
            
            # Handle NaN values the same way as training
            X = df_split[feature_names].copy()
            
            # Fill NaN values with median (or 0 for count features) - MATCH TRAINING
            for col in X.columns:
                if X[col].isna().any():
                    # For count/cumulative features, use 0
                    if any(x in col for x in ['cum', 'games_played', 'wins', 'losses']):
                        X[col] = X[col].fillna(0.0)
                    # For percentage/rate features, use median or 0.5
                    elif 'pct' in col or 'rate' in col:
                        median_val = X[col].median()
                        X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0.5)
                    # For rating/ppa features, use median or 0
                    else:
                        median_val = X[col].median()
                        X[col] = X[col].fillna(median_val if not pd.isna(median_val) else 0.0)
            
            X = X.to_numpy(dtype=float)
            y = df_split["rating_target"].to_numpy(dtype=float)
            return X, y
        
        metrics = {}
        
        for split_name, split_df in [("train", train), ("validation", val), ("test", test)]:
            if split_df.empty:
                continue
                
            X, y = extract_X_y(split_df)
            y_pred = model.predict(X)
            
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)
            
            # Mean Absolute Error
            mae = np.mean(np.abs(y - y_pred))
            
            # Mean Error (bias)
            mean_error = np.mean(y_pred - y)
            
            metrics[split_name] = {
                "rmse": float(rmse),
                "r2": float(r2),
                "mae": float(mae),
                "mean_error": float(mean_error),
                "n_samples": int(len(y))
            }
        
        return metrics
    except Exception as e:
        raise ValueError(f"Error calculating metrics: {str(e)}")

def get_feature_names() -> List[str]:
    """Get the list of feature names in the order used by the model."""
    # This should match the FEATURE_COLS in the training script
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

@router.get("/feature-importance")
async def get_feature_importance():
    """
    Get feature importance from the trained model.
    Returns feature names and their importance scores.
    """
    try:
        model = load_model()
        feature_names = get_feature_names()
        
        if len(model.feature_importances_) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: model has {len(model.feature_importances_)} features, "
                f"but we expect {len(feature_names)}"
            )
        
        # Create list of features with importance
        features = []
        for name, importance in zip(feature_names, model.feature_importances_):
            # Find category
            category = "Other"
            for cat, feats in FEATURE_CATEGORIES.items():
                if name in feats:
                    category = cat
                    break
            
            features.append({
                "name": name,
                "importance": float(importance),
                "description": FEATURE_DESCRIPTIONS.get(name, "No description available"),
                "category": category
            })
        
        # Sort by importance
        features.sort(key=lambda x: x["importance"], reverse=True)
        
        return {
            "features": features,
            "total_features": len(features)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@router.get("/info")
async def get_model_info():
    """
    Get general information about the model.
    """
    try:
        model = load_model()
        feature_names = get_feature_names()
        
        return {
            "model_type": "XGBoost Regressor",
            "objective": "Predict linear rating system scores",
            "total_features": len(feature_names),
            "hyperparameters": {
                "max_depth": model.get_params().get("max_depth", "N/A"),
                "learning_rate": model.get_params().get("learning_rate", "N/A"),
                "n_estimators": model.get_params().get("n_estimators", "N/A"),
                "subsample": model.get_params().get("subsample", "N/A"),
                "colsample_bytree": model.get_params().get("colsample_bytree", "N/A"),
            },
            "training_seasons": [2019, 2021, 2022],
            "validation_season": 2023,
            "test_season": 2024,
            "feature_categories": list(FEATURE_CATEGORIES.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

@router.get("/metrics")
async def get_model_metrics():
    """
    Get model performance metrics (RMSE, R², MAE) for train/val/test splits.
    """
    try:
        model = load_model()
        feature_names = get_feature_names()
        
        metrics = calculate_model_metrics(model, feature_names)
        
        return {
            "metrics": metrics,
            "description": {
                "rmse": "Root Mean Squared Error - lower is better",
                "r2": "R² Score - higher is better (1.0 = perfect, 0.0 = baseline)",
                "mae": "Mean Absolute Error - lower is better",
                "mean_error": "Mean prediction error (bias) - closer to 0 is better"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating metrics: {str(e)}")