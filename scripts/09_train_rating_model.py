import os

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


def load_training_data(path: str = "data/training_team_week_2019_2024.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print("Full training data shape:", df.shape)
    print(df.head())
    return df


def make_splits(df: pd.DataFrame):
    """
    Split by season to mimic real-world generalization.

    Train on: 2019, 2021, 2022
    Val on:   2023
    Test on:  2024
    """
    train_seasons = [2019, 2021, 2022]
    val_seasons = [2023]
    test_seasons = [2024]

    train = df[df["season"].isin(train_seasons)].copy()
    val = df[df["season"].isin(val_seasons)].copy()
    test = df[df["season"].isin(test_seasons)].copy()

    print("\nSplit sizes (rows):")
    print("  Train:", train.shape[0])
    print("  Val:  ", val.shape[0])
    print("  Test: ", test.shape[0])

    return train, val, test


def get_feature_and_target_matrices(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """
    Build X (features) and y (target) for each split.
    """

    feature_cols = [
        # Meta / usage
        "games_played",

        # Overall offense / defense
        "off_ppa_cum",
        "off_success_cum",
        "off_explosive_cum",
        "def_ppa_cum",
        "def_success_cum",
        "def_explosive_cum",

        # Offense pass/rush splits
        "off_pass_ppa_cum",
        "off_pass_success_cum",
        "off_rush_ppa_cum",
        "off_rush_success_cum",

        # Defense pass/rush splits
        "def_pass_ppa_cum",
        "def_pass_success_cum",
        "def_rush_ppa_cum",
        "def_rush_success_cum",

        # Strength-of-schedule (rating-based)
        "sos_opp_rating_mean_cum",
        "sos_opp_rating_min_cum",
        "sos_opp_rating_max_cum",
        "sos_games_cum",

        # Preseason priors
        "talent",
        "rp_total",
        "rp_offense",
        "rp_defense",

        # Opponent-adjusted EPA (lagged)
        "off_ppa_adj_lagged_cum",
        "def_ppa_adj_lagged_cum",

        # 🔥 New résumé / margin features
        "margin_cum",
        "wins_cum",
        "losses_cum",
        "win_pct_cum",
        "wins_top25_cum",
        "wins_top10_cum",
        "road_wins_cum",
        "neutral_wins_cum",
        "opp_rating_game_mean_cum",
    ]

    print("\nUsing feature columns:", feature_cols)

    def extract_X_y(df: pd.DataFrame):
        # Ensure all columns exist (defensive programming)
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing expected feature columns in DF: {missing}")

        # Handle NaN values BEFORE converting to numpy
        # XGBoost handles NaNs, but we should be explicit for consistency
        X = df[feature_cols].copy()
        
        # Fill NaN values with median (or 0 for count features)
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
        y = df["rating_target"].to_numpy(dtype=float)
        return X, y

    X_train, y_train = extract_X_y(train)
    X_val, y_val = extract_X_y(val)
    X_test, y_test = extract_X_y(test)

    print("X_train shape:", X_train.shape)
    print("X_val shape:  ", X_val.shape)
    print("X_test shape: ", X_test.shape)
    
    # Check for any remaining NaN or inf values
    print("\nChecking for NaN/Inf in features...")
    for name, X in [("Train", X_train), ("Val", X_val), ("Test", X_test)]:
        nan_count = np.isnan(X).sum()
        inf_count = np.isinf(X).sum()
        if nan_count > 0:
            print(f"  {name}: {nan_count} NaN values found!")
        if inf_count > 0:
            print(f"  {name}: {inf_count} Inf values found!")
        if nan_count == 0 and inf_count == 0:
            print(f"  {name}: ✅ No NaN/Inf values")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols


def train_xgb_model(X_train, y_train, X_val, y_val) -> XGBRegressor:
    """
    Train an XGBoost regressor to predict rating_linear from features.
    """
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1,
        max_depth=5,
        min_child_weight=3,
        learning_rate=0.05,
        n_estimators=600,
        subsample=0.7,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        reg_alpha=0.5,
        random_state=42,
    )


    print("\nTraining XGBoost model...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    return model



def evaluate_model(model: XGBRegressor, X, y, split_name: str):
    """
    Compute RMSE and R^2 for a given split.
    """
    y_pred = model.predict(X)

    # Older sklearn doesn't support squared=False argument
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, y_pred)

    print(f"\n[{split_name}] performance:")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R^2:  {r2:.3f}")

    return rmse, r2


def show_feature_importance(model: XGBRegressor, feature_names):
    print("\nFeature importance (gain-based):")
    importances = model.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        print(f"  {name:25s} {imp:.3f}")


def save_model(model: XGBRegressor, path: str = "models/xgb_rating_model.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save_model(path)
    print(f"\nSaved XGBoost model to {path}")


def main():
    df = load_training_data()
    train, val, test = make_splits(df)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_cols = get_feature_and_target_matrices(train, val, test)

    # 👇 pass X_val, y_val too
    model = train_xgb_model(X_train, y_train, X_val, y_val)

    # Evaluate
    evaluate_model(model, X_train, y_train, "Train")
    evaluate_model(model, X_val, y_val, "Val")
    evaluate_model(model, X_test, y_test, "Test")

    # Inspect feature importance
    show_feature_importance(model, feature_cols)

    # Save model for later use
    save_model(model)


if __name__ == "__main__":
    main()
