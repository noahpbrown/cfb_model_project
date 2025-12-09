import os
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import train_test_split


def load_training():
    df = pd.read_csv("data/training_team_week_2019_2024.csv")
    print("Loaded training data:", df.shape)

    feature_cols = [
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
        "off_ppa_adj_lagged_cum", "def_ppa_adj_lagged_cum"
    ]

    X = df[feature_cols].fillna(0).to_numpy()
    y = df["rating_linear"].to_numpy()
    return X, y, feature_cols


def main():
    X, y, feature_cols = load_training()

    # Split into train/holdout for cross-validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=-1
    )

    # Hyperparameter ranges
    param_dist = {
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "n_estimators": [300, 500, 700, 900],
        "min_child_weight": [1, 2, 3, 5, 7],
        "reg_lambda": [0.5, 1.0, 1.5, 2.0],
        "reg_alpha": [0.0, 0.1, 0.3, 0.5],
    }

    scorer = make_scorer(r2_score)

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=40,         # try 40 combinations
        scoring=scorer,
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42,
    )

    print("\nRunning hyperparameter search...")
    random_search.fit(X_train, y_train)

    print("\nBest hyperparameters:")
    print(random_search.best_params_)

    best_model = random_search.best_estimator_

    # Evaluate final model
    preds = best_model.predict(X_test)
    r2 = r2_score(y_test, preds)
    print("\nFinal R² on holdout:", r2)

    # Save tuned model
    os.makedirs("models", exist_ok=True)
    best_model.save_model("models/xgb_rating_model_tuned.json")
    print("Saved tuned model to models/xgb_rating_model_tuned.json")


if __name__ == "__main__":
    main()
