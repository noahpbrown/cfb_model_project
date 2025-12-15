# CFB Rating Model Project

A comprehensive college football team rating system that combines linear least-squares ratings with an XGBoost machine learning model to produce weekly team rankings. The model uses advanced statistics, strength of schedule, preseason priors, and game-level résumé features to predict team strength.

**Now includes a full-stack web application** for viewing rankings, exploring model details, and chatting with an AI assistant about the model.

## Overview

This project builds a two-stage rating system for college football teams:

1. **Linear Rating System**: Uses least-squares regression on game margins to produce baseline team ratings for each week of each season
2. **XGBoost Model**: A gradient boosting model that predicts the linear ratings using a rich set of features including:
   - Advanced offensive/defensive statistics (EPA, success rate, explosiveness)
   - Pass/rush splits for offense and defense
   - Strength of schedule metrics
   - Preseason priors (talent ratings, returning production)
   - Opponent-adjusted statistics
   - Game-level résumé features (wins, margins, quality wins)

The final output is weekly Top 25 rankings with visual graphics suitable for social media, plus a web application for interactive exploration.

## Project Structure

```
cfb_model_project/
├── data/                    # All data files (CSV)
│   ├── games_*.csv          # Game results by season
│   ├── ratings_*.csv        # Linear system ratings
│   ├── team_week_features_*.csv  # Feature engineering outputs
│   ├── training_team_week_2019_2024.csv  # Final training dataset
│   └── team_logos.csv       # Team branding data (ESPN logos, colors, IDs)
├── models/                  # Trained XGBoost models
│   ├── xgb_rating_model.json
│   └── xgb_rating_model_tuned.json
├── outputs/                 # Weekly ranking outputs
│   ├── top25_*.json         # JSON payloads for rankings
│   └── top25_*.png          # Rendered graphics
├── scripts/                 # Python scripts (numbered by execution order)
│   ├── 01_fetch_games.py
│   ├── 02_build_ratings.py
│   ├── ...
│   └── render_top25_graphic.py
└── web/                     # Web application
    ├── backend/             # FastAPI backend
    │   ├── app/
    │   │   ├── api/         # API endpoints
    │   │   │   ├── rankings.py
    │   │   │   ├── chat.py
    │   │   │   └── model/
    │   │   │       └── explain.py
    │   │   ├── models/      # Pydantic schemas
    │   │   ├── services/    # Business logic
    │   │   └── main.py
    │   ├── requirements.txt
    │   └── Procfile         # Railway deployment config
    └── frontend/            # Next.js frontend
        ├── app/
        │   ├── page.tsx     # Home page
        │   ├── rankings/    # Rankings page
        │   ├── model/       # Model explanation page
        │   ├── chat/        # AI chat page
        │   └── components/ # React components
        └── package.json
```

## Web Application

The project includes a full-stack web application for interactive exploration of rankings and model details.

### Features

- **Rankings Page**: View Top 25 rankings for any season and week with an interactive table
- **Model Explanation Page**: Explore model architecture, feature importance, and performance metrics
- **AI Chat Assistant**: Ask questions about the model, rankings, or college football statistics using OpenAI GPT-4o-mini

### Running Locally

#### Backend (FastAPI)

1. Navigate to the backend directory:
```bash
cd web/backend
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables:
```bash
export CFBD_API_KEY="your_cfbd_api_key"
export OPENAI_API_KEY="your_openai_api_key"  # Required for chat feature
export FRONTEND_URL="http://localhost:3000"  # Optional, defaults to localhost:3000
```

4. Run the server:
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`.

#### Frontend (Next.js)

1. Navigate to the frontend directory:
```bash
cd web/frontend
```

2. Install dependencies:
```bash
npm install
```

3. Set environment variable (create `.env.local`):
```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

4. Run the development server:
```bash
npm run dev
```

The frontend will be available at `http://localhost:3000`.

### API Endpoints

#### Rankings
- `GET /api/rankings/?season=2025&week=14` - Get Top 25 rankings for a specific season and week
- `GET /api/rankings/weeks?season=2025` - Get list of available weeks for a season

#### Model Information
- `GET /api/model/info` - Get model architecture and hyperparameters
- `GET /api/model/feature-importance` - Get feature importance rankings
- `GET /api/model/metrics` - Get model performance metrics (RMSE, R², MAE) for train/val/test splits

#### Chat
- `POST /api/chat/` - Chat with AI assistant about the model (requires `OPENAI_API_KEY`)

### Deployment

- **Backend**: Deployed on Railway (configured via `Procfile` and `railway.json`)
- **Frontend**: Deployed on Vercel
- CORS is configured to allow requests from the frontend domain and Vercel preview deployments

## Data Pipeline

The project follows a sequential pipeline where each script builds upon previous outputs:

### Stage 1: Data Collection
- **`01_fetch_games.py`** / **`06_fetch_games_multiseason.py`**: Fetches game results from the CFBD API for specified seasons
  - Output: `data/games_YYYY.csv` files with columns: season, week, home_team, away_team, home_points, away_points, neutral_site, conferences

### Stage 2: Linear Rating System
- **`02_build_ratings.py`** / **`07_build_ratings_multiseason.py`**: Computes least-squares ratings for each week
  - Uses game margins to solve: `margin = home_rating - away_rating + HFA`
  - Normalizes ratings to mean zero
  - Output: `data/ratings_YYYY.csv` with columns: season, week, team, rating_linear

### Stage 3: Feature Engineering

#### Base Advanced Stats
- **`04_build_team_week_features.py`** / **`05_build_team_week_features_multiseason.py`**: Builds cumulative team-week features from CFBD advanced stats
  - Fetches EPA (Points Per Attempt), success rate, explosiveness
  - Computes play-weighted cumulative averages through each week
  - Includes pass/rush splits for offense and defense
  - Output: `data/team_week_features_YYYY.csv`

#### Strength of Schedule
- **`11_add_sos_features.py`**: Adds strength-of-schedule metrics
  - Computes mean/min/max opponent ratings faced through each week
  - Output: `data/team_week_features_with_sos_2019_2024.csv`

#### Preseason Priors
- **`12_add_preseason_priors.py`**: Adds preseason information
  - Talent ratings (247 Sports composite)
  - Returning production (offense, defense, total)
  - Output: `data/team_week_features_full_2019_2024.csv`

#### Opponent Adjustments
- **`13_add_opponent_adjusted_stats.py`**: Creates opponent-adjusted EPA metrics
  - Adjusts offensive EPA by subtracting average opponent defensive EPA
  - Adjusts defensive EPA by subtracting average opponent offensive EPA
  - Uses same-week opponent stats (may have lookahead bias)

- **`14_add_lagged_opponent_stats.py`**: Creates lagged opponent-adjusted metrics
  - Uses opponent stats from previous week (no lookahead)
  - Output: `off_ppa_adj_lagged_cum`, `def_ppa_adj_lagged_cum`

#### Résumé Features
- **`15_add_margin_resume_features.py`**: Adds game-level résumé metrics
  - Cumulative margin, wins, losses, win percentage
  - Quality wins (top 10, top 25)
  - Road/neutral wins
  - Average opponent rating faced
  - Final output: `data/team_week_features_full_2019_2024.csv`

### Stage 4: Training Dataset
- **`08_build_training_dataset.py`**: Combines features with ratings and filters to FBS teams
  - Merges `team_week_features_full_2019_2024.csv` with `ratings_2019_2024.csv`
  - Filters to FBS teams only (varies by season)
  - Output: `data/training_team_week_2019_2024.csv`

### Stage 5: Model Training
- **`09_train_rating_model.py`**: Trains XGBoost model
  - Train/Val/Test split: 2019,2021,2022 / 2023 / 2024
  - Uses 34 features (see feature list below)
  - Output: `models/xgb_rating_model.json`

- **`09b_tune_model.py`**: Hyperparameter tuning (optional)
  - Uses RandomizedSearchCV
  - Output: `models/xgb_rating_model_tuned.json`

### Stage 6: Weekly Predictions
- **`10_generate_weekly_ratings.py`**: Generates weekly Top 25 rankings
  - Loads trained model and full feature set
  - Filters to specified season/week
  - Predicts ratings for all FBS teams
  - Computes spreads vs #1 using compressed gap method (see details below)
  - Merges team logos and branding data
  - Output: `outputs/top25_YYYY_weekN.json`
  - Arguments: `--season`, `--week`, `--logos-path` (default: `data/team_logos.csv`)

- **`render_top25_graphic.py`**: Creates visual graphic
  - Loads rankings JSON
  - Downloads and caches team logos (uses `team_id` for efficient caching)
  - Renders Top 25 poll-style graphic with column headers
  - Shows rank, logo, team name, record, rating, and spread vs #1
  - Output: `outputs/top25_YYYY_weekN.png`
  - Arguments: `--json_path`, `--out_path`

## Model Details

### Features Used (34 total)

**Meta:**
- `games_played`: Number of games played through this week

**Overall Offense/Defense:**
- `off_ppa_cum`: Cumulative offensive EPA (play-weighted)
- `off_success_cum`: Cumulative offensive success rate
- `off_explosive_cum`: Cumulative offensive explosiveness
- `def_ppa_cum`: Cumulative defensive EPA
- `def_success_cum`: Cumulative defensive success rate
- `def_explosive_cum`: Cumulative defensive explosiveness

**Offense Pass/Rush Splits:**
- `off_pass_ppa_cum`: Cumulative passing EPA
- `off_pass_success_cum`: Cumulative passing success rate
- `off_rush_ppa_cum`: Cumulative rushing EPA
- `off_rush_success_cum`: Cumulative rushing success rate

**Defense Pass/Rush Splits:**
- `def_pass_ppa_cum`: Cumulative passing defense EPA
- `def_pass_success_cum`: Cumulative passing defense success rate
- `def_rush_ppa_cum`: Cumulative rushing defense EPA
- `def_rush_success_cum`: Cumulative rushing defense success rate

**Strength of Schedule:**
- `sos_opp_rating_mean_cum`: Mean opponent rating faced
- `sos_opp_rating_min_cum`: Minimum opponent rating faced
- `sos_opp_rating_max_cum`: Maximum opponent rating faced
- `sos_games_cum`: Number of games with opponent ratings

**Preseason Priors:**
- `talent`: 247 Sports talent composite
- `rp_total`: Returning production (total)
- `rp_offense`: Returning production (offense)
- `rp_defense`: Returning production (defense)

**Opponent-Adjusted (Lagged):**
- `off_ppa_adj_lagged_cum`: Offensive EPA adjusted by previous-week opponent defense
- `def_ppa_adj_lagged_cum`: Defensive EPA adjusted by previous-week opponent offense

**Résumé Features:**
- `margin_cum`: Cumulative point margin
- `wins_cum`: Cumulative wins
- `losses_cum`: Cumulative losses
- `win_pct_cum`: Cumulative win percentage
- `wins_top25_cum`: Wins against top 25 teams
- `wins_top10_cum`: Wins against top 10 teams
- `road_wins_cum`: Road wins
- `neutral_wins_cum`: Neutral site wins
- `opp_rating_game_mean_cum`: Mean opponent rating at game time

### Model Architecture

- **Algorithm**: XGBoost Regressor
- **Objective**: Predict `rating_linear` (from linear system)
- **Hyperparameters** (default model):
  - `max_depth`: 5
  - `learning_rate`: 0.05
  - `n_estimators`: 600
  - `subsample`: 0.7
  - `colsample_bytree`: 0.9
  - `min_child_weight`: 3
  - `reg_lambda`: 2.0
  - `reg_alpha`: 0.5

### Training Strategy

- **Train**: 2019, 2021, 2022 seasons
- **Validation**: 2023 season
- **Test**: 2024 season
- **Evaluation Metrics**: RMSE, R², MAE
- **Note**: 2020 season is intentionally skipped (COVID-19 disruption)

## Setup

### Prerequisites

- Python 3.8+
- Node.js 18+ (for web application)
- CFBD API key (get one at [collegefootballdata.com](https://collegefootballdata.com))
- OpenAI API key (optional, required for chat feature)

### Installation

1. Clone this repository

2. Install Python dependencies:
```bash
pip install pandas numpy xgboost scikit-learn cfbd pillow requests fastapi uvicorn openai
```

3. Install Node.js dependencies (for web application):
```bash
cd web/frontend
npm install
```

4. Set environment variables:
```bash
export CFBD_API_KEY="your_api_key_here"
export OPENAI_API_KEY="your_openai_api_key_here"  # Optional, for chat feature
```

## Usage

### Full Pipeline (First Time)

Run scripts in numerical order:

```bash
# 1. Fetch game data
python scripts/06_fetch_games_multiseason.py

# 2. Build linear ratings
python scripts/07_build_ratings_multiseason.py

# 3. Build base features
python scripts/05_build_team_week_features_multiseason.py

# 4. Add strength of schedule
python scripts/11_add_sos_features.py

# 5. Add preseason priors
python scripts/12_add_preseason_priors.py

# 6. Add opponent-adjusted stats
python scripts/13_add_opponent_adjusted_stats.py

# 7. Add lagged opponent stats
python scripts/14_add_lagged_opponent_stats.py

# 8. Add résumé features
python scripts/15_add_margin_resume_features.py

# 9. Build training dataset
python scripts/08_build_training_dataset.py

# 10. Train model
python scripts/09_train_rating_model.py
```

### Weekly Predictions

After the model is trained, generate weekly rankings:

```bash
# Generate rankings for a specific week
python scripts/10_generate_weekly_ratings.py --season 2025 --week 14

# Optionally specify custom logo file
python scripts/10_generate_weekly_ratings.py \
    --season 2025 \
    --week 14 \
    --logos-path data/team_logos.csv

# Render graphic
python scripts/render_top25_graphic.py \
    --json_path outputs/top25_2025_week14.json \
    --out_path outputs/top25_2025_week14.png
```

### Updating for New Season/Week

To add a new week of data:

1. **Update features** (if new week):
   - Run `05_build_team_week_features_multiseason.py` (will update existing season files)
   - Run `11_add_sos_features.py` through `15_add_margin_resume_features.py` to propagate new features

2. **Generate predictions**:
   - Run `10_generate_weekly_ratings.py` with new season/week
   - Run `render_top25_graphic.py` to create graphic

3. **Web application** will automatically pick up new JSON files from the `outputs/` directory

## Data Files Reference

### Game Files (`games_YYYY.csv`)
- `season`: Year
- `week`: Week number
- `home_team`: Home team name
- `away_team`: Away team name
- `home_points`: Home team score
- `away_points`: Away team score
- `neutral_site`: Boolean for neutral site games
- `home_conference`: Home team conference
- `away_conference`: Away team conference

### Rating Files (`ratings_YYYY.csv`)
- `season`: Year
- `week`: Week number
- `team`: Team name
- `rating_linear`: Least-squares rating (mean-centered)

### Feature Files (`team_week_features_full_2019_2024.csv`)
- One row per (season, week, team)
- Contains all engineered features (see feature list above)
- Used for both training and inference

### Training File (`training_team_week_2019_2024.csv`)
- Merged features + ratings
- Filtered to FBS teams only
- Used for model training

## Output Format

### JSON Output (`top25_YYYY_weekN.json`)
Array of team objects with:
- `rank`: Ranking (1-25)
- `team`: Team name
- `abbr`: Team abbreviation
- `team_id`: ESPN team ID (used for logo caching)
- `wins_cum`: Cumulative wins
- `losses_cum`: Cumulative losses
- `games_played`: Games played
- `rating_pred`: Predicted rating
- `spread_vs_1`: Point spread vs #1 team (compressed gap method, see below)
- `primary_color`: Team color (hex)
- `primary_logo_url`: Logo URL (ESPN CDN)

**Spread Calculation**: The `spread_vs_1` field uses a compressed gap method to normalize the distance between #1 and the elite pack (next 4 teams). The raw rating gap is scaled so that #1 has a 5-point edge over the elite pack average, making spreads more interpretable and preventing #1 from appearing too dominant.

### PNG Output (`top25_YYYY_weekN.png`)
- 1080x1920 portrait graphic
- Top 25 teams with logos, records, ratings, and spreads
- Dark theme with team colors
- Column headers for "Rating" and "Spread"
- Logo caching for efficient rendering (uses `team_id`)
- Pillow 10+ compatible text rendering

## Notes

- **2020 Season**: Intentionally excluded from training due to COVID-19 disruptions
- **FBS Filtering**: Rankings only include FBS teams (varies by season)
- **Lookahead Prevention**: Lagged opponent stats use previous-week data to avoid lookahead bias
- **Missing Data**: XGBoost handles NaN values, but some features may be missing for early weeks or new teams
- **Web Application**: The frontend expects ranking JSON files to be in the `web/backend/outputs/` directory (or symlinked from the root `outputs/` directory)

## Script Reference

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `01_fetch_games.py` | Fetch single season games | CFBD API | `games_YYYY.csv` |
| `02_build_ratings.py` | Build single season ratings | `games_YYYY.csv` | `ratings_YYYY.csv` |
| `04_build_team_week_features.py` | Build single season features | CFBD API | `team_week_features_YYYY.csv` |
| `05_build_team_week_features_multiseason.py` | Build multi-season features | CFBD API | `team_week_features_YYYY.csv` |
| `06_fetch_games_multiseason.py` | Fetch multi-season games | CFBD API | `games_YYYY.csv` |
| `07_build_ratings_multiseason.py` | Build multi-season ratings | `games_YYYY.csv` | `ratings_2019_2024.csv` |
| `08_build_training_dataset.py` | Create training dataset | Features + Ratings | `training_team_week_2019_2024.csv` |
| `09_train_rating_model.py` | Train XGBoost model | Training dataset | `xgb_rating_model.json` |
| `09b_tune_model.py` | Hyperparameter tuning | Training dataset | `xgb_rating_model_tuned.json` |
| `10_generate_weekly_ratings.py` | Generate weekly rankings | Model + Features + Logos | `top25_YYYY_weekN.json` |
| `11_add_sos_features.py` | Add strength of schedule | Base features + Ratings | `team_week_features_with_sos_*.csv` |
| `12_add_preseason_priors.py` | Add preseason data | Features + CFBD API | `team_week_features_full_*.csv` |
| `13_add_opponent_adjusted_stats.py` | Add opponent-adjusted EPA | Features + Games | Updated features |
| `14_add_lagged_opponent_stats.py` | Add lagged opponent stats | Features + Games | Updated features |
| `15_add_margin_resume_features.py` | Add résumé features | Features + Games + Ratings | Updated features |
| `render_top25_graphic.py` | Render visual graphic | `top25_*.json` | `top25_*.png` |

## License

This project uses data from the College Football Data API. Please review their terms of service.

## Author

@noahdawg34

