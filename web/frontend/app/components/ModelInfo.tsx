'use client';

interface ModelInfoProps {
  info: {
    model_type: string;
    objective: string;
    total_features: number;
    hyperparameters: Record<string, any>;
    training_seasons: number[];
    validation_season: number;
    test_season: number;
    feature_categories: string[];
  };
  metrics?: {
    train?: {
      rmse: number;
      r2: number;
      mae: number;
      mean_error: number;
      n_samples: number;
    };
    validation?: {
      rmse: number;
      r2: number;
      mae: number;
      mean_error: number;
      n_samples: number;
    };
    test?: {
      rmse: number;
      r2: number;
      mae: number;
      mean_error: number;
      n_samples: number;
    };
  };
}

export default function ModelInfo({ info, metrics }: ModelInfoProps) {
  return (
    <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-6 mb-8">
      <h2 className="text-2xl font-bold text-black dark:text-zinc-50 mb-4">
        Model Overview
      </h2>
      
      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50 mb-2">
            How It Works
          </h3>
          <p className="text-zinc-700 dark:text-zinc-300 mb-4">
            The model uses an XGBoost gradient boosting algorithm to predict team strength ratings. 
            It learns from historical data (seasons {info.training_seasons.join(', ')}) to understand 
            which statistics best predict team performance.
          </p>
          <p className="text-zinc-700 dark:text-zinc-300">
            For each week, the model uses <strong>only data available up to that week</strong> - 
            ensuring no lookahead bias. Rankings reflect what we knew at that point in the season.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
          <div>
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50 mb-2">
              Training Data
            </h3>
            <ul className="list-disc list-inside text-zinc-700 dark:text-zinc-300 space-y-1">
              <li>Training: {info.training_seasons.join(', ')}</li>
              <li>Validation: {info.validation_season}</li>
              <li>Test: {info.test_season}</li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50 mb-2">
              Model Details
            </h3>
            <ul className="list-disc list-inside text-zinc-700 dark:text-zinc-300 space-y-1">
              <li>Type: {info.model_type}</li>
              <li>Features: {info.total_features}</li>
              <li>Max Depth: {info.hyperparameters.max_depth || 'N/A'}</li>
              <li>Learning Rate: {info.hyperparameters.learning_rate || 'N/A'}</li>
            </ul>
          </div>
        </div>

        {/* Performance Metrics Section */}
        {metrics && (
          <div className="mt-6 pt-6 border-t border-zinc-200 dark:border-zinc-700">
            <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50 mb-4">
              Model Performance
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {metrics.train && (
                <div className="bg-zinc-50 dark:bg-zinc-800 rounded-lg p-4">
                  <h4 className="font-semibold text-zinc-900 dark:text-zinc-50 mb-2">
                    Training Set
                  </h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-zinc-600 dark:text-zinc-400">RMSE:</span>
                      <span className="font-mono text-zinc-900 dark:text-zinc-50">{metrics.train.rmse.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-600 dark:text-zinc-400">R² Score:</span>
                      <span className="font-mono text-zinc-900 dark:text-zinc-50">{metrics.train.r2.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-600 dark:text-zinc-400">MAE:</span>
                      <span className="font-mono text-zinc-900 dark:text-zinc-50">{metrics.train.mae.toFixed(3)}</span>
                    </div>
                    <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-2">
                      {metrics.train.n_samples.toLocaleString()} samples
                    </div>
                  </div>
                </div>
              )}
              
              {metrics.validation && (
                <div className="bg-zinc-50 dark:bg-zinc-800 rounded-lg p-4">
                  <h4 className="font-semibold text-zinc-900 dark:text-zinc-50 mb-2">
                    Validation Set
                  </h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-zinc-600 dark:text-zinc-400">RMSE:</span>
                      <span className="font-mono text-zinc-900 dark:text-zinc-50">{metrics.validation.rmse.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-600 dark:text-zinc-400">R² Score:</span>
                      <span className="font-mono text-zinc-900 dark:text-zinc-50">{metrics.validation.r2.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-600 dark:text-zinc-400">MAE:</span>
                      <span className="font-mono text-zinc-900 dark:text-zinc-50">{metrics.validation.mae.toFixed(3)}</span>
                    </div>
                    <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-2">
                      {metrics.validation.n_samples.toLocaleString()} samples
                    </div>
                  </div>
                </div>
              )}
              
              {metrics.test && (
                <div className="bg-zinc-50 dark:bg-zinc-800 rounded-lg p-4">
                  <h4 className="font-semibold text-zinc-900 dark:text-zinc-50 mb-2">
                    Test Set
                  </h4>
                  <div className="space-y-1 text-sm">
                    <div className="flex justify-between">
                      <span className="text-zinc-600 dark:text-zinc-400">RMSE:</span>
                      <span className="font-mono text-zinc-900 dark:text-zinc-50">{metrics.test.rmse.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-600 dark:text-zinc-400">R² Score:</span>
                      <span className="font-mono text-zinc-900 dark:text-zinc-50">{metrics.test.r2.toFixed(3)}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-zinc-600 dark:text-zinc-400">MAE:</span>
                      <span className="font-mono text-zinc-900 dark:text-zinc-50">{metrics.test.mae.toFixed(3)}</span>
                    </div>
                    <div className="text-xs text-zinc-500 dark:text-zinc-400 mt-2">
                      {metrics.test.n_samples.toLocaleString()} samples
                    </div>
                  </div>
                </div>
              )}
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mt-4">
              <strong>RMSE</strong> (Root Mean Squared Error): Lower is better • 
              <strong> R²</strong>: Higher is better (1.0 = perfect) • 
              <strong> MAE</strong> (Mean Absolute Error): Lower is better
            </p>
          </div>
        )}

        <div className="mt-6">
          <h3 className="text-lg font-semibold text-zinc-900 dark:text-zinc-50 mb-2">
            Feature Categories
          </h3>
          <div className="flex flex-wrap gap-2">
            {info.feature_categories.map((cat) => (
              <span
                key={cat}
                className="px-3 py-1 bg-zinc-100 dark:bg-zinc-800 text-zinc-700 dark:text-zinc-300 rounded-full text-sm"
              >
                {cat}
              </span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}