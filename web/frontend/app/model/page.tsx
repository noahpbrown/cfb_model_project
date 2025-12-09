'use client';

import { useEffect, useState } from 'react';
import FeatureImportanceChart from '@/app/components/FeatureImportanceChart';
import ModelInfo from '@/app/components/ModelInfo';

interface Feature {
  name: string;
  importance: number;
  description: string;
  category: string;
}

interface ModelInfoData {
  model_type: string;
  objective: string;
  total_features: number;
  hyperparameters: Record<string, any>;
  training_seasons: number[];
  validation_season: number;
  test_season: number;
  feature_categories: string[];
}

interface MetricsData {
  metrics: {
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

export default function ModelPage() {
  const [features, setFeatures] = useState<Feature[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelInfoData | null>(null);
  const [metrics, setMetrics] = useState<MetricsData['metrics'] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [featuresRes, infoRes, metricsRes] = await Promise.all([
          fetch(`${API_BASE_URL}/api/model/feature-importance`),
          fetch(`${API_BASE_URL}/api/model/info`),
          fetch(`${API_BASE_URL}/api/model/metrics`)
        ]);

        if (!featuresRes.ok || !infoRes.ok) {
          throw new Error('Failed to fetch model data');
        }

        const featuresData = await featuresRes.json();
        const infoData = await infoRes.json();
        
        // Metrics might fail if training data isn't available, so handle gracefully
        let metricsData = null;
        if (metricsRes.ok) {
          const metricsResponse = await metricsRes.json();
          metricsData = metricsResponse.metrics;
        }

        setFeatures(featuresData.features);
        setModelInfo(infoData);
        setMetrics(metricsData);
        setLoading(false);
      } catch (err) {
        console.error('Error loading model data:', err);
        setError('Failed to load model information');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-black py-8 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="flex justify-center items-center py-20">
            <p className="text-zinc-600 dark:text-zinc-400">Loading model information...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-zinc-50 dark:bg-black py-8 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="flex justify-center items-center py-20">
            <p className="text-red-600 dark:text-red-400">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-black dark:text-zinc-50 mb-2">
          Model Explanation
        </h1>
        <p className="text-lg text-zinc-600 dark:text-zinc-400 mb-8">
          How the CFB Rating Model calculates team rankings
        </p>

        {modelInfo ? (
          <ModelInfo info={modelInfo} metrics={metrics || undefined} />
        ) : null}
        
        {features.length > 0 && (
          <div className="mt-8">
            <h2 className="text-2xl font-bold text-black dark:text-zinc-50 mb-4">
              Feature Importance
            </h2>
            <p className="text-zinc-600 dark:text-zinc-400 mb-6">
              Features are ranked by how much they contribute to the model&apos;s predictions. 
              Higher importance means the feature has more influence on rankings.
            </p>
            <FeatureImportanceChart features={features} />
          </div>
        )}
      </div>
    </div>
  );
}