'use client';

interface Feature {
  name: string;
  importance: number;
  description: string;
  category: string;
}

interface FeatureImportanceChartProps {
  features: Feature[];
}

export default function FeatureImportanceChart({ features }: FeatureImportanceChartProps) {
  // Get top 15 features for main display
  const topFeatures = features.slice(0, 15);
  const maxImportance = Math.max(...features.map(f => f.importance));

  // Group by category for summary
  const byCategory: Record<string, Feature[]> = {};
  features.forEach(f => {
    if (!byCategory[f.category]) {
      byCategory[f.category] = [];
    }
    byCategory[f.category].push(f);
  });

  // Calculate average importance per category
  const categoryAverages = Object.entries(byCategory).map(([cat, feats]) => ({
    category: cat,
    avgImportance: feats.reduce((sum, f) => sum + f.importance, 0) / feats.length,
    count: feats.length
  })).sort((a, b) => b.avgImportance - a.avgImportance);

  return (
    <div className="space-y-8">
      {/* Top Features Bar Chart */}
      <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-bold text-black dark:text-zinc-50 mb-4">
          Top 15 Most Important Features
        </h3>
        <div className="space-y-3">
          {topFeatures.map((feature, idx) => {
            const percentage = (feature.importance / maxImportance) * 100;
            return (
              <div key={feature.name} className="space-y-1">
                <div className="flex items-center justify-between mb-1">
                  <div className="flex items-center gap-3">
                    <span className="text-sm font-bold text-zinc-500 dark:text-zinc-400 w-6">
                      {idx + 1}
                    </span>
                    <span className="font-semibold text-zinc-900 dark:text-zinc-50">
                      {feature.name.replace(/_/g, ' ')}
                    </span>
                    <span className="text-xs px-2 py-1 bg-zinc-100 dark:bg-zinc-800 text-zinc-600 dark:text-zinc-400 rounded">
                      {feature.category}
                    </span>
                  </div>
                  <span className="text-sm font-mono text-zinc-600 dark:text-zinc-400">
                    {feature.importance.toFixed(4)}
                  </span>
                </div>
                <div className="w-full bg-zinc-200 dark:bg-zinc-700 rounded-full h-3">
                  <div
                    className="bg-blue-600 dark:bg-blue-500 h-3 rounded-full transition-all"
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <p className="text-xs text-zinc-600 dark:text-zinc-400 ml-9">
                  {feature.description}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* Category Summary */}
      <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-bold text-black dark:text-zinc-50 mb-4">
          Importance by Category
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {categoryAverages.map(({ category, avgImportance, count }) => {
            const percentage = (avgImportance / maxImportance) * 100;
            return (
              <div key={category} className="space-y-2">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-zinc-900 dark:text-zinc-50">
                    {category}
                  </span>
                  <span className="text-sm text-zinc-600 dark:text-zinc-400">
                    {count} features
                  </span>
                </div>
                <div className="w-full bg-zinc-200 dark:bg-zinc-700 rounded-full h-2">
                  <div
                    className="bg-green-600 dark:bg-green-500 h-2 rounded-full"
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <p className="text-xs text-zinc-500 dark:text-zinc-400">
                  Avg importance: {avgImportance.toFixed(4)}
                </p>
              </div>
            );
          })}
        </div>
      </div>

      {/* All Features Table */}
      <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg p-6">
        <h3 className="text-xl font-bold text-black dark:text-zinc-50 mb-4">
          All Features
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-zinc-100 dark:bg-zinc-800">
              <tr>
                <th className="px-4 py-2 text-left text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                  Rank
                </th>
                <th className="px-4 py-2 text-left text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                  Feature
                </th>
                <th className="px-4 py-2 text-left text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                  Category
                </th>
                <th className="px-4 py-2 text-right text-sm font-semibold text-zinc-700 dark:text-zinc-300">
                  Importance
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-zinc-200 dark:divide-zinc-700">
              {features.map((feature, idx) => (
                <tr key={feature.name} className="hover:bg-zinc-50 dark:hover:bg-zinc-800">
                  <td className="px-4 py-2 text-sm text-zinc-600 dark:text-zinc-400">
                    {idx + 1}
                  </td>
                  <td className="px-4 py-2">
                    <div>
                      <div className="font-medium text-zinc-900 dark:text-zinc-50">
                        {feature.name.replace(/_/g, ' ')}
                      </div>
                      <div className="text-xs text-zinc-500 dark:text-zinc-400">
                        {feature.description}
                      </div>
                    </div>
                  </td>
                  <td className="px-4 py-2 text-sm text-zinc-600 dark:text-zinc-400">
                    {feature.category}
                  </td>
                  <td className="px-4 py-2 text-right text-sm font-mono text-zinc-700 dark:text-zinc-300">
                    {feature.importance.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}