'use client';

import { useState } from 'react';
import RankingsTable from '@/app/components/RankingsTable';

export default function RankingsPage() {
  const [season, setSeason] = useState(2025);
  const [week, setWeek] = useState(15); // Current week

  return (
    <div className="min-h-screen bg-zinc-50 dark:bg-black py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <h1 className="text-4xl font-bold text-black dark:text-zinc-50 mb-2">
          CFB Top 25 Rankings
        </h1>
        <p className="text-lg text-zinc-600 dark:text-zinc-400 mb-4">
          Week {week}, {season} Season
        </p>
        <div className="flex gap-4 mb-8 items-end">
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
              Season
            </label>
            <input
              type="number"
              value={season}
              onChange={(e) => setSeason(parseInt(e.target.value) || 2025)}
              className="px-3 py-2 border border-zinc-300 dark:border-zinc-700 rounded-lg bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-50 w-24"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-zinc-700 dark:text-zinc-300 mb-1">
              Week
            </label>
            <input
              type="number"
              value={week}
              onChange={(e) => setWeek(parseInt(e.target.value) || 15)}
              className="px-3 py-2 border border-zinc-300 dark:border-zinc-700 rounded-lg bg-white dark:bg-zinc-800 text-zinc-900 dark:text-zinc-50 w-24"
            />
          </div>
        </div>
        <RankingsTable season={season} week={week} />
      </div>
    </div>
  );
}