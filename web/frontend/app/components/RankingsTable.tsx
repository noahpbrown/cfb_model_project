'use client';

import { useEffect, useState } from 'react';
import Image from 'next/image';

interface Team {
  rank: number;
  team: string;
  abbr: string;
  wins_cum: number;
  losses_cum: number;
  games_played: number;
  rating_pred: number;
  spread_vs_1: number;
  primary_color: string;
  primary_logo_url: string;
}

interface RankingsTableProps {
  season: number;
  week: number;
}

// Move API_BASE_URL outside component - cleaner and more efficient
const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000';

export default function RankingsTable({ season, week }: RankingsTableProps) {
  const [rankings, setRankings] = useState<Team[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    
    // Ensure no trailing slash on base URL
    const baseUrl = API_BASE_URL.replace(/\/$/, '');
    // Add trailing slash to match FastAPI route: /api/rankings/
    const url = `${baseUrl}/api/rankings/?season=${season}&week=${week}`;
    
    console.log('🔍 Final fetch URL:', url);
    
    fetch(url, {
      method: 'GET',
      mode: 'cors',
    })
      .then(async (res) => {
        if (!res.ok) {
          const text = await res.text();
          console.error('❌ API error status:', res.status, 'body:', text);
          throw new Error(`Failed to fetch rankings: ${res.status}`);
        }
        return res.json();
      })
      .then((data) => {
        setRankings(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error('Error loading rankings:', err);
        setError('Failed to load rankings. Please try again later.');
        setLoading(false);
      });
  }, [season, week]);

  if (loading) {
    return (
      <div className="flex justify-center items-center py-20">
        <p className="text-zinc-600 dark:text-zinc-400">Loading rankings...</p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="flex justify-center items-center py-20">
        <div className="text-center">
          <p className="text-red-600 dark:text-red-400 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-zinc-900 dark:bg-zinc-100 text-white dark:text-black rounded-lg hover:bg-zinc-800 dark:hover:bg-zinc-200 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (rankings.length === 0) {
    return (
      <div className="flex justify-center items-center py-20">
        <p className="text-zinc-600 dark:text-zinc-400">No rankings available.</p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-zinc-900 rounded-lg shadow-lg overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead className="bg-zinc-100 dark:bg-zinc-800">
            <tr>
              <th className="px-6 py-4 text-left text-sm font-semibold text-zinc-700 dark:text-zinc-300">Rank</th>
              <th className="px-6 py-4 text-left text-sm font-semibold text-zinc-700 dark:text-zinc-300">Team</th>
              <th className="px-6 py-4 text-center text-sm font-semibold text-zinc-700 dark:text-zinc-300">Record</th>
              <th className="px-6 py-4 text-right text-sm font-semibold text-zinc-700 dark:text-zinc-300">Rating</th>
              <th className="px-6 py-4 text-right text-sm font-semibold text-zinc-700 dark:text-zinc-300">Spread vs #1</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-200 dark:divide-zinc-700">
            {rankings.map((team) => (
              <tr 
                key={team.rank} 
                className="hover:bg-zinc-50 dark:hover:bg-zinc-800 transition-colors"
              >
                <td className="px-6 py-4 text-sm font-bold text-zinc-900 dark:text-zinc-100">
                  {team.rank}
                </td>
                <td className="px-6 py-4">
                  <div className="flex items-center gap-3">
                    <div 
                      className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
                      style={{ backgroundColor: team.primary_color + '20' }}
                    >
                      {team.primary_logo_url ? (
                        <Image
                          src={team.primary_logo_url}
                          alt={team.team}
                          width={32}
                          height={32}
                          className="rounded-full"
                          unoptimized
                        />
                      ) : (
                        <span className="text-xs font-bold" style={{ color: team.primary_color }}>
                          {team.abbr}
                        </span>
                      )}
                    </div>
                    <div>
                      <div className="font-semibold text-zinc-900 dark:text-zinc-100">
                        {team.team}
                      </div>
                      <div className="text-xs text-zinc-500 dark:text-zinc-400">
                        {team.abbr}
                      </div>
                    </div>
                  </div>
                </td>
                <td className="px-6 py-4 text-center text-sm text-zinc-700 dark:text-zinc-300">
                  {Math.round(team.wins_cum)}-{Math.round(team.losses_cum)}
                </td>
                <td className="px-6 py-4 text-right text-sm font-mono text-zinc-700 dark:text-zinc-300">
                  {team.rating_pred.toFixed(1)}
                </td>
                <td className="px-6 py-4 text-right text-sm font-mono">
                  {team.rank === 1 ? (
                    <span className="text-zinc-500 dark:text-zinc-400">--</span>
                  ) : (
                    <span className="font-semibold text-amber-600 dark:text-amber-400">
                      +{team.spread_vs_1.toFixed(1)}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}