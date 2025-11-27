const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export interface GitHubAnalysisResponse {
  username: string;
  avatar_url?: string;
  bio?: string;
  profile_url?: string;
  account_age_years: number;
  total_repos: number;
  public_gists: number;
  followers: number;
  following: number;
  commits_per_year: number;
  repos_per_year: number;
  average_repo_stars: number;
  total_stars: number;
  total_commits_past_year: number;
  // ML Insights
  expertise_level?: number;
  domain_label?: number;
  developer_rank?: number;
  recommended_skills?: Array<{ skill: string; priority: number }>;
  // Languages & Skills
  most_used_languages: string[];
  top_languages: Array<{ name: string; value: number; percentage: number }>;
  top_skills: Array<{ name: string; frequency: number; percentage: number; level?: string }>;
  // Repositories
  top_repos: Array<{
    name: string;
    stars: number;
    forks: number;
    watchers: number;
    commits_past_12m: number;
    contributors: number;
    pulls: number;
    issues: number;
    main_language: string;
    detected_tech: string[];
    dependencies: string[];
    size: number;
    last_pushed_at: string;
    created_at: string;
    description?: string;
    url?: string;
    complexity_score?: number;
  }>;
  // Charts
  repo_timeline: Array<{ year: number; count: number }>;
  stars_distribution: Array<{ range: string; count: number }>;
  language_growth: Array<{ month: string; [key: string]: number | string }>;
  frameworks: Array<{ name: string; count: number }>;
  all_repos?: any[];
}

export async function analyzeGitHubUser(username: string): Promise<GitHubAnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/api/analyze?username=${encodeURIComponent(username)}`);
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to analyze user' }));
    throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
  }
  
  return response.json();
}

export async function clearUserCache(username: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/api/clear-cache/${encodeURIComponent(username)}`, {
    method: 'POST',
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to clear cache' }));
    throw new Error(error.detail || `HTTP ${response.status}: ${response.statusText}`);
  }
}

export async function healthCheck(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE_URL}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
