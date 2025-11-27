import {
  Star,
  GitFork,
  Users,
  GitCommit,
  TrendingUp,
  Calendar,
  ExternalLink,
} from "lucide-react";
import { StatsCard } from "@/components/StatsCard";
import { TimelineChart } from "@/components/TimelineChart";
import { LanguagePieChart } from "@/components/LanguagePieChart";
import { StarsDistributionChart } from "@/components/StarsDistributionChart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { GitHubAnalysisResponse } from "@/lib/api";

interface OverviewProps {
  data: GitHubAnalysisResponse | any;
}

const Overview = ({ data }: OverviewProps) => {
  const safe = <T,>(val: T | undefined | null, fallback: T) =>
    val !== undefined && val !== null ? val : fallback;

  const avatar = safe<string>(data?.avatar_url, "");
  const username = safe<string>(data?.username, "Unknown");
  const bio = safe<string>(data?.bio, "");
  const profileUrl = safe<string>(data?.profile_url, "");

  const account_age_years = Number(safe<number>(data?.account_age_years, 0));
  const total_repos = Number(safe<number>(data?.total_repos, 0));
  const followers = Number(safe<number>(data?.followers, 0));

  const total_stars = Number(safe<number>(data?.total_stars, 0));
  const total_commits_past_year = Number(
    safe<number>(data?.total_commits_past_year, 0)
  );

  const commits_per_year = Number(safe<number>(data?.commits_per_year, 0));
  const repos_per_year = Number(safe<number>(data?.repos_per_year, 0));
  const average_repo_stars = Number(
    safe<number>(data?.average_repo_stars, 0)
  );

  const most_used_languages = Array.isArray(data?.most_used_languages)
    ? data.most_used_languages
    : [];

  const repo_timeline = Array.isArray(data?.repo_timeline)
    ? data.repo_timeline
    : [];

  const stars_distribution = Array.isArray(data?.stars_distribution)
    ? data.stars_distribution
    : [];

  const top_languages = Array.isArray(data?.top_languages)
    ? data.top_languages
    : [];

  return (
    <div className="space-y-6">
      {/* Profile Header */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col sm:flex-row items-center sm:items-start gap-4">
            <Avatar className="h-20 w-20">
              <AvatarImage src={avatar} alt={username} />
              <AvatarFallback className="text-2xl">
                {username.slice(0, 2).toUpperCase()}
              </AvatarFallback>
            </Avatar>

            <div className="flex-1 text-center sm:text-left">
              <div className="flex flex-col sm:flex-row sm:items-center gap-2">
                <h1 className="text-3xl font-bold tracking-tight">
                  @{username}
                </h1>

                {profileUrl && (
                  <Button variant="ghost" size="sm" asChild>
                    <a
                      href={profileUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      <ExternalLink className="h-4 w-4 mr-1" />
                      GitHub
                    </a>
                  </Button>
                )}
              </div>

              {bio && (
                <p className="text-muted-foreground mt-1 max-w-2xl">{bio}</p>
              )}

              <p className="text-sm text-muted-foreground mt-2">
                {account_age_years.toFixed(1)} years on GitHub ·{" "}
                {total_repos} repositories · {followers} followers
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatsCard title="Total Stars" value={total_stars} icon={Star} />
        <StatsCard title="Total Repositories" value={total_repos} icon={GitFork} />
        <StatsCard
          title="Commits (Past Year)"
          value={total_commits_past_year}
          icon={GitCommit}
        />
        <StatsCard title="Followers" value={followers} icon={Users} />
      </div>

      {/* Charts Row 1 */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Calendar className="h-5 w-5 text-primary" />
              Repository Creation Timeline
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <TimelineChart data={repo_timeline} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Star className="h-5 w-5 text-primary" />
              Stars Distribution
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[300px]">
              <StarsDistributionChart data={stars_distribution} />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Charts Row 2 */}
      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Top Languages</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[350px]">
              <LanguagePieChart data={top_languages} />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Activity Metrics
            </CardTitle>
          </CardHeader>

          <CardContent className="space-y-6">
            {/* Commits per Year */}
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm font-medium">Commits per Year</span>
                <span className="text-sm font-bold">{commits_per_year}</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary"
                  style={{
                    width: `${Math.min(
                      (commits_per_year / 1000) * 100,
                      100
                    )}%`,
                  }}
                />
              </div>
            </div>

            {/* Repos per Year */}
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm font-medium">Repos per Year</span>
                <span className="text-sm font-bold">{repos_per_year}</span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-chart-2"
                  style={{
                    width: `${Math.min((repos_per_year / 20) * 100, 100)}%`,
                  }}
                />
              </div>
            </div>

            {/* Average Repo Stars */}
            <div>
              <div className="flex justify-between mb-2">
                <span className="text-sm font-medium">Average Repo Stars</span>
                <span className="text-sm font-bold">
                  {average_repo_stars.toFixed(1)}
                </span>
              </div>
              <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-chart-3"
                  style={{
                    width: `${Math.min(
                      (average_repo_stars / 200) * 100,
                      100
                    )}%`,
                  }}
                />
              </div>
            </div>

            {/* Most Used Languages */}
            <div className="pt-4 border-t">
              <p className="text-sm text-muted-foreground">
                Most Used Languages
              </p>
              <div className="flex flex-wrap gap-2 mt-2">
                {most_used_languages.map((lang: string) => (
                  <span
                    key={lang}
                    className="px-2 py-1 text-xs bg-muted rounded-md font-medium"
                  >
                    {lang}
                  </span>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Overview;
