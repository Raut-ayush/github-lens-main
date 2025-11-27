import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { RepoTable } from "@/components/RepoTable";
import { FolderGit2 } from "lucide-react";
import { GitHubAnalysisResponse } from "@/lib/api";

interface RepositoriesProps {
  data: GitHubAnalysisResponse;
}

const Repositories = ({ data }: RepositoriesProps) => {
  const repositories = data.all_repos || data.top_repos || [];

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Repositories Explorer</h1>
        <p className="text-muted-foreground mt-1">
          Browse and analyze all {repositories.length} repositories
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <FolderGit2 className="h-5 w-5 text-primary" />
            All Repositories
          </CardTitle>
        </CardHeader>
        <CardContent>
          <RepoTable repositories={repositories} />
        </CardContent>
      </Card>
    </div>
  );
};

export default Repositories;
