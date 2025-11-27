import { useState } from "react";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";

interface Repository {
  id?: number | string;
  name: string;
  stars?: number;
  forks?: number;
  watchers?: number;
  commits_past_12m?: number;
  contributors?: number;
  pulls?: number;
  issues?: number;
  main_language?: string;
  detected_tech?: string[];
  dependencies?: string[];
  size?: number;
  last_pushed_at?: string;
  created_at?: string;
}

interface RepoTableProps {
  repositories: Repository[];
}

export function RepoTable({ repositories }: RepoTableProps) {
  const [search, setSearch] = useState("");
  const [selectedRepo, setSelectedRepo] = useState<Repository | null>(null);

  // defensive defaults
  const repoList = Array.isArray(repositories) ? repositories : [];

  const filteredRepos = repoList.filter((repo) => {
    const name = (repo.name || "").toString().toLowerCase();
    const lang = (repo.main_language || "").toString().toLowerCase();
    const q = search.toLowerCase();
    return name.includes(q) || lang.includes(q);
  });

  const formatDate = (dateString?: string) => {
    if (!dateString) return "—";
    const d = new Date(dateString);
    if (isNaN(d.getTime())) return "—";
    return d.toLocaleDateString("en-US", {
      year: "numeric",
      month: "short",
      day: "numeric",
    });
  };

  return (
    <>
      <div className="mb-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            placeholder="Search repositories..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-10"
          />
        </div>
      </div>

      <div className="rounded-md border">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Repository</TableHead>
              <TableHead className="text-right">Stars</TableHead>
              <TableHead className="text-right">Forks</TableHead>
              <TableHead className="text-right">Commits (12m)</TableHead>
              <TableHead>Language</TableHead>
              <TableHead>Last Updated</TableHead>
              <TableHead></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredRepos.map((repo, idx) => {
              const key = repo.id ?? `${repo.name ?? "repo"}-${idx}`;
              return (
                <TableRow key={String(key)} className="cursor-pointer hover:bg-muted/50">
                  <TableCell className="font-medium">{repo.name || "Unnamed"}</TableCell>
                  <TableCell className="text-right">{(repo.stars ?? 0).toLocaleString()}</TableCell>
                  <TableCell className="text-right">{repo.forks ?? 0}</TableCell>
                  <TableCell className="text-right">{repo.commits_past_12m ?? 0}</TableCell>
                  <TableCell>
                    <Badge variant="outline">{repo.main_language || "Unknown"}</Badge>
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {formatDate(repo.last_pushed_at)}
                  </TableCell>
                  <TableCell>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => setSelectedRepo(repo)}
                    >
                      View
                    </Button>
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </div>

      <Dialog open={!!selectedRepo} onOpenChange={() => setSelectedRepo(null)}>
        <DialogContent className="max-w-2xl">
          <DialogHeader>
            <DialogTitle className="text-2xl">{selectedRepo?.name ?? "Repository"}</DialogTitle>
          </DialogHeader>
          {selectedRepo && (
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <p className="text-sm text-muted-foreground">Stars</p>
                  <p className="text-lg font-semibold">{(selectedRepo.stars ?? 0).toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Forks</p>
                  <p className="text-lg font-semibold">{selectedRepo.forks ?? 0}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Contributors</p>
                  <p className="text-lg font-semibold">{selectedRepo.contributors ?? 0}</p>
                </div>
                <div>
                  <p className="text-sm text-muted-foreground">Pull Requests</p>
                  <p className="text-lg font-semibold">{selectedRepo.pulls ?? 0}</p>
                </div>
              </div>

              <div>
                <p className="text-sm text-muted-foreground mb-2">Technologies</p>
                <div className="flex flex-wrap gap-2">
                  {(selectedRepo.detected_tech ?? []).length === 0 ? (
                    <span className="text-sm text-muted-foreground">None detected</span>
                  ) : (
                    (selectedRepo.detected_tech ?? []).map((tech, i) => (
                      <Badge key={`${tech ?? "tech"}-${i}`} variant="secondary">
                        {tech || "Unknown"}
                      </Badge>
                    ))
                  )}
                </div>
              </div>

              <div>
                <p className="text-sm text-muted-foreground mb-2">Dependencies</p>
                <div className="flex flex-wrap gap-2">
                  {(selectedRepo.dependencies ?? []).length === 0 ? (
                    <span className="text-sm text-muted-foreground">None</span>
                  ) : (
                    (selectedRepo.dependencies ?? []).map((dep, i) => (
                      <Badge key={`${dep ?? "dep"}-${i}`} variant="outline" className="text-xs">
                        {dep || "Unknown"}
                      </Badge>
                    ))
                  )}
                </div>
              </div>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Created</p>
                  <p>{formatDate(selectedRepo.created_at)}</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Last Updated</p>
                  <p>{formatDate(selectedRepo.last_pushed_at)}</p>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </>
  );
}
