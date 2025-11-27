import { useEffect, useState } from "react";
import { useParams, Routes, Route, Navigate } from "react-router-dom";
import { SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { AppSidebar } from "@/components/AppSidebar";
import { useGitHubAnalysis } from "@/hooks/useGitHubAnalysis";
import Overview from "./Overview";
import Repositories from "./Repositories";
import Skills from "./Skills";
import MLInsights from "./MLInsights";
import { FruitSliceGame } from "@/components/FruitSliceGame";
import { AnalysisProgress } from "@/components/AnalysisProgress";
import { Button } from "@/components/ui/button";
import { Gamepad2 } from "lucide-react";

const Dashboard = () => {
  const { username } = useParams<{ username: string }>();
  const { data, loading, analyze, elapsedSeconds, setPollInterval } = useGitHubAnalysis();
  const [showGame, setShowGame] = useState(false);

  useEffect(() => {
    if (username) analyze(username);
  }, [username]);

  // Adjust poll interval based on game state
  useEffect(() => {
    setPollInterval(showGame ? 30000 : 3000); // 30s during game, 3s otherwise
  }, [showGame, setPollInterval]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center p-4">
        <div className="text-center space-y-6 max-w-lg w-full">
          <h2 className="text-2xl font-bold">
            Analyzing @{username}'s GitHub profile...
          </h2>
          
          <AnalysisProgress elapsedSeconds={elapsedSeconds} />
          
          <div className="pt-4">
            {!showGame ? (
              <div className="space-y-3">
                <p className="text-sm text-muted-foreground">
                  This takes a few minutes. Want to play a game while you wait?
                </p>
                <Button variant="outline" onClick={() => setShowGame(true)}>
                  <Gamepad2 className="mr-2 h-4 w-4" />
                  Play Fruit Slice
                </Button>
              </div>
            ) : (
              <div className="space-y-4">
                <FruitSliceGame isVisible={showGame} />
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setShowGame(false)}
                >
                  Hide Game
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center space-y-4">
          <p className="text-lg text-muted-foreground">No data available</p>
        </div>
      </div>
    );
  }

  return (
    <SidebarProvider>
      <div className="min-h-screen flex w-full bg-background">
        <AppSidebar username={username!} />
        <div className="flex-1 flex flex-col">
          <header className="sticky top-0 z-10 flex h-14 items-center gap-4 border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-6">
            <SidebarTrigger />
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">Analyzing:</span>
              <span className="font-semibold">@{username}</span>
            </div>
          </header>
          <main className="flex-1 p-6">
            <Routes>
              <Route path="/" element={<Navigate to="overview" replace />} />
              <Route path="overview" element={<Overview data={data} />} />
              <Route path="repositories" element={<Repositories data={data} />} />
              <Route path="skills" element={<Skills data={data} />} />
              <Route path="ml-insights" element={<MLInsights data={data} />} />
            </Routes>
          </main>
        </div>
      </div>
    </SidebarProvider>
  );
};

export default Dashboard;
