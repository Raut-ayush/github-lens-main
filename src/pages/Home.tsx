import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Github, BarChart3, TrendingUp, Zap, Upload } from "lucide-react";
import { useGitHubAnalysis } from "@/hooks/useGitHubAnalysis";

const Home = () => {
  const [username, setUsername] = useState("");
  const { analyze, loading } = useGitHubAnalysis();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    const cleanUsername = username.trim().replace(/^https?:\/\/(www\.)?github\.com\//, '').replace(/\/$/, '');
    
    if (!cleanUsername) return;

    try {
      await analyze(cleanUsername);
      navigate(`/dashboard/${cleanUsername}`);
    } catch (err) {
      // Error is handled by the hook
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-background via-background to-muted/20">
      <div className="container mx-auto px-4 py-16">
        {/* Hero Section */}
        <div className="text-center space-y-6 mb-16">
          <div className="flex justify-center mb-4">
            <div className="p-3 bg-primary/10 rounded-full">
              <Github className="h-12 w-12 text-primary" />
            </div>
          </div>
          
          <h1 className="text-4xl md:text-6xl font-bold tracking-tight">
            GitHub Profile Intelligence
          </h1>
          
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Unlock deep insights into any GitHub profile. Analyze repositories, skills, 
            and development patterns with AI-powered intelligence.
          </p>
        </div>

        {/* Main Input Card */}
        <Card className="max-w-2xl mx-auto mb-12">
          <CardHeader>
            <CardTitle>Analyze a GitHub Profile</CardTitle>
            <CardDescription>
              Enter a GitHub username or profile URL to get started
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="flex gap-2">
                <Input
                  type="text"
                  placeholder="username or github.com/username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  disabled={loading}
                  className="flex-1"
                />
                <Button type="submit" disabled={loading || !username.trim()}>
                  {loading ? "Analyzing..." : "Analyze"}
                </Button>
              </div>
            </form>

            <div className="mt-6 pt-6 border-t">
              <div className="flex items-center justify-between">
                <div className="space-y-1">
                  <p className="text-sm font-medium">Bulk Analysis</p>
                  <p className="text-xs text-muted-foreground">
                    Compare multiple profiles at once (Coming Soon)
                  </p>
                </div>
                <Button variant="outline" disabled>
                  <Upload className="h-4 w-4 mr-2" />
                  Upload CSV
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
          <Card>
            <CardHeader>
              <div className="mb-2">
                <BarChart3 className="h-8 w-8 text-primary" />
              </div>
              <CardTitle className="text-lg">Repository Analytics</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Deep dive into repository metrics, stars, forks, and commit history.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="mb-2">
                <TrendingUp className="h-8 w-8 text-primary" />
              </div>
              <CardTitle className="text-lg">Skills Intelligence</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                Analyze programming languages, frameworks, and technical expertise.
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <div className="mb-2">
                <Zap className="h-8 w-8 text-primary" />
              </div>
              <CardTitle className="text-lg">ML Insights</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground">
                AI-powered scoring for seniority, complexity, and code quality.
              </p>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
};

export default Home;
