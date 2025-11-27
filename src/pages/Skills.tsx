import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { SkillCloud } from "@/components/SkillCloud";
import { LanguageGrowthChart } from "@/components/LanguageGrowthChart";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { Code2, TrendingUp, Package } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { GitHubAnalysisResponse } from "@/lib/api";

interface SkillsProps {
  data: GitHubAnalysisResponse;
}

const Skills = ({ data }: SkillsProps) => {
  const skills = data.top_skills || [];
  const frameworks = data.frameworks || [];
  const languageGrowth = data.language_growth || [];
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Skills Intelligence</h1>
        <p className="text-muted-foreground mt-1">
          Deep dive into your technical expertise and growth
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code2 className="h-5 w-5 text-primary" />
            Weighted Skill Cloud
          </CardTitle>
        </CardHeader>
        <CardContent>
          <SkillCloud skills={skills} />
        </CardContent>
      </Card>

      {languageGrowth.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Language Growth Over Time
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="h-[350px]">
              <LanguageGrowthChart data={languageGrowth} />
            </div>
          </CardContent>
        </Card>
      )}

      <div className="grid gap-4 md:grid-cols-2">
        {frameworks.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Package className="h-5 w-5 text-primary" />
                Framework Usage
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="h-[300px]">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={frameworks} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                    <XAxis
                      type="number"
                      className="text-xs"
                      tick={{ fill: "hsl(var(--muted-foreground))" }}
                    />
                    <YAxis
                      type="category"
                      dataKey="name"
                      className="text-xs"
                      tick={{ fill: "hsl(var(--muted-foreground))" }}
                      width={80}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        border: "1px solid hsl(var(--border))",
                        borderRadius: "var(--radius)",
                      }}
                    />
                    <Bar dataKey="count" fill="hsl(var(--chart-3))" radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        )}

        <Card>
          <CardHeader>
            <CardTitle>Top Skills Breakdown</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {skills.slice(0, 8).map((skill, index) => (
                <div key={`${skill.name}-${index}`}>
                  <div className="flex justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-sm font-medium">{skill.name}</span>
                      <Badge variant="outline" className="text-xs">
                        {skill.frequency}x
                      </Badge>
                    </div>
                    <span className="text-sm font-bold text-primary">
                      {skill.percentage}%
                    </span>
                  </div>
                  <div className="h-2 bg-muted rounded-full overflow-hidden">
                    <div
                      className="h-full transition-all"
                      style={{
                        width: `${skill.percentage}%`,
                        backgroundColor: `hsl(var(--chart-${(index % 5) + 1}))`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
};

export default Skills;
