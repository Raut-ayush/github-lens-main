import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";

interface ScoreCardProps {
  title: string;
  score: number;
  description: string;
}

export function ScoreCard({ title, score, description }: ScoreCardProps) {
  const getScoreColor = (score: number) => {
    if (score >= 80) return "text-success";
    if (score >= 60) return "text-primary";
    if (score >= 40) return "text-chart-4";
    return "text-destructive";
  };

  return (
    <Card className="transition-all hover:shadow-lg">
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="flex items-baseline gap-2">
          <span className={`text-4xl font-bold ${getScoreColor(score)}`}>
            {score}
          </span>
          <span className="text-muted-foreground">/100</span>
        </div>
        <Progress value={score} className="h-2" />
        <p className="text-sm text-muted-foreground">{description}</p>
      </CardContent>
    </Card>
  );
}
