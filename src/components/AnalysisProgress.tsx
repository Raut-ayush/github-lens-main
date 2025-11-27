import { Progress } from "@/components/ui/progress";
import { Clock, Loader2 } from "lucide-react";

interface AnalysisProgressProps {
  elapsedSeconds: number;
  estimatedTotalSeconds?: number;
}

export const AnalysisProgress = ({
  elapsedSeconds,
  estimatedTotalSeconds = 240, // ~4 min average
}: AnalysisProgressProps) => {
  const progress = Math.min((elapsedSeconds / estimatedTotalSeconds) * 100, 95);
  
  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, "0")}`;
  };

  const remainingSeconds = Math.max(0, estimatedTotalSeconds - elapsedSeconds);
  
  const getStageMessage = () => {
    if (elapsedSeconds < 30) return "Starting analysis...";
    if (elapsedSeconds < 60) return "Fetching GitHub profile...";
    if (elapsedSeconds < 120) return "Analyzing repositories...";
    if (elapsedSeconds < 180) return "Processing languages & skills...";
    if (elapsedSeconds < 240) return "Running ML predictions...";
    return "Finalizing results...";
  };

  return (
    <div className="w-full max-w-md space-y-4">
      <div className="flex items-center justify-center gap-2 text-primary">
        <Loader2 className="h-5 w-5 animate-spin" />
        <span className="font-medium">{getStageMessage()}</span>
      </div>
      
      <Progress value={progress} className="h-2" />
      
      <div className="flex justify-between text-sm text-muted-foreground">
        <div className="flex items-center gap-1">
          <Clock className="h-4 w-4" />
          <span>Elapsed: {formatTime(elapsedSeconds)}</span>
        </div>
        <span>~{formatTime(remainingSeconds)} remaining</span>
      </div>
      
      <p className="text-xs text-center text-muted-foreground">
        Analysis typically takes 2-4 minutes depending on profile size
      </p>
    </div>
  );
};
