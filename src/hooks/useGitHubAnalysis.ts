import { useState, useEffect, useRef } from "react";
import { GitHubAnalysisResponse } from "@/lib/api";

/**
 * Hook for GitHub analysis with polling and elapsed time tracking
 * - Polls /api/analyze_status 
 * - Tracks elapsed time for progress display
 * - Supports adjustable poll interval for game mode
 */

export function useGitHubAnalysis() {
  const [data, setData] = useState<GitHubAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const pollRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);
  const runningRef = useRef(false);
  const pollIntervalRef = useRef(3000);

  const clearTimers = () => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    runningRef.current = false;
    pollIntervalRef.current = 3000;
  };

  const setPollInterval = (ms: number) => {
    pollIntervalRef.current = ms;
  };

  const analyze = async (username: string) => {
    if (runningRef.current) return;
    runningRef.current = true;
    setLoading(true);
    setData(null);
    setElapsedSeconds(0);

    // Start elapsed timer
    timerRef.current = window.setInterval(() => {
      setElapsedSeconds((prev) => prev + 1);
    }, 1000);

    try {
      // Start analysis job
      try {
        const startResp = await fetch(`/api/analyze?username=${encodeURIComponent(username)}`, { method: "GET" });
        if (!startResp.ok) {
          console.warn("Analyzer start returned non-OK:", startResp.status, await startResp.text().catch(() => ""));
        }
      } catch (e) {
        console.warn("Failed to trigger analyzer start:", e);
      }

      // Polling function
      const poll = async () => {
        try {
          const res = await fetch(`/api/analyze_status?username=${encodeURIComponent(username)}`, { method: "GET" });
          const statusJson = await (async () => {
            try {
              const text = await res.text();
              if (text.trim().startsWith("<")) {
                throw new Error("Status endpoint returned HTML");
              }
              return JSON.parse(text);
            } catch (e) {
              console.warn("Failed to parse analyze_status response:", e);
              return null;
            }
          })();

          if (!statusJson) return;

          const status = statusJson.status;
          const ready = !!statusJson.loveble_ready;

          if (status === "done" && ready) {
            try {
              const finalRes = await fetch(`/api/analyze?username=${encodeURIComponent(username)}`);
              const finalCt = finalRes.headers.get("content-type") || "";
              if (!finalRes.ok) {
                const txt = await finalRes.text().catch(() => "");
                throw new Error(`Final analyze fetch failed ${finalRes.status}: ${txt}`);
              }
              if (!finalCt.includes("application/json")) {
                const txt = await finalRes.text().catch(() => "");
                throw new Error(`Final analyze returned non-json: ${finalCt}. Body starts: ${txt.slice(0, 200)}`);
              }
              const finalJson = await finalRes.json();
              if (finalJson && finalJson.loveble) {
                setData(finalJson.loveble);
              } else {
                console.warn("Final analyze did not include loveble JSON", finalJson);
              }
            } catch (e) {
              console.error("Failed to fetch final result:", e);
            } finally {
              setLoading(false);
              clearTimers();
            }
          } else if (status === "failed") {
            console.warn("Analysis failed:", statusJson);
            setLoading(false);
            clearTimers();
          }
        } catch (e) {
          console.warn("Polling error:", e);
        }
      };

      // Initial poll
      await poll();
      
      // Setup interval polling
      if (!pollRef.current) {
        pollRef.current = window.setInterval(async () => {
          await poll();
        }, pollIntervalRef.current);
      }
    } catch (e) {
      console.error("Analyze() top-level error:", e);
      setLoading(false);
      runningRef.current = false;
    }
  };

  useEffect(() => {
    return () => {
      clearTimers();
    };
  }, []);

  return { data, loading, analyze, elapsedSeconds, setPollInterval };
}
