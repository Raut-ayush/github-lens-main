import { useEffect, useRef, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Gamepad2, Trophy, Timer, Zap } from "lucide-react";

interface Fruit {
  id: number;
  x: number;
  y: number;
  vx: number;
  vy: number;
  type: string;
  emoji: string;
  sliced: boolean;
  size: number;
}

const FRUITS = [
  { type: "apple", emoji: "🍎" },
  { type: "orange", emoji: "🍊" },
  { type: "watermelon", emoji: "🍉" },
  { type: "grape", emoji: "🍇" },
  { type: "banana", emoji: "🍌" },
  { type: "strawberry", emoji: "🍓" },
  { type: "peach", emoji: "🍑" },
  { type: "cherry", emoji: "🍒" },
];

const GAME_DURATION = 120; // 2 minutes

interface FruitSliceGameProps {
  onHighScore?: (score: number) => void;
  isVisible?: boolean;
}

export const FruitSliceGame = ({ onHighScore, isVisible = true }: FruitSliceGameProps) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [score, setScore] = useState(0);
  const [highScore, setHighScore] = useState(() => {
    const saved = localStorage.getItem("fruitSliceHighScore");
    return saved ? parseInt(saved, 10) : 0;
  });
  const [timeLeft, setTimeLeft] = useState(GAME_DURATION);
  const [isPlaying, setIsPlaying] = useState(false);
  const [gameOver, setGameOver] = useState(false);
  const [combo, setCombo] = useState(0);
  
  const fruitsRef = useRef<Fruit[]>([]);
  const mouseTrailRef = useRef<{ x: number; y: number; time: number }[]>([]);
  const frameRef = useRef<number>(0);
  const lastSpawnRef = useRef(0);
  const idCounterRef = useRef(0);

  const spawnFruit = useCallback((canvasWidth: number, canvasHeight: number) => {
    const fruit = FRUITS[Math.floor(Math.random() * FRUITS.length)];
    const fromLeft = Math.random() > 0.5;
    
    const newFruit: Fruit = {
      id: idCounterRef.current++,
      x: fromLeft ? -30 : canvasWidth + 30,
      y: canvasHeight - 50,
      vx: fromLeft ? (Math.random() * 3 + 2) : -(Math.random() * 3 + 2),
      vy: -(Math.random() * 8 + 10),
      type: fruit.type,
      emoji: fruit.emoji,
      sliced: false,
      size: 40 + Math.random() * 20,
    };
    
    fruitsRef.current.push(newFruit);
  }, []);

  const checkSlice = useCallback((mouseX: number, mouseY: number) => {
    if (!isPlaying) return;
    
    mouseTrailRef.current.push({ x: mouseX, y: mouseY, time: Date.now() });
    mouseTrailRef.current = mouseTrailRef.current.filter(p => Date.now() - p.time < 100);
    
    if (mouseTrailRef.current.length < 2) return;
    
    let slicedAny = false;
    
    fruitsRef.current.forEach(fruit => {
      if (fruit.sliced) return;
      
      const dist = Math.hypot(fruit.x - mouseX, fruit.y - mouseY);
      if (dist < fruit.size) {
        fruit.sliced = true;
        slicedAny = true;
        setScore(prev => prev + 10 + combo * 5);
        setCombo(prev => prev + 1);
      }
    });
    
    if (!slicedAny && mouseTrailRef.current.length > 5) {
      setCombo(0);
    }
  }, [isPlaying, combo]);

  const startGame = () => {
    setScore(0);
    setTimeLeft(GAME_DURATION);
    setIsPlaying(true);
    setGameOver(false);
    setCombo(0);
    fruitsRef.current = [];
    mouseTrailRef.current = [];
  };

  const endGame = useCallback(() => {
    setIsPlaying(false);
    setGameOver(true);
    
    if (score > highScore) {
      setHighScore(score);
      localStorage.setItem("fruitSliceHighScore", score.toString());
      onHighScore?.(score);
    }
  }, [score, highScore, onHighScore]);

  // Game timer
  useEffect(() => {
    if (!isPlaying) return;
    
    const timer = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          endGame();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
    
    return () => clearInterval(timer);
  }, [isPlaying, endGame]);

  // Game loop
  useEffect(() => {
    if (!isPlaying || !canvasRef.current) return;
    
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const gameLoop = () => {
      const now = Date.now();
      
      // Spawn fruits
      if (now - lastSpawnRef.current > 800) {
        spawnFruit(canvas.width, canvas.height);
        lastSpawnRef.current = now;
      }
      
      // Clear canvas
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      
      // Draw trail
      if (mouseTrailRef.current.length > 1) {
        ctx.strokeStyle = "hsl(var(--primary))";
        ctx.lineWidth = 3;
        ctx.lineCap = "round";
        ctx.beginPath();
        mouseTrailRef.current.forEach((point, i) => {
          if (i === 0) ctx.moveTo(point.x, point.y);
          else ctx.lineTo(point.x, point.y);
        });
        ctx.stroke();
      }
      
      // Update and draw fruits
      fruitsRef.current = fruitsRef.current.filter(fruit => {
        fruit.x += fruit.vx;
        fruit.y += fruit.vy;
        fruit.vy += 0.3; // gravity
        
        // Draw fruit
        ctx.font = `${fruit.size}px serif`;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        
        if (fruit.sliced) {
          ctx.globalAlpha = 0.5;
          ctx.fillText(fruit.emoji, fruit.x - 10, fruit.y);
          ctx.fillText(fruit.emoji, fruit.x + 10, fruit.y);
          ctx.globalAlpha = 1;
        } else {
          ctx.fillText(fruit.emoji, fruit.x, fruit.y);
        }
        
        // Remove if off screen
        return fruit.y < canvas.height + 100;
      });
      
      frameRef.current = requestAnimationFrame(gameLoop);
    };
    
    frameRef.current = requestAnimationFrame(gameLoop);
    
    return () => {
      if (frameRef.current) {
        cancelAnimationFrame(frameRef.current);
      }
    };
  }, [isPlaying, spawnFruit]);

  // Mouse/touch handlers
  const handleMove = (clientX: number, clientY: number) => {
    if (!canvasRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    checkSlice(clientX - rect.left, clientY - rect.top);
  };

  if (!isVisible) return null;

  return (
    <Card className="w-full max-w-lg mx-auto">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Gamepad2 className="h-5 w-5 text-primary" />
            Fruit Slice
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="flex items-center gap-1">
              <Trophy className="h-3 w-3" />
              {highScore}
            </Badge>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {!isPlaying && !gameOver && (
          <div className="text-center py-8">
            <p className="text-muted-foreground mb-4">
              Slice fruits while waiting for analysis!
            </p>
            <p className="text-sm text-muted-foreground mb-4">
              Your high score will be saved
            </p>
            <Button onClick={startGame} size="lg">
              <Gamepad2 className="mr-2 h-4 w-4" />
              Start Game
            </Button>
          </div>
        )}

        {gameOver && (
          <div className="text-center py-8">
            <h3 className="text-2xl font-bold mb-2">Game Over!</h3>
            <p className="text-4xl font-bold text-primary mb-4">{score}</p>
            {score >= highScore && score > 0 && (
              <Badge className="mb-4 bg-yellow-500">New High Score!</Badge>
            )}
            <div className="space-y-2">
              <Button onClick={startGame} size="lg" className="w-full">
                Play Again
              </Button>
            </div>
          </div>
        )}

        {isPlaying && (
          <>
            <div className="flex justify-between items-center mb-2">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-primary" />
                <span className="font-bold">{score}</span>
                {combo > 1 && (
                  <Badge variant="secondary" className="text-xs">
                    {combo}x combo
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-2">
                <Timer className="h-4 w-4" />
                <span className={timeLeft <= 10 ? "text-destructive font-bold" : ""}>
                  {Math.floor(timeLeft / 60)}:{(timeLeft % 60).toString().padStart(2, "0")}
                </span>
              </div>
            </div>
            <canvas
              ref={canvasRef}
              width={400}
              height={300}
              className="w-full border rounded-lg bg-gradient-to-b from-sky-100 to-sky-200 dark:from-sky-900 dark:to-sky-950 cursor-crosshair touch-none"
              onMouseMove={(e) => handleMove(e.clientX, e.clientY)}
              onTouchMove={(e) => {
                e.preventDefault();
                const touch = e.touches[0];
                handleMove(touch.clientX, touch.clientY);
              }}
            />
          </>
        )}
      </CardContent>
    </Card>
  );
};
