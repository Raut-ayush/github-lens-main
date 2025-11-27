import { Area, AreaChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend } from "recharts";

interface LanguageGrowthChartProps {
  data: Array<{
    month: string;
    [key: string]: string | number;
  }>;
}

export function LanguageGrowthChart({ data }: LanguageGrowthChartProps) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={data}>
        <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
        <XAxis
          dataKey="month"
          className="text-xs"
          tick={{ fill: "hsl(var(--muted-foreground))" }}
        />
        <YAxis
          className="text-xs"
          tick={{ fill: "hsl(var(--muted-foreground))" }}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "hsl(var(--card))",
            border: "1px solid hsl(var(--border))",
            borderRadius: "var(--radius)",
          }}
        />
        <Legend />
        <Area
          type="monotone"
          dataKey="TypeScript"
          stackId="1"
          stroke="hsl(var(--chart-1))"
          fill="hsl(var(--chart-1))"
        />
        <Area
          type="monotone"
          dataKey="Python"
          stackId="1"
          stroke="hsl(var(--chart-2))"
          fill="hsl(var(--chart-2))"
        />
        <Area
          type="monotone"
          dataKey="JavaScript"
          stackId="1"
          stroke="hsl(var(--chart-3))"
          fill="hsl(var(--chart-3))"
        />
        <Area
          type="monotone"
          dataKey="Go"
          stackId="1"
          stroke="hsl(var(--chart-4))"
          fill="hsl(var(--chart-4))"
        />
        <Area
          type="monotone"
          dataKey="Rust"
          stackId="1"
          stroke="hsl(var(--chart-5))"
          fill="hsl(var(--chart-5))"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
