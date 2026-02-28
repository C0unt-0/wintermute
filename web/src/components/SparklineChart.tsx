import { ResponsiveContainer, LineChart, Line } from "recharts";

interface SparklineChartProps {
  data: number[];
  color?: string;
  height?: number;
}

export default function SparklineChart({
  data,
  color = "var(--data)",
  height = 60,
}: SparklineChartProps) {
  const chartData = data.map((v) => ({ v }));

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <Line
            type="monotone"
            dataKey="v"
            stroke={color}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
