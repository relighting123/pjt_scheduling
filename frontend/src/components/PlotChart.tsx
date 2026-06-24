import Plot from "react-plotly.js";
import type { Data, Layout, Config, PlotMouseEvent } from "plotly.js";

interface PlotChartProps {
  data: Data[];
  layout: Partial<Layout>;
  className?: string;
  onPointClick?: (pointIndex: number, xValue?: number) => void;
}

const plotConfig: Partial<Config> = {
  responsive: true,
  displayModeBar: false,
  displaylogo: false,
};

export default function PlotChart({ data, layout, className, onPointClick }: PlotChartProps) {
  const handleClick = (ev: Readonly<PlotMouseEvent>) => {
    const pt = ev.points?.[0];
    if (pt && typeof pt.pointIndex === "number" && onPointClick) {
      const xVal = typeof pt.x === "number" ? pt.x : undefined;
      onPointClick(pt.pointIndex, xVal);
    }
  };

  return (
    <div className={className ?? "plot-chart"}>
      <Plot
        data={data}
        layout={{ ...layout, autosize: true }}
        config={plotConfig}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        onClick={onPointClick ? handleClick : undefined}
      />
    </div>
  );
}
