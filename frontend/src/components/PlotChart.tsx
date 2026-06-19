import Plot from "react-plotly.js";
import type { Data, Layout, Config } from "plotly.js";

interface PlotChartProps {
  data: Data[];
  layout: Partial<Layout>;
  className?: string;
}

const plotConfig: Partial<Config> = {
  responsive: true,
  displayModeBar: true,
  displaylogo: false,
};

export default function PlotChart({ data, layout, className }: PlotChartProps) {
  return (
    <div className={className ?? "plot-chart"}>
      <Plot
        data={data}
        layout={{ ...layout, autosize: true }}
        config={plotConfig}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
      />
    </div>
  );
}
