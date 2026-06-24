import { useCallback, useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { Data, Layout, Config, PlotMouseEvent, PlotRelayoutEvent } from "plotly.js";

interface PlotChartProps {
  data: Data[];
  layout: Partial<Layout>;
  className?: string;
  onPointClick?: (pointIndex: number) => void;
  /** 간트 pan 시 X축이 0 미만으로 가지 않도록 clamp */
  clampXMin?: number;
}

const plotConfig: Partial<Config> = {
  responsive: true,
  displayModeBar: false,
  displaylogo: false,
  scrollZoom: false,
  doubleClick: false,
  modeBarButtonsToRemove: [
    "zoom2d",
    "pan2d",
    "select2d",
    "lasso2d",
    "zoomIn2d",
    "zoomOut2d",
    "autoScale2d",
    "resetScale2d",
  ],
};

function relayoutClampXMin(
  event: PlotRelayoutEvent,
  minX: number,
): Record<string, [number, number]> | null {
  const updates: Record<string, [number, number]> = {};
  const axisPrefixes = new Set<string>();

  for (const key of Object.keys(event)) {
    const m = key.match(/^(xaxis\d*)\.range\[0\]$/);
    if (m) axisPrefixes.add(m[1] === "xaxis" ? "xaxis" : m[1]);
  }

  for (const prefix of axisPrefixes) {
    const startKey = `${prefix}.range[0]`;
    const endKey = `${prefix}.range[1]`;
    const start = event[startKey as keyof PlotRelayoutEvent];
    const end = event[endKey as keyof PlotRelayoutEvent];
    if (typeof start !== "number" || typeof end !== "number") continue;
    if (start < minX) {
      const span = end - start;
      updates[`${prefix}.range`] = [minX, minX + span];
    }
  }

  return Object.keys(updates).length > 0 ? updates : null;
}

function patchLayoutRanges(
  base: Partial<Layout>,
  updates: Record<string, [number, number]>,
): Partial<Layout> {
  const next: Record<string, unknown> = { ...base };
  for (const [key, range] of Object.entries(updates)) {
    const axisKey = key.replace(".range", "");
    const prev = (next[axisKey] as Record<string, unknown> | undefined) ?? {};
    next[axisKey] = { ...prev, range };
  }
  return next as Partial<Layout>;
}

export default function PlotChart({
  data,
  layout,
  className,
  onPointClick,
  clampXMin,
}: PlotChartProps) {
  const [rangePatch, setRangePatch] = useState<Partial<Layout>>({});

  useEffect(() => {
    setRangePatch({});
  }, [data]);

  const mergedLayout = useMemo(
    () => ({ ...layout, ...rangePatch, autosize: true }),
    [layout, rangePatch],
  );

  const handleClick = (ev: Readonly<PlotMouseEvent>) => {
    const pt = ev.points?.[0];
    if (pt && typeof pt.pointIndex === "number" && onPointClick) {
      onPointClick(pt.pointIndex);
    }
  };

  const handleRelayout = useCallback(
    (ev: PlotRelayoutEvent) => {
      if (clampXMin === undefined) return;
      const updates = relayoutClampXMin(ev, clampXMin);
      if (!updates) return;
      setRangePatch((prev) => patchLayoutRanges(prev, updates));
    },
    [clampXMin],
  );

  const isGanttPan = layout.dragmode === "pan" && clampXMin !== undefined;

  return (
    <div className={className ?? "plot-chart"}>
      <Plot
        data={data}
        layout={mergedLayout}
        config={plotConfig}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        onClick={onPointClick ? handleClick : undefined}
        onRelayout={isGanttPan ? handleRelayout : undefined}
      />
    </div>
  );
}
