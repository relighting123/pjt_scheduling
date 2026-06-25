import { useCallback, useRef } from "react";
import Plot, { Plotly } from "../lib/plotlyComponent";
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

type AxisRange = { prefix: string; start: number; end: number };

function collectAxisRanges(event: PlotRelayoutEvent): AxisRange[] {
  const ranges = new Map<string, AxisRange>();

  for (const key of Object.keys(event)) {
    const bracket = key.match(/^(xaxis\d*)\.range\[0\]$/);
    if (bracket) {
      const prefix = bracket[1] === "xaxis" ? "xaxis" : bracket[1];
      const start = event[key as keyof PlotRelayoutEvent];
      const end = event[`${prefix}.range[1]` as keyof PlotRelayoutEvent];
      if (typeof start === "number" && typeof end === "number") {
        ranges.set(prefix, { prefix, start, end });
      }
      continue;
    }

    const arrayKey = key.match(/^(xaxis\d*)\.range$/);
    if (arrayKey) {
      const prefix = arrayKey[1] === "xaxis" ? "xaxis" : arrayKey[1];
      const value = event[key as keyof PlotRelayoutEvent];
      if (Array.isArray(value) && value.length === 2) {
        const [start, end] = value;
        if (typeof start === "number" && typeof end === "number") {
          ranges.set(prefix, { prefix, start, end });
        }
      }
    }
  }

  return [...ranges.values()];
}

function relayoutClampXMin(
  event: PlotRelayoutEvent,
  minX: number,
): Record<string, [number, number]> | null {
  const updates: Record<string, [number, number]> = {};

  for (const { prefix, start, end } of collectAxisRanges(event)) {
    if (start < minX) {
      updates[`${prefix}.range`] = [minX, minX + (end - start)];
    }
  }

  return Object.keys(updates).length > 0 ? updates : null;
}

export default function PlotChart({
  data,
  layout,
  className,
  onPointClick,
  clampXMin,
}: PlotChartProps) {
  const graphDivRef = useRef<HTMLElement | null>(null);

  const handleClick = (ev: Readonly<PlotMouseEvent>) => {
    const pt = ev.points?.[0];
    if (pt && typeof pt.pointIndex === "number" && onPointClick) {
      onPointClick(pt.pointIndex);
    }
  };

  const clampPanX = useCallback(
    (ev: PlotRelayoutEvent) => {
      if (clampXMin === undefined) return;
      const updates = relayoutClampXMin(ev, clampXMin);
      if (!updates || !graphDivRef.current) return;
      void Plotly.relayout(graphDivRef.current, updates);
    },
    [clampXMin],
  );

  const isGanttPan = layout.dragmode === "pan" && clampXMin !== undefined;
  const ganttPanHandlers = isGanttPan
    ? ({ onRelayouting: clampPanX, onRelayout: clampPanX } as Record<string, unknown>)
    : {};

  return (
    <div className={className ?? "plot-chart"}>
      <Plot
        data={data}
        layout={{ ...layout, autosize: true }}
        config={plotConfig}
        useResizeHandler
        style={{ width: "100%", height: "100%" }}
        onClick={onPointClick ? handleClick : undefined}
        onInitialized={(_, graphDiv) => {
          graphDivRef.current = graphDiv;
        }}
        {...ganttPanHandlers}
      />
    </div>
  );
}
