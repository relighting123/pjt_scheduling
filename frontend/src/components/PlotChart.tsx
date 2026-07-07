import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { loadPlotly, type PlotlyBundle } from "../lib/plotlyComponent";
import { CHART_HOVERLABEL } from "../lib/charts";
import type { Data, Layout, Config, PlotMouseEvent, PlotRelayoutEvent } from "plotly.js";

interface PlotChartProps {
  data: Data[];
  layout: Partial<Layout>;
  className?: string;
  onPointClick?: (pointIndex: number) => void;
  /** 간트 pan 시 X축이 0 미만으로 가지 않도록 clamp */
  clampXMin?: number;
  /** 세로 스크롤 시 상단 X축(시간 눈금) 고정 */
  scrollable?: boolean;
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

const STICKY_X_BG_CLASS = "sticky-xaxis-bg";

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

function stickyXElements(graphEl: HTMLElement): Element[] {
  const layers = [...graphEl.querySelectorAll(".xaxislayer-above")];
  // 다중 서브플롯 간트: coupled x축 1개만 sticky (겹침·정렬 오류 방지)
  if (layers.length > 1) return [layers[0]];
  return layers;
}

function measureStickyBases(scrollEl: HTMLElement, graphEl: HTMLElement): Map<Element, number> {
  const bases = new Map<Element, number>();
  const scrollRect = scrollEl.getBoundingClientRect();
  const scrollTop = scrollEl.scrollTop;

  for (const el of stickyXElements(graphEl)) {
    const rect = el.getBoundingClientRect();
    bases.set(el, rect.top - scrollRect.top + scrollTop);
  }
  return bases;
}

function ensureXAxisBackgrounds(graphEl: HTMLElement, fill: string) {
  graphEl.querySelectorAll<SVGGElement>(".xaxislayer-above").forEach((layer) => {
    if (layer.querySelector(`.${STICKY_X_BG_CLASS}`)) return;
    try {
      const bbox = layer.getBBox();
      const padX = 8;
      const padY = 6;
      const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rect.setAttribute("class", STICKY_X_BG_CLASS);
      rect.setAttribute("x", String(bbox.x - padX));
      rect.setAttribute("y", String(bbox.y - padY));
      rect.setAttribute("width", String(bbox.width + padX * 2));
      rect.setAttribute("height", String(bbox.height + padY * 2));
      rect.setAttribute("fill", fill);
      layer.insertBefore(rect, layer.firstChild);
    } catch {
      /* getBBox unavailable before first paint */
    }
  });
}

function applyStickyXAxis(
  scrollEl: HTMLElement,
  bases: Map<Element, number>,
) {
  const scrollTop = scrollEl.scrollTop;
  bases.forEach((base, el) => {
    const dy = Math.max(0, scrollTop - base);
    const next = dy > 0 ? `translateY(${dy}px)` : "";
    const svg = el as SVGElement;
    if (svg.style.transform !== next) {
      svg.style.transform = next;
    }
  });
}

function layoutPixelHeight(layout: Partial<Layout>): number | undefined {
  const h = layout.height;
  return typeof h === "number" && Number.isFinite(h) ? h : undefined;
}

export default function PlotChart({
  data,
  layout,
  className,
  onPointClick,
  clampXMin,
  scrollable = false,
}: PlotChartProps) {
  const graphDivRef = useRef<HTMLElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const stickyBasesRef = useRef<Map<Element, number>>(new Map());
  const paperBg = (layout.paper_bgcolor as string | undefined) ?? "#ffffff";
  const plotPixelHeight = layoutPixelHeight(layout);

  // Plotly(약 5MB)는 첫 차트 렌더 때 동적 로드 → 초기 번들 분리
  const [bundle, setBundle] = useState<PlotlyBundle | null>(null);
  const plotlyRef = useRef<PlotlyBundle["Plotly"] | null>(null);
  useEffect(() => {
    let mounted = true;
    loadPlotly().then((b) => {
      if (!mounted) return;
      plotlyRef.current = b.Plotly;
      setBundle(b);
    });
    return () => { mounted = false; };
  }, []);

  const handleClick = (ev: Readonly<PlotMouseEvent>) => {
    const pt = ev.points?.[0];
    if (pt && typeof pt.pointIndex === "number" && onPointClick) {
      onPointClick(pt.pointIndex);
    }
  };

  const refreshStickyXAxis = useCallback(() => {
    const scrollEl = scrollRef.current;
    const graphEl = graphDivRef.current;
    if (!scrollable || !scrollEl || !graphEl) return;

    ensureXAxisBackgrounds(graphEl, paperBg);
    stickyBasesRef.current = measureStickyBases(scrollEl, graphEl);
    applyStickyXAxis(scrollEl, stickyBasesRef.current);
  }, [scrollable, paperBg]);

  const clearHover = useCallback(() => {
    const graphEl = graphDivRef.current;
    if (!graphEl) return;
    plotlyRef.current?.Fx.unhover(graphEl);
  }, []);

  const handleScroll = useCallback(() => {
    const scrollEl = scrollRef.current;
    if (!scrollEl) return;
    applyStickyXAxis(scrollEl, stickyBasesRef.current);
    clearHover();
  }, [clearHover]);

  const scheduleStickyRefresh = useCallback(() => {
    if (!scrollable) return;
    requestAnimationFrame(() => refreshStickyXAxis());
  }, [scrollable, refreshStickyXAxis]);

  useEffect(() => {
    if (!scrollable) return;
    const scrollEl = scrollRef.current;
    if (!scrollEl) return;

    const ro = new ResizeObserver(() => scheduleStickyRefresh());
    ro.observe(scrollEl);
    return () => ro.disconnect();
  }, [scrollable, scheduleStickyRefresh]);

  const handleGraphInit = useCallback((_: unknown, graphDiv: HTMLElement) => {
    graphDivRef.current = graphDiv;
    if (scrollable) scheduleStickyRefresh();
  }, [scrollable, scheduleStickyRefresh]);

  const clampPanX = useCallback(
    (ev: PlotRelayoutEvent) => {
      if (clampXMin === undefined) return;
      const updates = relayoutClampXMin(ev, clampXMin);
      if (!updates || !graphDivRef.current || !plotlyRef.current) return;
      void plotlyRef.current.relayout(graphDivRef.current, updates).then(() => scheduleStickyRefresh());
    },
    [clampXMin, scheduleStickyRefresh],
  );

  const isGanttPan = layout.dragmode === "pan" && clampXMin !== undefined;
  // pan 중에만 clamp – onRelayout은 hover 직후에도 호출되어 툴팁이 깜빡일 수 있음
  const ganttPanHandlers = isGanttPan
    ? ({ onRelayouting: clampPanX } as Record<string, unknown>)
    : {};

  const plotStyle: CSSProperties = plotPixelHeight
    ? { width: "100%", height: plotPixelHeight }
    : { width: "100%", height: "100%" };

  const Plot = bundle?.Plot;
  const loadingStyle: CSSProperties = plotPixelHeight
    ? { height: plotPixelHeight }
    : { minHeight: 220 };
  const plot = Plot ? (
    <Plot
      {...({
        data,
        layout: {
          ...layout,
          // scrollable도 너비는 컨테이너에 맞춰 autosize(높이는 style의 고정 px 유지)
          autosize: true,
          hoverlabel: { ...CHART_HOVERLABEL, ...layout.hoverlabel },
        },
        config: plotConfig,
        useResizeHandler: true,
        style: plotStyle,
        onClick: onPointClick ? handleClick : undefined,
        onInitialized: handleGraphInit,
        ...ganttPanHandlers,
      } as Record<string, unknown>)}
    />
  ) : (
    <div className="chart-loading" style={loadingStyle}>
      <span className="chart-loading-spinner" /> 차트 로딩…
    </div>
  );

  return (
    <div className={className ?? "plot-chart"}>
      {scrollable ? (
        <div
          className="gantt-chart-scroll"
          ref={scrollRef}
          onScroll={handleScroll}
        >
          {plot}
        </div>
      ) : (
        plot
      )}
    </div>
  );
}
