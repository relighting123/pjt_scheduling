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

/**
 * Plotly가 (스크롤 컨테이너처럼) 실제 SVG 높이보다 작게 클리핑된 영역 안에
 * 그려질 때, hover label의 배경 박스(화살표+사각형 path)는 원래 위치에
 * 그대로 두고 안의 텍스트만 "화면 안에 들어오게" 별도로 밀어 넣는 내부
 * 버그가 있다 — 그 결과 텍스트가 자기 박스 밖(주로 위쪽)으로 떨어져 나가
 * 붕 떠 보인다(스크롤 여부와 무관하게, 이 박스의 path와 text의 y좌표가
 * 서로 어긋나 있음을 실측으로 확인함). 박스 안에서 텍스트를 다시 수직
 * 중앙 정렬해 이 어긋남을 없앤다.
 */
function realignHoverText(layer: SVGGElement) {
  layer.querySelectorAll<SVGGElement>("g.hovertext").forEach((g) => {
    const path = g.querySelector<SVGPathElement>("path");
    const text = g.querySelector<SVGTextElement>("text");
    if (!path || !text) return;
    if (text.style.transform) text.style.transform = "";
    try {
      const pathBox = path.getBBox();
      const textBox = text.getBBox();
      if (pathBox.height === 0 || textBox.height === 0) return;
      const desiredTop = pathBox.y + (pathBox.height - textBox.height) / 2;
      const dy = desiredTop - textBox.y;
      if (Math.abs(dy) > 0.5) {
        text.style.transform = `translateY(${dy}px)`;
      }
    } catch {
      /* getBBox unavailable before first paint */
    }
  });
}

/**
 * Plotly는 hover label을 배치할 때 "SVG 전체" 기준으로 위/아래 여유 공간을
 * 계산한다 — 스크롤 컨테이너가 overflow-y:auto로 그 SVG의 일부만 보여주고
 * 있다는 사실을 모른다. 그래서 스크롤된 뷰포트 위쪽 가장자리 근처의 막대를
 * 가리키면, SVG 좌표로는 "위에 공간 있음"이라 label을 그 위에 그리지만 실제
 * 화면에서는 그 자리가 스크롤 컨테이너 바깥(클리핑 영역)이라 label이 잘려서
 * 텍스트만 붕 떠 보이거나 아예 안 보이게 된다(스크롤 안 한 상태에서 맨 위
 * 행을 호버해도 동일하게 재현됨 — sticky X축 자체가 아니라 overflow 클리핑이
 * 원인). sticky X축과 같은 방식(translateY)으로 hoverlayer를 보이는 영역
 * 안으로 밀어 넣어 보정한다.
 */
function clampHoverLayer(scrollEl: HTMLElement, graphEl: HTMLElement) {
  const layer = graphEl.querySelector<SVGGElement>(".hoverlayer");
  if (!layer) return;

  // 이전에 적용한 보정을 먼저 지워야 "보정된 위치" 기준이 아니라 Plotly가
  // 실제로 그리려 한 원래 위치를 기준으로 다시 계산할 수 있다.
  if (layer.style.transform) layer.style.transform = "";
  if (!layer.hasChildNodes()) return;

  realignHoverText(layer);

  const scrollRect = scrollEl.getBoundingClientRect();
  const labelRect = layer.getBoundingClientRect();
  if (labelRect.width === 0 && labelRect.height === 0) return;

  const margin = 4;
  // 위쪽 경계는 스크롤 컨테이너 자체가 아니라, sticky X축(있다면 그 위에
  // 그려진 시간 눈금 배경)의 아래 가장자리로 잡는다 — 안 그러면 label을
  // 컨테이너 안으로는 넣어도 sticky 축 라벨과 겹쳐서 글자가 뒤섞여 보인다.
  const stickyBottoms = [...graphEl.querySelectorAll(".xaxislayer-above")]
    .map((el) => el.getBoundingClientRect().bottom);
  const topBound = Math.max(scrollRect.top, ...stickyBottoms);

  const overflowTop = topBound + margin - labelRect.top;
  const overflowBottom = labelRect.bottom - (scrollRect.bottom - margin);

  let dy = 0;
  if (overflowTop > 0) dy = overflowTop;
  else if (overflowBottom > 0) dy = -overflowBottom;

  if (dy !== 0) {
    layer.style.transform = `translateY(${dy}px)`;
  }
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

  const hoverObserverRef = useRef<MutationObserver | null>(null);
  const hoverClampScheduledRef = useRef(false);

  const setupHoverClamp = useCallback((graphEl: HTMLElement) => {
    hoverObserverRef.current?.disconnect();
    if (!scrollable) return;
    const layer = graphEl.querySelector(".hoverlayer");
    if (!layer) return;
    const obs = new MutationObserver(() => {
      // Plotly가 hover label DOM(테두리 박스, 화살표, 텍스트 줄들)을 한
      // 뮤테이션이 아니라 여러 번에 걸쳐 순차적으로 추가하므로, 첫 뮤테이션
      // 시점에 바로 측정하면 아직 완성되지 않은 크기로 잘못 clamp될 수 있다
      // — 같은 프레임 내 나머지 뮤테이션이 끝난 뒤 한 번만 측정한다.
      if (hoverClampScheduledRef.current) return;
      hoverClampScheduledRef.current = true;
      requestAnimationFrame(() => {
        hoverClampScheduledRef.current = false;
        const scrollEl = scrollRef.current;
        if (scrollEl) clampHoverLayer(scrollEl, graphEl);
      });
    });
    obs.observe(layer, { childList: true, subtree: true });
    hoverObserverRef.current = obs;
  }, [scrollable]);

  useEffect(() => () => hoverObserverRef.current?.disconnect(), []);

  const handleGraphInit = useCallback((_: unknown, graphDiv: HTMLElement) => {
    graphDivRef.current = graphDiv;
    if (scrollable) {
      scheduleStickyRefresh();
      setupHoverClamp(graphDiv);
    }
  }, [scrollable, scheduleStickyRefresh, setupHoverClamp]);

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
