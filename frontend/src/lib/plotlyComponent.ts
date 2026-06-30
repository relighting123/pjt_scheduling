import createPlotlyComponent from "react-plotly.js/factory";
import type { ComponentType } from "react";

/** 동적 로드되는 Plotly 번들 + react wrapper 컴포넌트 */
export interface PlotlyBundle {
  Plot: ComponentType<Record<string, unknown>>;
  Plotly: {
    relayout: (el: HTMLElement, update: Record<string, unknown>) => Promise<unknown>;
    Fx: { unhover: (el: HTMLElement) => void };
    [key: string]: unknown;
  };
}

let _promise: Promise<PlotlyBundle> | null = null;

/**
 * Plotly(약 5MB)를 첫 차트 렌더 시점에 동적 import 한다.
 * - 초기 앱 번들에서 분리(코드 스플리팅) → 첫 로딩 속도 개선
 * - 싱글톤 프로미스로 1회만 로드해 모든 차트가 공유
 */
export function loadPlotly(): Promise<PlotlyBundle> {
  if (!_promise) {
    // @ts-expect-error dist/plotly has no separate typings
    _promise = import("plotly.js/dist/plotly").then((mod) => {
      const Plotly = (mod.default ?? mod) as PlotlyBundle["Plotly"];
      const Plot = createPlotlyComponent(Plotly) as unknown as PlotlyBundle["Plot"];
      return { Plot, Plotly };
    });
  }
  return _promise;
}
