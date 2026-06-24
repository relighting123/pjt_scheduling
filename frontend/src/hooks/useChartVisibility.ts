
import { useCallback, useEffect, useMemo, useState } from "react";

const STORAGE_PREFIX = "pjt-chart-visibility:";

function loadVisibility(key: string, ids: string[]): Record<string, boolean> {
  const defaults = Object.fromEntries(ids.map((id) => [id, true]));
  try {
    const raw = localStorage.getItem(STORAGE_PREFIX + key);
    if (!raw) return defaults;
    const parsed = JSON.parse(raw) as Record<string, boolean>;
    return { ...defaults, ...parsed };
  } catch {
    return defaults;
  }
}

export function useChartVisibility(storageKey: string, chartIds: string[]) {
  const idsKey = useMemo(() => chartIds.join("|"), [chartIds]);
  const [visibility, setVisibility] = useState<Record<string, boolean>>(() =>
    loadVisibility(storageKey, chartIds),
  );

  useEffect(() => {
    setVisibility((prev) => {
      const next = loadVisibility(storageKey, chartIds);
      return { ...next, ...prev };
    });
  }, [storageKey, idsKey, chartIds]);

  useEffect(() => {
    localStorage.setItem(STORAGE_PREFIX + storageKey, JSON.stringify(visibility));
  }, [storageKey, visibility]);

  const setVisible = useCallback((id: string, visible: boolean) => {
    setVisibility((prev) => ({ ...prev, [id]: visible }));
  }, []);

  const toggle = useCallback((id: string) => {
    setVisibility((prev) => ({ ...prev, [id]: !prev[id] }));
  }, []);

  const showAll = useCallback(() => {
    setVisibility(Object.fromEntries(chartIds.map((id) => [id, true])));
  }, [chartIds]);

  const hideAll = useCallback(() => {
    setVisibility(Object.fromEntries(chartIds.map((id) => [id, false])));
  }, [chartIds]);

  const isVisible = useCallback(
    (id: string) => visibility[id] !== false,
    [visibility],
  );

  return { visibility, setVisible, toggle, showAll, hideAll, isVisible };
}

export interface ChartVisibilityItem {
  id: string;
  title: string;
}
