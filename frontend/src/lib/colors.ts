export const PROD_COLORS = [
  "#4C72B0", "#DD8452", "#55A868", "#C44E52",
  "#8172B3", "#937860", "#DA8BC3", "#CCB974",
  "#64B5CD", "#76B7B2",
];

export const OPER_BORDER_COLORS = [
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
  "#bcbd22", "#17becf", "#aec7e8", "#ffbb78",
  "#98df8a", "#ff9896", "#c5b0d5",
];

export function buildColorMap(keys: string[], palette: string[]): Record<string, string> {
  const unique = [...new Set(keys)].sort();
  return Object.fromEntries(
    unique.map((k, i) => [k, palette[i % palette.length]]),
  );
}

export function parseCompletedKey(key: string): [string, string] {
  const parts = key.split("|");
  return [parts[0] ?? "", parts[1] ?? ""];
}
