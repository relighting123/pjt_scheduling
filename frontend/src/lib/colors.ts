export function buildColorMap(keys: string[], palette: string[]): Record<string, string> {
  const unique = [...new Set(keys)].sort();
  return Object.fromEntries(
    unique.map((k, i) => [k, palette[i % palette.length]]),
  );
}
