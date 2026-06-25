const MS_PER_MIN = 60_000;

/** API sim_base_time 또는 RULE_TIMEKEY → epoch ms */
export function parseSimBaseMs(simBaseTime?: string): number | null {
  if (!simBaseTime) return null;
  const normalized = simBaseTime.includes("T")
    ? simBaseTime
    : simBaseTime.replace(" ", "T");
  const ms = Date.parse(normalized);
  return Number.isNaN(ms) ? null : ms;
}

/** dataset 경로에서 RULE_TIMEKEY(14자리) 추출 */
export function ruleTimekeyFromFolder(folder: string): string | null {
  for (const part of folder.split(/[/\\]/)) {
    if (/^\d{14}$/.test(part)) return part;
    if (/^\d{8}$/.test(part)) return `${part}070000`;
  }
  return null;
}

/** RULE_TIMEKEY → ISO 로컬 시각 문자열 */
export function simBaseTimeFromRuleTimekey(rtk: string): string {
  const key = rtk.length === 8 ? `${rtk}070000` : rtk;
  const y = key.slice(0, 4);
  const m = key.slice(4, 6);
  const d = key.slice(6, 8);
  const h = key.slice(8, 10);
  const mi = key.slice(10, 12);
  const s = key.slice(12, 14) || "00";
  return `${y}-${m}-${d}T${h}:${mi}:${s}`;
}

export function minutesToTimestamp(minutes: number, baseMs: number): number {
  return baseMs + minutes * MS_PER_MIN;
}

export function minutesToDurationMs(minutes: number): number {
  return minutes * MS_PER_MIN;
}

export function formatSimClock(minutes: number, baseMs: number): string {
  const d = new Date(minutesToTimestamp(minutes, baseMs));
  const hh = d.getHours().toString().padStart(2, "0");
  const mm = d.getMinutes().toString().padStart(2, "0");
  return `${hh}:${mm}`;
}

export function formatGanttMinuteLabel(minutes: number, baseMs: number | null): string {
  return baseMs == null ? `${minutes}분` : formatSimClock(minutes, baseMs);
}

export function ganttBarAxisCoords(
  startMin: number,
  widthMin: number,
  baseMs: number | null,
): { base: number; x: number } {
  if (baseMs == null) {
    return { base: startMin, x: widthMin };
  }
  return {
    base: minutesToTimestamp(startMin, baseMs),
    x: minutesToDurationMs(widthMin),
  };
}

export function ganttAxisValue(minutes: number, baseMs: number | null): number {
  return baseMs == null ? minutes : minutesToTimestamp(minutes, baseMs);
}

export function ganttXMinClamp(baseMs: number | null, minMinutes = 0): number {
  return baseMs == null ? minMinutes : minutesToTimestamp(minMinutes, baseMs);
}

export function ganttTickFormat(rangeMinutes: number): string {
  return rangeMinutes > 24 * 60 ? "%m/%d %H:%M" : "%H:%M";
}
