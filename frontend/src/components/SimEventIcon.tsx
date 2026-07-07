import type { ReactNode } from "react";
import type { SimEventKind } from "../types";
import { normalizeSimEventKind, SIM_EVENT_CLASS } from "../lib/simEvents";

interface SimEventIconProps {
  kind: string;
  className?: string;
}

function IconSvg({ children }: { children: ReactNode }) {
  return (
    <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
      {children}
    </svg>
  );
}

const ICONS: Record<SimEventKind, ReactNode> = {
  MOVE_OUT: (
    <IconSvg>
      <path d="M19 19H5V5h6V3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-6h-2v6z" />
      <path d="M15 3h6v6h-2V6.41l-9.29 9.3-1.42-1.42 9.3-9.29H15V3z" />
    </IconSvg>
  ),
  IDLE: (
    <IconSvg>
      <rect x="7" y="6" width="3.5" height="12" rx="1.2" />
      <rect x="13.5" y="6" width="3.5" height="12" rx="1.2" />
    </IconSvg>
  ),
  JOB_ASSIGNED: (
    <IconSvg>
      <path d="M8 5h11a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V7a2 2 0 0 1 2-2zm4.5 3.5v9l6.5-4.5-6.5-4.5z" />
    </IconSvg>
  ),
  CONV_ASSIGNED: (
    <IconSvg>
      <path d="M7 7h10v3l4-4-4-4v3H5v6h2V7zm10 10H7v-3l-4 4 4 4v-3h12v-6h-2v4z" />
    </IconSvg>
  ),
  TOOL_RELEASE: (
    <IconSvg>
      <path d="M22.7 19.3l-3.4-3.4c.6-1 .9-2.1.9-3.3 0-3.3-2.7-6-6-6-1.2 0-2.3.3-3.3.9L7.3 4.3 5.9 5.7l2.5 2.5C7.7 9.1 7 10.5 7 12c0 3.3 2.7 6 6 6 1.5 0 2.9-.7 3.8-1.4l2.5 2.5 1.4-1.4zM12 16c-2.2 0-4-1.8-4-4s1.8-4 4-4 4 1.8 4 4-1.8 4-4 4z" />
    </IconSvg>
  ),
  WIP_INJECT: (
    <IconSvg>
      <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 14h-2v-4H6v-2h4V7h2v4h4v2h-4v4z" />
    </IconSvg>
  ),
  TOOL_OCCUPY: (
    <IconSvg>
      <path d="M22.7 19.3l-3.4-3.4c.6-1 .9-2.1.9-3.3 0-3.3-2.7-6-6-6-1.2 0-2.3.3-3.3.9L7.3 4.3 5.9 5.7l2.5 2.5C7.7 9.1 7 10.5 7 12c0 3.3 2.7 6 6 6 1.5 0 2.9-.7 3.8-1.4l2.5 2.5 1.4-1.4z" />
    </IconSvg>
  ),
  PROCESS_END: (
    <IconSvg>
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
    </IconSvg>
  ),
  IDLE_DECISION: (
    <IconSvg>
      <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-5 14H7v-2h7v2zm3-4H7v-2h10v2zm0-4H7V7h10v2z" />
    </IconSvg>
  ),
  CONV_START: (
    <IconSvg>
      <path d="M12 4V1L8 5l4 4V6c3.31 0 6 2.69 6 6 0 1.01-.25 1.97-.7 2.8l1.46 1.46C19.54 15.03 20 13.57 20 12c0-4.42-3.58-8-8-8zm0 14c-3.31 0-6-2.69-6-6 0-1.01.25-1.97.7-2.8L5.24 7.74C4.46 8.97 4 10.43 4 12c0 4.42 3.58 8 8 8v3l4-4-4-4v3z" />
    </IconSvg>
  ),
  CONV_END: (
    <IconSvg>
      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
    </IconSvg>
  ),
};

const DEFAULT_ICON = (
  <IconSvg>
    <circle cx="12" cy="12" r="4" />
  </IconSvg>
);

export function SimEventIcon({ kind, className = "" }: SimEventIconProps) {
  const normalized = normalizeSimEventKind(kind) as SimEventKind;
  const tone = SIM_EVENT_CLASS[normalized] ?? "evt-default";
  const icon = ICONS[normalized] ?? DEFAULT_ICON;

  return (
    <span className={`evt-icon ${tone} ${className}`.trim()}>
      {icon}
    </span>
  );
}
