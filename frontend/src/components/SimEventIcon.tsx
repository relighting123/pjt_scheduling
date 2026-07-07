import type { IconType } from "react-icons";
import {
  PiArrowSquareOutFill,
  PiArrowsClockwiseFill,
  PiArrowUUpLeftFill,
  PiCheckCircleFill,
  PiCircleFill,
  PiClipboardTextFill,
  PiPauseCircleFill,
  PiPlayCircleFill,
  PiPlusCircleFill,
  PiWrenchFill,
} from "react-icons/pi";
import type { SimEventKind } from "../types";
import { normalizeSimEventKind, SIM_EVENT_CLASS } from "../lib/simEvents";

interface SimEventIconProps {
  kind: string;
  className?: string;
}

const ICONS: Record<SimEventKind, IconType> = {
  MOVE_OUT: PiArrowSquareOutFill,
  IDLE: PiPauseCircleFill,
  JOB_ASSIGNED: PiPlayCircleFill,
  CONV_ASSIGNED: PiArrowsClockwiseFill,
  TOOL_RELEASE: PiArrowUUpLeftFill,
  WIP_INJECT: PiPlusCircleFill,
  TOOL_OCCUPY: PiWrenchFill,
  PROCESS_END: PiCheckCircleFill,
  IDLE_DECISION: PiClipboardTextFill,
  CONV_START: PiArrowsClockwiseFill,
  CONV_END: PiCheckCircleFill,
};

export function SimEventIcon({ kind, className = "" }: SimEventIconProps) {
  const normalized = normalizeSimEventKind(kind) as SimEventKind;
  const tone = SIM_EVENT_CLASS[normalized] ?? "evt-default";
  const Icon = ICONS[normalized] ?? PiCircleFill;

  return (
    <span className={`evt-icon ${tone} ${className}`.trim()}>
      <Icon aria-hidden="true" />
    </span>
  );
}
