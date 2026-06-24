import { useEffect, useState } from "react";

interface Props {
  message: string;
  variant?: "err" | "warn";
}

function previewLine(message: string): string {
  const line = message.split("\n").find((l) => l.trim()) ?? message;
  return line.length > 140 ? `${line.slice(0, 140)}…` : line;
}

export default function ExpandableErrorBanner({ message, variant = "err" }: Props) {
  const [open, setOpen] = useState(true);
  const lines = message.split("\n").filter((l) => l.trim());
  const expandable = lines.length > 1 || message.length > 140;

  useEffect(() => {
    setOpen(true);
  }, [message]);

  const bannerClass = variant === "warn" ? "banner banner-warn" : "banner banner-err";

  if (!expandable) {
    return <div className={bannerClass}>{message}</div>;
  }

  return (
    <div className={`${bannerClass} expandable-banner${open ? " open" : ""}`}>
      <button
        type="button"
        className="expandable-banner-toggle"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
      >
        <span className="expandable-banner-text">
          {open ? (
            <span className="expandable-banner-body">{message}</span>
          ) : (
            <span className="expandable-banner-preview">{previewLine(message)}</span>
          )}
        </span>
        <span className="expandable-banner-action">
          {open ? "▲ 접기" : "▼ 클릭하여 상세 보기"}
        </span>
      </button>
    </div>
  );
}
