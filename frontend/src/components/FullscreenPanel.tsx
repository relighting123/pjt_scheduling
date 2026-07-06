import { useEffect, useRef, useState, type ReactNode } from "react";

interface FullscreenPanelProps {
  /** 헤더에 표시할 제목 */
  title: ReactNode;
  /** 바깥 wrapper에 추가할 클래스 (예: "card gantt-production-panel") */
  className?: string;
  /** 제목 옆(전체화면 버튼 앞)에 표시할 추가 버튼 (예: 엑셀 다운로드) */
  actions?: ReactNode;
  children: ReactNode;
}

/** 카드/차트/테이블 등 개별 항목을 브라우저 전체화면으로 볼 수 있게 감싸는 래퍼. */
export default function FullscreenPanel({ title, className, actions, children }: FullscreenPanelProps) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  useEffect(() => {
    const onChange = () => setIsFullscreen(document.fullscreenElement === ref.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);

  const toggleFullscreen = () => {
    const el = ref.current;
    if (!el) return;
    if (document.fullscreenElement === el) {
      void document.exitFullscreen();
    } else if (el.requestFullscreen) {
      void el.requestFullscreen();
    }
  };

  return (
    <div
      ref={ref}
      className={`fullscreen-panel${isFullscreen ? " is-fullscreen" : ""}${className ? ` ${className}` : ""}`}
    >
      <div className="fullscreen-panel-head">
        <span className="fullscreen-panel-title">{title}</span>
        <div className="fullscreen-panel-actions">
          {actions}
          <button
            type="button"
            className="btn btn-ghost btn-xs fullscreen-toggle-btn"
            onClick={toggleFullscreen}
            title={isFullscreen ? "전체화면 종료" : "전체화면으로 보기"}
          >
            {isFullscreen ? "⤡ 축소" : "⤢ 전체화면"}
          </button>
        </div>
      </div>
      <div className="fullscreen-panel-body">{children}</div>
    </div>
  );
}
