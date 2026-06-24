import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ReactNode,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { createRoot, type Root } from "react-dom/client";

const POPOUT_CSS = `
  html, body { margin: 0; height: 100%; background: #f4f6fb; font-family: system-ui, sans-serif; }
  #chart-popout-root { height: 100%; display: flex; flex-direction: column; }
  .chart-popout-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0.55rem 0.85rem; background: #fff;
    border-bottom: 1px solid rgba(148,163,184,0.25);
    font-size: 0.9rem; font-weight: 600; color: #1e293b;
  }
  .chart-popout-body { flex: 1; min-height: 0; padding: 0.5rem; }
  .plot-chart {
    width: 100%; height: 100%; min-height: 420px;
    border-radius: 12px; overflow: hidden; background: #f8f9fc;
    border: 1px solid rgba(148,163,184,0.2);
    box-shadow: 0 8px 24px rgba(15,23,42,0.06);
    padding: 0.35rem; box-sizing: border-box;
  }
`;

interface ChartPanelProps {
  id: string;
  title: string;
  visible?: boolean;
  onVisibleChange?: (visible: boolean) => void;
  className?: string;
  renderChart: () => ReactNode;
}

interface PopoutHandle {
  win: Window;
  root: Root;
}

export default function ChartPanel({
  id,
  title,
  visible = true,
  onVisibleChange,
  className,
  renderChart,
}: ChartPanelProps) {
  const [poppedOut, setPoppedOut] = useState(false);
  const [dragging, setDragging] = useState(false);
  const popoutRef = useRef<PopoutHandle | null>(null);
  const panelRef = useRef<HTMLElement | null>(null);
  const dragRef = useRef<{ x: number; y: number; active: boolean }>({
    x: 0,
    y: 0,
    active: false,
  });

  const closePopout = useCallback(() => {
    const handle = popoutRef.current;
    if (handle && !handle.win.closed) {
      handle.root.unmount();
      handle.win.close();
    }
    popoutRef.current = null;
    setPoppedOut(false);
  }, []);

  const openPopout = useCallback(() => {
    if (popoutRef.current && !popoutRef.current.win.closed) {
      popoutRef.current.win.focus();
      return;
    }

    const win = window.open(
      "",
      `chart_popout_${id}`,
      "width=1024,height=720,menubar=no,toolbar=no,location=no,status=no",
    );
    if (!win) {
      window.alert("팝업이 차단되었습니다. 브라우저에서 팝업을 허용해 주세요.");
      return;
    }

    win.document.title = title;
    win.document.head.innerHTML = `<style>${POPOUT_CSS}</style>`;
    win.document.body.innerHTML =
      '<div id="chart-popout-root"><header class="chart-popout-header"></header><div class="chart-popout-body"></div></div>';

    const header = win.document.querySelector(".chart-popout-header");
    if (header) {
      header.textContent = title;
    }

    const body = win.document.getElementById("chart-popout-root")?.querySelector(".chart-popout-body");
    if (!body) {
      win.close();
      return;
    }

    const mount = win.document.createElement("div");
    mount.className = "plot-chart";
    body.appendChild(mount);

    const root = createRoot(mount);
    root.render(<>{renderChart()}</>);

    popoutRef.current = { win, root };
    setPoppedOut(true);

    win.addEventListener("beforeunload", () => {
      root.unmount();
      popoutRef.current = null;
      setPoppedOut(false);
    });
  }, [id, title, renderChart]);

  useEffect(() => {
    const handle = popoutRef.current;
    if (!poppedOut || !handle || handle.win.closed) return;
    handle.root.render(<>{renderChart()}</>);
  }, [poppedOut, renderChart]);

  useEffect(() => () => closePopout(), [closePopout]);

  const onDragStart = (ev: ReactMouseEvent) => {
    dragRef.current = { x: ev.clientX, y: ev.clientY, active: true };
    setDragging(true);
  };

  useEffect(() => {
    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current.active) return;
      const dx = ev.clientX - dragRef.current.x;
      const dy = ev.clientY - dragRef.current.y;
      if (Math.hypot(dx, dy) > 48) {
        dragRef.current.active = false;
        setDragging(false);
        openPopout();
      }
    };
    const onUp = () => {
      dragRef.current.active = false;
      setDragging(false);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [openPopout]);

  if (!visible) {
    return (
      <section className={`chart-panel chart-panel-hidden ${className ?? ""}`} data-chart-id={id}>
        <div className="chart-panel-hidden-bar">
          <span>{title}</span>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => onVisibleChange?.(true)}
          >
            표시
          </button>
        </div>
      </section>
    );
  }

  return (
    <section
      ref={panelRef}
      className={`chart-panel card ${dragging ? "chart-panel-dragging" : ""} ${poppedOut ? "chart-panel-popped" : ""} ${className ?? ""}`}
      data-chart-id={id}
    >
      <header className="chart-panel-header">
        <button
          type="button"
          className="chart-panel-drag-handle"
          title="드래그하여 별도 창으로 분리"
          aria-label={`${title} 드래그하여 분리`}
          onMouseDown={onDragStart}
        >
          ⠿
        </button>
        <h3 className="chart-panel-title">{title}</h3>
        <div className="chart-panel-actions">
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            title="별도 창으로 열기"
            onClick={openPopout}
          >
            ↗
          </button>
          {poppedOut && (
            <button type="button" className="btn btn-secondary btn-sm" onClick={closePopout}>
              복귀
            </button>
          )}
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            title="숨기기"
            onClick={() => onVisibleChange?.(false)}
          >
            −
          </button>
        </div>
      </header>

      {poppedOut ? (
        <p className="chart-panel-popout-note hint">
          별도 창에서 표시 중입니다.
          <button type="button" className="btn btn-secondary btn-sm" onClick={openPopout}>
            창 포커스
          </button>
          <button type="button" className="btn btn-secondary btn-sm" onClick={closePopout}>
            페이지로 복귀
          </button>
        </p>
      ) : (
        <div className="chart-panel-body">{renderChart()}</div>
      )}
    </section>
  );
}
