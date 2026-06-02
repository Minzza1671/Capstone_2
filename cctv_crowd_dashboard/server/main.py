import asyncio
import json
import queue
import sqlite3
import sys
import threading
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path

ANALYZER_DIR = Path(__file__).resolve().parents[1] / "analyzer"
sys.path.insert(0, str(ANALYZER_DIR))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

DB_PATH = Path(__file__).resolve().parents[1] / "analyzer" / "outputs" / "analysis.db"
STATIC_DIR = Path(__file__).parent / "static"

latest_frame: bytes = None  # 최신 프레임 (GIL로 thread-safe)
raw_queue: queue.Queue = queue.Queue(maxsize=500)


def _set_frame(jpg: bytes):
    global latest_frame
    latest_frame = jpg


# ── 위험도 산출 엔진 ──────────────────────────────────────────
class RiskEngine:
    def __init__(self, area_m2: float = 0.0, growth_window: int = 60):
        self.area_m2 = area_m2
        self._history: deque = deque(maxlen=growth_window)

    def compute(self, snapshot: dict) -> dict:
        count = snapshot["roi_person_count"]
        self._history.append(count)
        density_per_m2, density_s = self._density_score(count, self.area_m2)
        flow_s = self._flow_score(snapshot["flow_imbalance"])
        growth_s = self._growth_score(self._history)
        risk_score = round(0.40 * density_s + 0.35 * flow_s + 0.25 * growth_s, 3)
        if risk_score < 0.25:    risk_level = "NORMAL"
        elif risk_score < 0.45:  risk_level = "CAUTION"
        elif risk_score < 0.65:  risk_level = "WARNING"
        elif risk_score < 0.80:  risk_level = "DANGER"
        else:                    risk_level = "CRITICAL"
        return {**snapshot, "density_per_m2": density_per_m2,
                "risk_score": risk_score, "risk_level": risk_level}

    @staticmethod
    def _density_score(count: int, area_m2: float):
        if area_m2 > 0:
            d = count / area_m2
            if d < 0.3:    s = d / 0.3 * 0.2
            elif d < 0.7:  s = 0.2 + (d - 0.3) / 0.4 * 0.2
            elif d < 1.2:  s = 0.4 + (d - 0.7) / 0.5 * 0.2
            elif d < 2.0:  s = 0.6 + (d - 1.2) / 0.8 * 0.2
            else:          s = min(0.8 + (d - 2.0) / 2.0 * 0.2, 1.0)
            return (round(d, 3), round(s, 3))
        else:
            if count <= 20:    s = count / 20 * 0.25
            elif count <= 40:  s = 0.25 + (count - 20) / 20 * 0.25
            elif count <= 65:  s = 0.50 + (count - 40) / 25 * 0.25
            else:              s = min(0.75 + (count - 65) / 35 * 0.25, 1.0)
            return (0.0, round(s, 3))

    @staticmethod
    def _flow_score(imbalance: int) -> float:
        if imbalance <= 0:    return 0.0
        elif imbalance <= 3:  return imbalance / 3 * 0.3
        elif imbalance <= 8:  return 0.3 + (imbalance - 3) / 5 * 0.4
        else:                 return min(0.7 + (imbalance - 8) / 7 * 0.3, 1.0)

    @staticmethod
    def _growth_score(history: deque) -> float:
        if len(history) < 10: return 0.0
        delta = history[-1] - history[0]
        return 0.0 if delta <= 0 else min(delta / 20.0, 1.0)


# ── ROI에서 area_m2 읽기 ──────────────────────────────────────
def _load_area_m2() -> float:
    try:
        from main_analyzer import build_arg_parser, default_roi_path
        args = build_arg_parser().parse_args([])
        roi_path = default_roi_path(Path(args.video))
        if roi_path.exists():
            import json as _json
            data = _json.loads(roi_path.read_text(encoding="utf-8"))
            return float(data.get("area_m2", 0.0))
    except Exception:
        pass
    return 0.0


def _start_analyzer():
    try:
        from main_analyzer import build_arg_parser, run_analyzer
    except ImportError as e:
        print(f"[SERVER] Analyzer import failed: {e}")
        return

    parser = build_arg_parser()
    args = parser.parse_args([])
    args.show = False
    args.skip_interactive_setup = True

    video_path = Path(args.video)
    roi_path = Path(args.roi) if args.roi else (
        Path(__file__).resolve().parents[1] / "analyzer" / "configs" / f"{video_path.stem}_roi.json"
    )
    if not roi_path.exists():
        print(f"[SERVER] ROI not found: {roi_path}")
        print("[SERVER] Run 'python main_analyzer.py' first to set up ROI.")
        return

    try:
        run_analyzer(args, frame_callback=_set_frame, raw_queue=raw_queue)
    except Exception as e:
        print(f"[SERVER] Analyzer error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    area_m2 = _load_area_m2()
    app.state.risk_engine = RiskEngine(area_m2=area_m2)
    from modules.db_logger import DBLogger
    app.state.db_logger = DBLogger(db_path=DB_PATH, video_name="live")
    threading.Thread(target=_start_analyzer, daemon=True).start()
    asyncio.create_task(_process_raw_queue(app))
    yield
    app.state.db_logger.close()


async def _process_raw_queue(app: FastAPI):
    while True:
        if not raw_queue.empty():
            snap = raw_queue.get_nowait()
            result = app.state.risk_engine.compute(snap)
            app.state.db_logger.insert(
                camera_id=result.get("camera_id", "cam_001"),
                frame_index=result["frame_index"],
                in_count=result["in_count"],
                out_count=result["out_count"],
                roi_person_count=result["roi_person_count"],
                flow_imbalance=result["flow_imbalance"],
                density_per_m2=result["density_per_m2"],
                risk_score=result["risk_score"],
                risk_level=result["risk_level"],
            )
        else:
            await asyncio.sleep(0.033)


app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def query_db(sql: str, params: tuple = ()):
    if not DB_PATH.exists():
        return []
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/")
async def index():
    return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8"))


@app.get("/api/videos")
async def videos():
    rows = query_db("SELECT DISTINCT video_name FROM crowd_log ORDER BY video_name")
    return [r["video_name"] for r in rows]


@app.get("/api/history")
async def history(limit: int = 300, video_name: str = ""):
    if video_name:
        rows = query_db(
            "SELECT * FROM crowd_log WHERE video_name=? ORDER BY id DESC LIMIT ?",
            (video_name, limit),
        )
    else:
        rows = query_db("SELECT * FROM crowd_log ORDER BY id DESC LIMIT ?", (limit,))
    return list(reversed(rows))


@app.get("/snapshot")
async def snapshot():
    if latest_frame is not None:
        return Response(content=latest_frame, media_type="image/jpeg",
                        headers={"Cache-Control": "no-cache"})
    return Response(status_code=204)




@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_id = 0
    rows = query_db("SELECT * FROM crowd_log ORDER BY id DESC LIMIT 300")
    rows = list(reversed(rows))
    if rows:
        last_id = rows[-1]["id"]
        await websocket.send_text(json.dumps({"type": "init", "rows": rows}))
    try:
        while True:
            await asyncio.sleep(0.5)
            new_rows = query_db(
                "SELECT * FROM crowd_log WHERE id > ? ORDER BY id ASC LIMIT 50",
                (last_id,),
            )
            if new_rows:
                last_id = new_rows[-1]["id"]
                await websocket.send_text(json.dumps({"type": "update", "rows": new_rows}))
    except (WebSocketDisconnect, Exception):
        pass
