import asyncio
import json
import queue
import sqlite3
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

# analyzer/ 폴더를 sys.path에 추가해야 analyzer.py import 가능
ANALYZER_DIR = Path(__file__).resolve().parents[1] / "analyzer"
sys.path.insert(0, str(ANALYZER_DIR))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from services.db_logger import DBLogger
from services.risk_engine import RiskEngine

DB_PATH = Path(__file__).resolve().parents[1] / "analyzer" / "outputs" / "analysis.db"
STATIC_DIR = Path(__file__).parent / "static"

latest_frame: bytes = None
raw_queue: queue.Queue = queue.Queue(maxsize=500)


def _set_frame(jpg: bytes):
    global latest_frame
    latest_frame = jpg


def _start_analyzer(args):
    try:
        from analyzer import run_analyzer
    except ImportError as e:
        print(f"[SERVER] Analyzer import failed: {e}")
        return
    try:
        run_analyzer(args, frame_callback=_set_frame, raw_queue=raw_queue)
    except Exception as e:
        print(f"[SERVER] Analyzer error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    from analyzer import build_arg_parser, default_roi_path

    args = build_arg_parser().parse_args([])
    args.show = False

    video_path = Path(args.video)
    roi_path = Path(args.roi) if args.roi else default_roi_path(video_path)

    if not roi_path.exists():
        print(f"[SERVER] ROI not found: {roi_path}")
        print("[SERVER] Run 'python setup_roi.py' first, then restart the server.")
        yield
        return

    area_m2 = float(json.loads(roi_path.read_text(encoding="utf-8")).get("area_m2", 0.0))
    app.state.risk_engine = RiskEngine(area_m2=area_m2)
    app.state.db_logger = DBLogger(db_path=DB_PATH, video_name="live")

    threading.Thread(target=_start_analyzer, args=(args,), daemon=True).start()
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
