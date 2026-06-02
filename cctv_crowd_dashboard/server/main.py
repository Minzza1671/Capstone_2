import asyncio
import json
import queue
import sqlite3
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

DB_PATH = Path(__file__).resolve().parents[1] / "analyzer" / "outputs" / "analysis.db"
STATIC_DIR = Path(__file__).parent / "static"
ANALYZER_DIR = Path(__file__).resolve().parents[1] / "analyzer"

frame_queue: queue.Queue = queue.Queue(maxsize=2)


def _start_analyzer():
    sys.path.insert(0, str(ANALYZER_DIR))
    try:
        from main_analyzer import build_arg_parser, run_analyzer
    except ImportError as e:
        print(f"[SERVER] Analyzer import failed: {e}")
        return

    parser = build_arg_parser()
    args = parser.parse_args([])
    args.show = False

    from pathlib import Path as _Path
    video_path = _Path(args.video)
    roi_path = _Path(args.roi) if args.roi else (
        _Path(__file__).resolve().parents[1] / "analyzer" / "configs" / f"{video_path.stem}_roi.json"
    )

    if not roi_path.exists():
        print(f"[SERVER] ROI config not found: {roi_path}")
        print("[SERVER] Run 'python main_analyzer.py' first to set up ROI, then restart the server.")
        return

    args.skip_interactive_setup = True

    try:
        run_analyzer(args, frame_queue=frame_queue)
    except Exception as e:
        print(f"[SERVER] Analyzer error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    t = threading.Thread(target=_start_analyzer, daemon=True)
    t.start()
    yield


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
        rows = query_db(
            "SELECT * FROM crowd_log ORDER BY id DESC LIMIT ?",
            (limit,),
        )
    return list(reversed(rows))


@app.get("/video_feed")
async def video_feed():
    async def generate():
        while True:
            if not frame_queue.empty():
                jpg = frame_queue.get_nowait()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            else:
                await asyncio.sleep(0.033)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


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
