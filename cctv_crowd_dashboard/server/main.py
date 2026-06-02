import asyncio
import json
import sqlite3
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

DB_PATH = Path(__file__).resolve().parents[1] / "analyzer" / "outputs" / "analysis.db"
STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI()
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


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    last_id = 0

    # 초기 데이터 전송
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
