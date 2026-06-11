"""
FastAPI 서버 — 분석 파이프라인을 실시간 대시보드/스트림에 연결.

  분석기(별도 스레드) ─ iter_points ─► (visual frame, result)
      ├─ MJPEG  /stream   : draw_visual 영상 (수치 없음, CCTV 뷰)
      ├─ WS     /ws       : result 수치 push (대시보드)
      └─ DB     risk_log  : 비동기 이력 기록

실행:  python -m server.main      (기본 test.mp4 + test_roi.json)
"""

import asyncio
import json
import queue
import threading
import time
from pathlib import Path

import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from analyzer.analyzer import build_arg_parser, default_roi_path, iter_points
from server.services.db_logger import DBLogger

ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = Path(__file__).parent / "static"
DB_PATH = ROOT / "analyzer" / "outputs" / "risk.db"

# 공유 상태 (분석 스레드 → 서버)
_latest_jpeg: bytes | None = None
_result_q: "queue.Queue[dict]" = queue.Queue(maxsize=1000)
_latest_result: dict = {}
_clients: set[WebSocket] = set()
_stop = threading.Event()


def _analyzer_loop(args, camera_id: str):
    """분석기를 반복 실행(영상 끝나면 재시작 — 데모 연속 재생)."""
    global _latest_jpeg
    while not _stop.is_set():
        try:
            for _idx, frame, result in iter_points(args):
                if _stop.is_set():
                    return
                ok, buf = cv2.imencode(".jpg", frame,
                                       [cv2.IMWRITE_JPEG_QUALITY, 80])
                if ok:
                    _latest_jpeg = buf.tobytes()
                result["camera_id"] = camera_id
                try:
                    _result_q.put_nowait(result)
                except queue.Full:
                    pass
        except Exception as e:
            print(f"[SERVER] analyzer error: {e}")
            time.sleep(1.0)


async def _drain_results(app: FastAPI):
    """큐 소비 → DB 기록 + WebSocket 브로드캐스트."""
    global _latest_result
    db: DBLogger = app.state.db
    loop = asyncio.get_running_loop()
    while True:
        try:
            result = await loop.run_in_executor(None, _result_q.get, True, 1.0)
        except queue.Empty:
            continue
        db.insert(result, camera_id=result.get("camera_id", "cam_001"))
        payload = {**result, "ts": time.time()}
        _latest_result = payload
        dead = []
        for ws in list(_clients):
            try:
                await ws.send_text(json.dumps(payload))
            except Exception:
                dead.append(ws)
        for ws in dead:
            _clients.discard(ws)


def _make_app() -> FastAPI:
    args = build_arg_parser().parse_args([])
    args.show = False
    roi_path = Path(args.roi) if args.roi else default_roi_path(Path(args.video))
    video_name = Path(args.video).stem

    app = FastAPI(title="Crowd Risk Dashboard")
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.on_event("startup")
    async def _startup():
        if not roi_path.exists():
            print(f"[SERVER] ROI not found: {roi_path} — setup_roi 먼저 실행.")
        app.state.db = DBLogger(DB_PATH, video_name=video_name)
        _stop.clear()
        app.state.thread = threading.Thread(
            target=_analyzer_loop, args=(args, args.camera_id), daemon=True)
        app.state.thread.start()
        app.state.drain = asyncio.create_task(_drain_results(app))
        print(f"[SERVER] up. video={video_name} roi={roi_path.name}")

    @app.on_event("shutdown")
    async def _shutdown():
        _stop.set()
        app.state.drain.cancel()
        app.state.db.close()

    @app.get("/", response_class=HTMLResponse)
    async def index():
        f = STATIC_DIR / "index.html"
        if f.exists():
            return f.read_text(encoding="utf-8")
        return "<h1>Crowd Risk</h1><p>dashboard: static/index.html 없음 (Phase 5)</p>"

    @app.get("/stream")
    async def stream():
        async def gen():
            boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
            while True:
                if _latest_jpeg is not None:
                    yield boundary + _latest_jpeg + b"\r\n"
                await asyncio.sleep(0.04)  # ~25fps 상한
        return StreamingResponse(
            gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.get("/api/history")
    async def history(limit: int = 300):
        return app.state.db.recent(limit=limit)

    @app.get("/api/latest")
    async def latest():
        return _latest_result

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        _clients.add(ws)
        if _latest_result:
            await ws.send_text(json.dumps(_latest_result))
        try:
            while True:
                await ws.receive_text()  # 클라 ping 대기(연결 유지)
        except WebSocketDisconnect:
            pass
        finally:
            _clients.discard(ws)

    return app


app = _make_app()


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
