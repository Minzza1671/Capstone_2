"""
위험도 결과 SQLite 로깅 (이력/사후분석용).

risk_engine.compute() 결과 dict를 그대로 받아 배치 기록. WAL 모드.
실시간 경로를 막지 않도록 BATCH_SIZE마다 flush, close 시 잔여 flush.
"""

import sqlite3
from pathlib import Path


class DBLogger:
    BATCH_SIZE = 30

    _COLUMNS = [
        "roi_person_count", "global_density", "local_max_density",
        "local_reliable", "mean_speed", "var_speed",
        "global_level", "local_level", "weidmann_level",
        "risk_level", "risk_score",
    ]

    def __init__(self, db_path: Path, video_name: str = "live"):
        self.video_name = video_name
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._batch = []
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id         TEXT    NOT NULL DEFAULT 'cam_001',
                video_name        TEXT    NOT NULL,
                frame_index       INTEGER NOT NULL,
                timestamp         DATETIME DEFAULT CURRENT_TIMESTAMP,
                roi_person_count  INTEGER NOT NULL DEFAULT 0,
                global_density    REAL    NOT NULL DEFAULT 0.0,
                local_max_density REAL    NOT NULL DEFAULT 0.0,
                local_reliable    INTEGER NOT NULL DEFAULT 1,
                mean_speed        REAL    NOT NULL DEFAULT 0.0,
                var_speed         REAL    NOT NULL DEFAULT 0.0,
                global_level      TEXT    NOT NULL DEFAULT 'NORMAL',
                local_level       TEXT    NOT NULL DEFAULT 'NORMAL',
                weidmann_level    TEXT    NOT NULL DEFAULT 'NORMAL',
                risk_level        TEXT    NOT NULL DEFAULT 'NORMAL',
                risk_score        REAL    NOT NULL DEFAULT 0.0
            )
        """)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_risk_log_ts ON risk_log(timestamp)")
        self.conn.commit()

    def insert(self, result: dict, camera_id: str = "cam_001"):
        """risk_engine.compute() 결과 dict를 배치에 추가."""
        row = [camera_id, self.video_name, int(result.get("frame_index", 0))]
        for col in self._COLUMNS:
            v = result.get(col, 0)
            if col == "local_reliable":
                v = int(bool(v))
            row.append(v)
        self._batch.append(tuple(row))
        if len(self._batch) >= self.BATCH_SIZE:
            self._flush()

    def _flush(self):
        if not self._batch:
            return
        cols = ", ".join(["camera_id", "video_name", "frame_index"] + self._COLUMNS)
        ph = ", ".join(["?"] * (3 + len(self._COLUMNS)))
        self.conn.executemany(
            f"INSERT INTO risk_log ({cols}) VALUES ({ph})", self._batch)
        self.conn.commit()
        self._batch = []

    def recent(self, limit: int = 300, camera_id: str | None = None):
        """최근 기록 조회 (대시보드 추세/이력용). 시간 오름차순 반환."""
        q = "SELECT * FROM risk_log"
        params = []
        if camera_id:
            q += " WHERE camera_id = ?"
            params.append(camera_id)
        q += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        cur = self.conn.execute(q, params)
        cols = [c[0] for c in cur.description]
        rows = [dict(zip(cols, r)) for r in cur.fetchall()]
        return list(reversed(rows))

    def close(self):
        self._flush()
        self.conn.close()
