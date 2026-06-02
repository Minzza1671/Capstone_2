import sqlite3
from pathlib import Path


class DBLogger:
    BATCH_SIZE = 30

    def __init__(self, db_path: Path, video_name: str):
        self.video_name = video_name
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self._batch = []
        self._init_schema()

    def _init_schema(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS crowd_log (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id        TEXT    NOT NULL DEFAULT 'cam_001',
                video_name       TEXT    NOT NULL,
                frame_index      INTEGER NOT NULL,
                timestamp        DATETIME DEFAULT CURRENT_TIMESTAMP,
                in_count         INTEGER NOT NULL,
                out_count        INTEGER NOT NULL,
                roi_person_count INTEGER NOT NULL,
                flow_imbalance   INTEGER NOT NULL DEFAULT 0,
                density_per_m2   REAL    NOT NULL DEFAULT 0.0,
                risk_score       REAL    NOT NULL DEFAULT 0.0,
                risk_level       TEXT    NOT NULL DEFAULT 'NORMAL'
            )
        """)
        self.conn.commit()

    def insert(self, frame_index: int, in_count: int, out_count: int,
               roi_person_count: int, flow_imbalance: int = 0,
               density_per_m2: float = 0.0, risk_score: float = 0.0,
               risk_level: str = "NORMAL", camera_id: str = "cam_001"):
        self._batch.append((
            camera_id, self.video_name, frame_index,
            in_count, out_count, roi_person_count,
            flow_imbalance, density_per_m2, risk_score, risk_level,
        ))
        if len(self._batch) >= self.BATCH_SIZE:
            self._flush()

    def _flush(self):
        if not self._batch:
            return
        self.conn.executemany(
            """INSERT INTO crowd_log
               (camera_id, video_name, frame_index, in_count, out_count,
                roi_person_count, flow_imbalance, density_per_m2, risk_score, risk_level)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            self._batch,
        )
        self.conn.commit()
        self._batch = []

    def close(self):
        self._flush()
        self.conn.close()
