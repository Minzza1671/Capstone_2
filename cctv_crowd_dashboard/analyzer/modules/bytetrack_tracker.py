from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from ultralytics import YOLO

Point = Tuple[int, int]
BBox = Tuple[int, int, int, int]


@dataclass
class TrackedPerson:
    track_id: int
    point: Point
    bbox: BBox
    confidence: float


class ByteTrackPersonTracker:
    def __init__(self, model_path: str, conf: float = 0.20, imgsz: int = 960, device: str = "cpu", tracker_config: str = "bytetrack.yaml"):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {path}")
        self.model = YOLO(str(path))
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.tracker_config = tracker_config

    def update(self, frame) -> List[TrackedPerson]:
        results = self.model.track(
            source=frame, persist=True, tracker=self.tracker_config,
            conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False,
        )
        tracked: List[TrackedPerson] = []
        if not results or results[0].boxes is None:
            return tracked
        for box in results[0].boxes:
            if box.id is None:
                continue
            track_id = int(box.id[0].detach().cpu().item())
            xyxy_raw = box.xyxy[0].detach().cpu().numpy().tolist()
            x1, y1, x2, y2 = [int(v) for v in xyxy_raw]
            confidence = float(box.conf[0].detach().cpu().item()) if box.conf is not None else 0.0
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            tracked.append(TrackedPerson(track_id, (cx, cy), (x1, y1, x2, y2), confidence))
        return tracked
