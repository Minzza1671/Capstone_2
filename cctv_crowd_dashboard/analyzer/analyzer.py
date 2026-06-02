import argparse
import time
from collections import deque
from pathlib import Path

import cv2

from modules.bytetrack_tracker import ByteTrackPersonTracker, HeadDetector
from modules.roi_manager import ROIManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_roi_path(video_path: Path) -> Path:
    return PROJECT_ROOT / "analyzer" / "configs" / f"{video_path.stem}_roi.json"


def default_tracker_config_path() -> str:
    custom_path = PROJECT_ROOT / "analyzer" / "configs" / "bytetrack_custom.yaml"
    if custom_path.exists():
        return str(custom_path)
    return "bytetrack.yaml"


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


class ROIFlowCounter:
    def __init__(self, cooldown_frames: int = 20, recent_window_frames: int = 150):
        self.cooldown_frames = cooldown_frames
        self.recent_window_frames = recent_window_frames
        self.total_in = 0
        self.total_out = 0
        self.previous_inside = {}
        self.last_cross_frame = {}
        self.recent_events = deque()

    def update(self, tracked_persons, roi_manager: ROIManager, frame_index: int):
        events = []
        for person in tracked_persons:
            track_id = person.track_id
            curr_inside = roi_manager.contains_point(person.point)
            if track_id not in self.previous_inside:
                self.previous_inside[track_id] = curr_inside
                continue
            prev_inside = self.previous_inside[track_id]
            self.previous_inside[track_id] = curr_inside
            if prev_inside == curr_inside:
                continue
            last_frame = self.last_cross_frame.get(track_id, -999999)
            if frame_index - last_frame < self.cooldown_frames:
                continue
            if not prev_inside and curr_inside:
                count_type = "IN"
                self.total_in += 1
            elif prev_inside and not curr_inside:
                count_type = "OUT"
                self.total_out += 1
            else:
                continue
            self.last_cross_frame[track_id] = frame_index
            event = {"frame_index": frame_index, "track_id": track_id, "count_type": count_type}
            events.append(event)
            self.recent_events.append(event)
        while self.recent_events and frame_index - self.recent_events[0]["frame_index"] > self.recent_window_frames:
            self.recent_events.popleft()
        return events

    def get_summary(self, current_roi_person_count: int):
        recent_in = sum(1 for e in self.recent_events if e["count_type"] == "IN")
        recent_out = sum(1 for e in self.recent_events if e["count_type"] == "OUT")
        return {
            "total_in": self.total_in,
            "total_out": self.total_out,
            "flow_imbalance": recent_in - recent_out,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-model", default=str(PROJECT_ROOT / "models" / "ccrv_head_v5.pt"))
    parser.add_argument("--body-model", default=str(PROJECT_ROOT / "models" / "body_wider_labeling.pt"))
    parser.add_argument("--video", default=str(PROJECT_ROOT / "data" / "test.mp4"))
    parser.add_argument("--camera-id", default="cam_001")
    parser.add_argument("--max-display-width", type=int, default=1280)
    parser.add_argument("--roi", default=None)
    parser.add_argument("--head-conf", type=float, default=0.20)
    parser.add_argument("--body-conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-show", dest="show", action="store_false", default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tracker-config", default=None)
    parser.add_argument("--flow-recent-window", type=int, default=150)
    parser.add_argument("--cross-cooldown", type=int, default=20)
    return parser


def run_analyzer(args, frame_callback=None, raw_queue=None):
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    roi_path = Path(args.roi) if args.roi else default_roi_path(video_path)
    if not roi_path.exists():
        raise FileNotFoundError(
            f"ROI config not found: {roi_path}\n"
            "Run 'python setup_roi.py' first to configure the ROI."
        )

    tracker_config = args.tracker_config if args.tracker_config else default_tracker_config_path()
    resolved_device = resolve_device(args.device)

    print(f"[INFO] Device   : {resolved_device}")
    print(f"[INFO] Tracker  : {tracker_config}")

    roi_manager = ROIManager.from_json(str(roi_path))
    print(f"[INFO] ROI      : {roi_path}")

    body_tracker = ByteTrackPersonTracker(
        body_model_path=args.body_model, conf=args.body_conf,
        imgsz=args.imgsz, device=resolved_device, tracker_config=tracker_config,
    )
    head_detector = HeadDetector(
        head_model_path=args.head_model, conf=args.head_conf,
        imgsz=args.imgsz, device=resolved_device,
    )
    flow_counter = ROIFlowCounter(
        cooldown_frames=args.cross_cooldown,
        recent_window_frames=args.flow_recent_window,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_index = 0
    start_time = time.time()
    print(f"[INFO] Start    : {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        all_tracks = body_tracker.update(frame)
        all_heads = head_detector.detect(frame)
        roi_tracks = [p for p in all_tracks if roi_manager.contains_point(p.point)]
        roi_heads = roi_manager.filter_detections(all_heads, point_attr="center")

        flow_counter.update(tracked_persons=all_tracks, roi_manager=roi_manager, frame_index=frame_index)
        roi_person_count = max(len(roi_tracks), len(roi_heads))
        flow_summary = flow_counter.get_summary(current_roi_person_count=roi_person_count)

        if raw_queue is not None:
            try:
                raw_queue.put_nowait({
                    "camera_id": args.camera_id,
                    "frame_index": frame_index,
                    "roi_person_count": roi_person_count,
                    "in_count": flow_summary["total_in"],
                    "out_count": flow_summary["total_out"],
                    "flow_imbalance": flow_summary["flow_imbalance"],
                })
            except Exception:
                pass

        need_draw = args.show or frame_callback is not None
        if need_draw:
            roi_manager.draw(frame)
            for person in roi_tracks:
                x1, y1, x2, y2 = person.bbox
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
            for head in roi_heads:
                x1, y1, x2, y2 = head.xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

            if frame_callback is not None:
                _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                frame_callback(jpg.tobytes())

            if args.show:
                cv2.imshow("CCTV Crowd Analyzer", frame)
                cv2.waitKey(1)
                if cv2.getWindowProperty("CCTV Crowd Analyzer", cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed.")
                    break

        if args.max_frames > 0 and frame_index >= args.max_frames:
            break

        if frame_index % 30 == 0:
            print(
                f"[INFO] frame={frame_index}"
                f"  persons={roi_person_count}"
                f"  in={flow_summary['total_in']}"
                f"  out={flow_summary['total_out']}"
                f"  imbalance={flow_summary['flow_imbalance']}"
            )

    cap.release()
    if args.show:
        cv2.destroyAllWindows()
    elapsed = time.time() - start_time
    print(f"[DONE] frames={frame_index}, elapsed={elapsed:.2f}s")


def main():
    args = build_arg_parser().parse_args()
    run_analyzer(args)


if __name__ == "__main__":
    main()
