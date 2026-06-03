import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from modules.p2pnet_counter import P2PNetCounter
from modules.optical_flow import FarnebackFlowAnalyzer
from modules.roi_manager import ROIManager

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_roi_path(video_path: Path) -> Path:
    return PROJECT_ROOT / "analyzer" / "configs" / f"{video_path.stem}_roi.json"


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:
        pass
    return "cpu"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2p-model", default=str(PROJECT_ROOT / "models" / "best_mae.pth"))
    parser.add_argument("--p2p-threshold", type=float, default=0.50)
    parser.add_argument("--video", default=str(PROJECT_ROOT / "data" / "test.mp4"))
    parser.add_argument("--camera-id", default="cam_001")
    parser.add_argument("--roi", default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--no-show", dest="show", action="store_false", default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--p2p-stride", type=int, default=3,
                        help="Run P2PNet every N frames; reuse count in between")
    return parser


def run_analyzer(args, frame_callback=None, raw_queue=None):
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    roi_path = Path(args.roi) if args.roi else default_roi_path(video_path)
    if not roi_path.exists():
        raise FileNotFoundError(
            f"ROI config not found: {roi_path}\n"
            "Run 'python setup_roi.py' first."
        )

    resolved_device = resolve_device(args.device)
    print(f"[INFO] Device   : {resolved_device}")

    roi_manager = ROIManager.from_json(str(roi_path))
    roi_polygon = np.array(roi_manager.points, dtype=np.int32) if roi_manager.points else None
    print(f"[INFO] ROI      : {roi_path}")

    print("[INFO] Loading P2PNet...")
    counter = P2PNetCounter(
        weight_path=args.p2p_model,
        threshold=args.p2p_threshold,
        device=resolved_device,
    )

    flow_analyzer = FarnebackFlowAnalyzer()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    frame_index = 0
    cached_roi_count = 0
    cached_roi_pts = []
    has_valid_count = False
    start_time = time.time()
    print(f"[INFO] Start    : {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_index += 1

        if frame_index % args.p2p_stride == 1 or args.p2p_stride == 1:
            cached_roi_count, _, cached_roi_pts = counter.count_in_roi(frame, roi_manager)
            has_valid_count = True

        roi_count = cached_roi_count
        roi_pts = cached_roi_pts
        flow_imbalance = flow_analyzer.update(frame, roi_polygon)

        if raw_queue is not None and has_valid_count:
            try:
                raw_queue.put_nowait({
                    "camera_id": args.camera_id,
                    "frame_index": frame_index,
                    "roi_person_count": roi_count,
                    "in_count": 0,
                    "out_count": 0,
                    "flow_imbalance": flow_imbalance,
                })
            except Exception:
                pass

        need_draw = args.show or frame_callback is not None
        if need_draw:
            roi_manager.draw(frame)

            for x, y, _ in roi_pts:
                cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)
                cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 1)

            if args.show:
                lines = [
                    f"frame : {frame_index}",
                    f"count : {roi_count}",
                    f"flow  : {flow_imbalance}",
                ]
                for i, text in enumerate(lines):
                    cv2.putText(frame, text, (10, 28 + i * 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)

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
            print(f"[INFO] frame={frame_index}  count={roi_count}  flow={flow_imbalance}")

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
