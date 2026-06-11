"""
영상 → P2PNet(머리점) + Optical Flow(속도) → 위험도 산출 + 시각화.

  python -m analyzer.analyzer --video data/test.mp4 --roi analyzer/configs/test_roi.json

시각화:
  - ROI 폴리곤
  - 격자 셀: 셀 국소밀도 위험등급 색으로 채움 (압사 핫스팟)
  - P2PNet 점: 속한 셀의 위험등급 색
  - HUD: 위험등급(색) + 전역/국소 밀도 + 평균속도 + 인원

위험도 = Green Guide(전역+격자국소) + Weidmann(속도-밀도). risk_engine 참조.
"""

import argparse
import time
from pathlib import Path
from typing import Iterator, List, Tuple

import cv2
import numpy as np

from analyzer.modules.optical_flow import FarnebackFlowAnalyzer
from analyzer.modules.p2pnet_counter import P2PNetCounter
from analyzer.modules.roi_manager import ROIManager
from server.services.risk_engine import RiskEngine, density_level

PROJECT_ROOT = Path(__file__).resolve().parents[1]

Point = Tuple[float, float, float]  # (x, y, score)

# 위험등급별 색 (BGR).
LEVEL_COLORS = {
    "NORMAL":   (0, 200, 0),     # 초록
    "CAUTION":  (0, 255, 255),   # 노랑
    "WARNING":  (0, 165, 255),   # 주황
    "DANGER":   (0, 80, 255),    # 진주황/빨강
    "CRITICAL": (0, 0, 255),     # 빨강
}


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
    parser.add_argument("--debug-hud", action="store_true", default=False,
                        help="영상에 수치 오버레이(디버그). 운영은 시각 레이어만.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--knn-k", type=int, default=8,
                        help="kNN 국소밀도 이웃 수. 주변 k명 기준 밀도. (8 권장)")
    parser.add_argument("--p2p-stride", type=int, default=3,
                        help="Run P2PNet every N frames; reuse points in between")
    return parser


def _load_pipeline(args):
    """비디오 캡처 + P2PNet + ROI 매니저 준비."""
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    roi_path = Path(args.roi) if args.roi else default_roi_path(video_path)
    if not roi_path.exists():
        raise FileNotFoundError(
            f"ROI config not found: {roi_path}\n"
            "Run 'python -m analyzer.setup_roi' first."
        )

    resolved_device = resolve_device(args.device)
    print(f"[INFO] Device   : {resolved_device}")

    roi_manager = ROIManager.from_json(str(roi_path))
    print(f"[INFO] ROI      : {roi_path}")
    if not roi_manager.has_homography:
        print("[WARN] homography 없음 → 밀도/속도 물리단위 산출 불가. "
              "setup_roi로 보정 권장.")
    else:
        print(f"[INFO] Area     : {roi_manager.area_m2:.1f} m^2")

    print("[INFO] Loading P2PNet...")
    counter = P2PNetCounter(
        weight_path=args.p2p_model,
        threshold=args.p2p_threshold,
        device=resolved_device,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    return cap, counter, roi_manager, fps


def build_snapshot(frame_index, roi_pts, roi_manager, mean_speed, var_speed, knn_k):
    """점좌표 + 속도 → risk_engine 입력 snapshot + kNN 밀도 시각화 데이터."""
    count = len(roi_pts)
    area = roi_manager.area_m2 or 0.0
    global_density = count / area if area > 0 else 0.0

    dens = roi_manager.local_density_knn([(p[0], p[1]) for p in roi_pts], k=knn_k)

    # n < k+1 이면 kNN 국소밀도 신뢰 불가(이웃이 멀리서 잡혀 무의미).
    # 희소→밀집 전이 사각지대 방어: 전역 면적밀도로 fallback + 플래그.
    local_reliable = count >= knn_k + 1
    local_max = dens["max_density"] if local_reliable else round(global_density, 3)

    snapshot = {
        "frame_index": frame_index,
        "roi_person_count": count,
        "global_density": round(global_density, 3),
        "local_max_density": local_max,
        "local_reliable": local_reliable,
        "mean_speed": round(mean_speed, 3),
        "var_speed": round(var_speed, 4),
    }
    return snapshot, dens


def draw_visual(frame, roi_manager, roi_pts, dens, result):
    """
    운영용 시각 레이어 (CCTV/MJPEG). 원시 수치 없음 — 글랜스 판정만.
      - ROI 폴리곤 + kNN 국소밀도 점 히트맵(등급색)
      - 위험등급 배지(색+단어)
      - DANGER/CRITICAL 시 화면 테두리 위험색 (관제실 시선 유도)
    수치(밀도/속도/count)는 snapshot으로 대시보드에 전달.
    """
    roi_manager.draw(frame)

    pdens = dens["point_density"]
    if pdens:
        overlay = frame.copy()
        for i, (x, y, _) in enumerate(roi_pts):
            d = pdens[i] if i < len(pdens) else 0.0
            cv2.circle(overlay, (int(x), int(y)), 14,
                       LEVEL_COLORS[density_level(d)], -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    for i, (x, y, _) in enumerate(roi_pts):
        d = pdens[i] if i < len(pdens) else 0.0
        color = LEVEL_COLORS[density_level(d)] if pdens else (0, 0, 255)
        cv2.circle(frame, (int(x), int(y)), 4, color, -1)
        cv2.circle(frame, (int(x), int(y)), 5, (255, 255, 255), 1)

    level = result["risk_level"]
    lc = LEVEL_COLORS[level]
    h, w = frame.shape[:2]

    # 고위험: 테두리 강조
    if level in ("DANGER", "CRITICAL"):
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), lc, 8)

    # 위험등급 배지 (수치 아님 — 판정)
    cv2.rectangle(frame, (0, 0), (230, 40), (0, 0, 0), -1)
    cv2.rectangle(frame, (0, 0), (10, 40), lc, -1)
    cv2.putText(frame, level, (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, lc, 2)
    return frame


def draw_debug_hud(frame, result):
    """디버그 전용 수치 오버레이 (--debug-hud). 운영에선 미사용."""
    y = frame.shape[0]
    cv2.rectangle(frame, (0, y - 78), (360, y), (0, 0, 0), -1)
    cv2.putText(frame, f"count:{result['roi_person_count']} "
                       f"v:{result['mean_speed']:.2f}m/s", (10, y - 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"density g:{result['global_density']:.2f} /m2", (10, y - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    rel = "" if result.get("local_reliable", True) else " (fallback)"
    cv2.putText(frame, f"density l:{result['local_max_density']:.2f} /m2{rel}", (10, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def iter_points(args) -> Iterator[Tuple[int, "np.ndarray", dict]]:
    """
    프레임별 (frame_index, visual_frame_bgr, risk_result) yield. 서버용.
    visual_frame = draw_visual 적용된 운영 시각레이어(수치 없음, MJPEG용).
    result = 수치 전체(WebSocket/대시보드용).
    """
    cap, counter, roi_manager, fps = _load_pipeline(args)
    flow = FarnebackFlowAnalyzer()
    risk = RiskEngine()
    dt = 1.0 / fps

    frame_index = 0
    cached_pts: List[Point] = []
    mean_speed = var_speed = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            mean_speed, var_speed = flow.update(frame, roi_manager, dt)

            if (frame_index - 1) % args.p2p_stride == 0:
                _, _, cached_pts = counter.count_in_roi(frame, roi_manager)

            snapshot, dens = build_snapshot(
                frame_index, cached_pts, roi_manager, mean_speed, var_speed, args.knn_k)
            result = risk.compute(snapshot)
            draw_visual(frame, roi_manager, cached_pts, dens, result)
            yield frame_index, frame, result

            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    finally:
        cap.release()


def run_analyzer(args):
    """단독 실행: 위험도 산출 + 시각화."""
    start_time = time.time()
    frame_index = 0

    cap, counter, roi_manager, fps = _load_pipeline(args)
    print(f"[INFO] Start    : {Path(args.video)}  fps={fps:.1f}")

    flow = FarnebackFlowAnalyzer()
    risk = RiskEngine()
    dt = 1.0 / fps

    cached_pts: List[Point] = []
    mean_speed = var_speed = 0.0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1

            mean_speed, var_speed = flow.update(frame, roi_manager, dt)

            if (frame_index - 1) % args.p2p_stride == 0:
                _, _, cached_pts = counter.count_in_roi(frame, roi_manager)

            snapshot, dens = build_snapshot(
                frame_index, cached_pts, roi_manager, mean_speed, var_speed, args.knn_k)
            result = risk.compute(snapshot)

            if args.show:
                draw_visual(frame, roi_manager, cached_pts, dens, result)
                if args.debug_hud:
                    draw_debug_hud(frame, result)
                cv2.imshow("Crowd Risk", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    print("[INFO] quit key.")
                    break
                if cv2.getWindowProperty("Crowd Risk", cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed.")
                    break

            if frame_index % 30 == 0:
                print(f"[INFO] f={frame_index} risk={result['risk_level']} "
                      f"g={result['global_density']:.2f} l={result['local_max_density']:.2f} "
                      f"v={result['mean_speed']:.2f}")

            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    finally:
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
