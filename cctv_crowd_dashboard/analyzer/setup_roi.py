import argparse
import json
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def default_roi_output_path(video_path: Path) -> Path:
    """
    영상별 ROI config 경로를 자동 생성한다.

    예:
    data/E05_008.mp4
    -> analyzer/configs/E05_008_roi.json
    """
    return PROJECT_ROOT / "analyzer" / "configs" / f"{video_path.stem}_roi.json"


def _shoelace(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


class _ClickEditor:
    """클릭으로 점을 찍는 공통 에디터. 좌클릭 추가, 우클릭 취소, R 리셋."""

    def __init__(self, frame, window_name: str, color, max_display_width: int = 1280):
        self.original_frame = frame
        self.window_name = window_name
        self.color = color
        self.points = []

        self.original_height, self.original_width = frame.shape[:2]
        self.scale = 1.0
        if self.original_width > max_display_width:
            self.scale = max_display_width / self.original_width
        self.display_width = int(self.original_width * self.scale)
        self.display_height = int(self.original_height * self.scale)

    def _disp(self, point):
        return int(point[0] * self.scale), int(point[1] * self.scale)

    def _orig(self, point):
        return int(point[0] / self.scale), int(point[1] / self.scale)

    def mouse_callback(self, event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:
            p = self._orig((x, y))
            self.points.append(p)
            print(f"[CLICK] point {len(self.points)} = {p}")
        elif event == cv2.EVENT_RBUTTONDOWN and self.points:
            print(f"[UNDO] removed = {self.points.pop()}")

    def draw(self):
        display = cv2.resize(self.original_frame.copy(),
                             (self.display_width, self.display_height))
        dpts = [self._disp(p) for p in self.points]
        for i, point in enumerate(dpts):
            cv2.circle(display, point, 5, self.color, -1)
            cv2.putText(display, str(i + 1), (point[0] + 6, point[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 1)
        for i in range(len(dpts) - 1):
            cv2.line(display, dpts[i], dpts[i + 1], self.color, 2)
        if len(dpts) >= 3:
            cv2.line(display, dpts[-1], dpts[0], self.color, 2)
        return display

    def collect(self, min_points: int) -> bool:
        """점 수집 루프. Enter로 확정(min_points 이상), ESC 취소."""
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.display_width, self.display_height)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        while True:
            cv2.imshow(self.window_name, self.draw())
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                cv2.destroyWindow(self.window_name)
                return False
            if key == ord("r"):
                self.points = []
                print("[INFO] reset.")
            if key in (13, 10):
                if len(self.points) >= min_points:
                    cv2.destroyWindow(self.window_name)
                    return True
                print(f"[ERROR] need >= {min_points} points (have {len(self.points)}).")


def calibrate_homography(frame, max_display_width: int = 1280):
    """
    바닥평면 기준점 4개 이상을 클릭받고, 각 점의 실세계 좌표(m)를 콘솔 입력받아
    homography(3x3, 픽셀→바닥평면 m)를 계산한다.

    Returns (H_list, pixel_pts) 또는 취소 시 (None, None).
    """
    print("\n[STEP 2/2] Homography 보정")
    print("[INFO] 바닥평면 기준점 4개 이상 클릭 (직사각형 모서리 권장).")
    print("[INFO] Enter: 확정 / R: 리셋 / ESC: 취소")

    editor = _ClickEditor(frame, "Step 2/2 - Homography", (255, 0, 0), max_display_width)
    if not editor.collect(min_points=4):
        print("[INFO] Homography 보정 취소.")
        return None, None

    print(f"\n[INPUT] 클릭한 {len(editor.points)}개 점의 실세계 좌표(m)를 입력하세요.")
    print("[INFO] 한 점이 원점(0,0), 한 축 방향이 양수가 되게 잡으면 편함.")
    world = []
    for i, px in enumerate(editor.points):
        while True:
            try:
                raw = input(f"  점 {i+1} {px} 실세계 (x_m y_m): ").strip().replace(",", " ")
                xs, ys = raw.split()[:2]
                world.append((float(xs), float(ys)))
                break
            except (ValueError, IndexError):
                print("  [ERROR] 'x y' 형식 숫자 2개 입력.")

    pixel = np.float32(editor.points)
    world = np.float32(world)
    if len(pixel) == 4:
        H = cv2.getPerspectiveTransform(pixel, world)
    else:
        H, _ = cv2.findHomography(pixel, world, method=0)
    return H.tolist(), editor.points


def write_roi_config(output_path: Path, camera_id: str, roi_name: str,
                     roi_points, homography):
    """ROI + homography + 자동산출 면적을 json으로 저장."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    area_m2 = 0.0
    if homography is not None and len(roi_points) >= 3:
        H = np.array(homography, dtype=np.float64)
        poly = np.float32(roi_points).reshape(-1, 1, 2)
        ground = cv2.perspectiveTransform(poly, H).reshape(-1, 2)
        area_m2 = round(_shoelace(ground), 2)

    data = {
        "camera_id": camera_id,
        "roi_name": roi_name,
        "area_m2": area_m2,
        "points": [[int(x), int(y)] for x, y in roi_points],
    }
    if homography is not None:
        data["homography"] = homography

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[DONE] ROI saved: {output_path}  (area={area_m2} m^2)")
    print(json.dumps(data, ensure_ascii=False, indent=2))


def run_roi_editor_on_frame(
    *,
    frame,
    video_path: Path,
    output_path: Path | None = None,
    camera_id: str = "cam_001",
    roi_name: str | None = None,
    max_display_width: int = 1280,
) -> Path:
    output = output_path if output_path else default_roi_output_path(video_path)
    name = roi_name if roi_name else f"{video_path.stem}_roi"
    print(f"[INFO] ROI output: {output}")

    print("\n[STEP 1/2] ROI 폴리곤")
    print("[INFO] ROI 꼭짓점 3개 이상 클릭. Enter: 확정 / R: 리셋 / ESC: 취소")
    roi_editor = _ClickEditor(frame, "Step 1/2 - ROI Setup", (0, 255, 255),
                              max_display_width)
    if not roi_editor.collect(min_points=3):
        raise RuntimeError("ROI setup was canceled.")

    homography, _ = calibrate_homography(frame, max_display_width)

    write_roi_config(output, camera_id, name, roi_editor.points, homography)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default=str(PROJECT_ROOT / "data" / "sibuya.mp4"),
                        help="Path to input video")
    parser.add_argument("--output", default=None,
                        help="ROI config path. 기본 analyzer/configs/{video_stem}_roi.json")
    parser.add_argument("--camera-id", default="cam_001")
    parser.add_argument("--roi-name", default=None)
    parser.add_argument("--max-display-width", type=int, default=1280)
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path = Path(args.output) if args.output else default_roi_output_path(video_path)
    roi_name = args.roi_name if args.roi_name else f"{video_path.stem}_roi"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed to read first frame from video.")

    print(f"[INFO] Video: {video_path}")
    run_roi_editor_on_frame(
        frame=frame,
        video_path=video_path,
        output_path=output_path,
        camera_id=args.camera_id,
        roi_name=roi_name,
        max_display_width=args.max_display_width,
    )


if __name__ == "__main__":
    main()
