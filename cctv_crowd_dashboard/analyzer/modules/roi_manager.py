import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]


def _polygon_area(poly: np.ndarray) -> float:
    """Shoelace 면적. poly: (N,2)."""
    x = poly[:, 0]
    y = poly[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


@dataclass
class ROIConfig:
    camera_id: str
    roi_name: str
    points: List[Point]
    area_m2: float = 0.0
    homography: list | None = None  # 3x3 픽셀→바닥평면(m) 변환행렬


class ROIManager:
    """
    Polygon ROI 관리 모듈.

    points가 비어 있거나 3개 미만이면 전체 화면을 ROI로 본다.
    points가 3개 이상이면 polygon ROI로 판단한다.
    """

    def __init__(self, config: ROIConfig):
        self.config = config
        self.points = config.points

        if len(self.points) >= 3:
            self.polygon = np.array(self.points, dtype=np.int32)
        else:
            self.polygon = None

        if config.homography is not None:
            self.H = np.array(config.homography, dtype=np.float64)
            self.H_inv = np.linalg.inv(self.H)  # 바닥평면 → 픽셀 (격자 시각화용)
        else:
            self.H = None
            self.H_inv = None

        # 바닥평면 폴리곤 캐시 (면적 산출용, homography 있을 때만)
        self._ground_polygon = None
        if self.H is not None and self.polygon is not None:
            self._ground_polygon = self.image_to_ground(self.polygon.astype(np.float64))

    @classmethod
    def from_json(cls, path: str):
        path_obj = Path(path)

        if not path_obj.exists():
            raise FileNotFoundError(f"ROI config not found: {path_obj}")

        with path_obj.open("r", encoding="utf-8") as f:
            data = json.load(f)

        raw_points = data.get("points", [])
        points: List[Point] = []

        for p in raw_points:
            if len(p) != 2:
                continue
            points.append((int(p[0]), int(p[1])))

        config = ROIConfig(
            camera_id=data.get("camera_id", "cam_001"),
            roi_name=data.get("roi_name", "default_roi"),
            points=points,
            area_m2=float(data.get("area_m2", 0.0)),
            homography=data.get("homography"),
        )

        return cls(config)

    @property
    def area_m2(self) -> float:
        # homography 있으면 바닥평면 폴리곤 면적을 우선 사용(수동입력 대체).
        if self._ground_polygon is not None:
            return _polygon_area(self._ground_polygon)
        return self.config.area_m2

    @property
    def has_homography(self) -> bool:
        return self.H is not None

    @property
    def is_full_frame(self) -> bool:
        return self.polygon is None

    def image_to_ground(self, pts) -> np.ndarray:
        """픽셀좌표 (N,2) → 바닥평면 좌표(m) (N,2). homography 필수."""
        if self.H is None:
            raise RuntimeError("homography not set")
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(arr, self.H)
        return out.reshape(-1, 2)

    def ground_to_image(self, pts) -> np.ndarray:
        """바닥평면 좌표(m) (N,2) → 픽셀좌표 (N,2). homography 필수."""
        if self.H_inv is None:
            raise RuntimeError("homography not set")
        arr = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(arr, self.H_inv)
        return out.reshape(-1, 2)

    MIN_SPACING_M = 0.3  # 사람 머리 최소 간격(m). 중복검출 시 밀도 폭발 차단.

    def local_density_knn(self, image_points, k: int = 8):
        """
        kNN 국소밀도 (Helbing·Moussaïd 방식). 임의 격자 없음.

        각 머리점에서 k번째 가까운 이웃까지 거리 r_k(바닥평면 m)로
            rho_i = k / (pi * r_k^2)   [명/㎡]
        를 산출. r_k = 실제 사람 간격이라 밀집할수록 rho 커짐.

        k=8 권장: 면적밀도와 스케일 일치(검증), 중복검출에 견고.
        작은 k는 근접점 1~2개에 r_k→0 → 밀도 폭발하므로 비권장.
        r_k는 MIN_SPACING_M로 하한 클램프(중복검출 방어).

        Returns dict:
            max_density   float        rho_i 최댓값 (국소 위험)
            point_density List[float]  점마다 rho_i (점 색칠용, 입력 순서)
            point_radius  List[float]  점마다 r_k (m, 시각화용)
        homography 없거나 점 부족(<2)이면 0/빈값.
        """
        empty = {"max_density": 0.0, "point_density": [], "point_radius": []}
        n = len(image_points)
        if self.H is None or n < 2:
            return empty

        ground = self.image_to_ground(image_points)  # (N,2) m
        kk = min(k, n - 1)  # 이웃 수는 자기 제외 최대 N-1

        # 쌍거리 행렬 (N 작음 → O(N^2) 무방)
        diff = ground[:, None, :] - ground[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))  # (N,N), 대각=0
        dist.sort(axis=1)  # 행별 오름차순, [:,0]=자기(0)
        r_k = np.maximum(dist[:, kk], self.MIN_SPACING_M)  # 하한 클램프

        rho = kk / (np.pi * r_k ** 2)
        return {
            "max_density": round(float(rho.max()), 3),
            "point_density": [round(float(v), 3) for v in rho],
            "point_radius": [round(float(v), 3) for v in r_k],
        }

    def contains_point(self, point: Sequence[int]) -> bool:
        if self.is_full_frame:
            return True

        x, y = int(point[0]), int(point[1])
        result = cv2.pointPolygonTest(self.polygon, (x, y), False)

        return result >= 0

    def draw(self, frame):
        if self.is_full_frame:
            return frame

        overlay = frame.copy()

        cv2.fillPoly(
            overlay,
            [self.polygon],
            color=(0, 255, 255),
        )

        alpha = 0.08
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.polylines(
            frame,
            [self.polygon],
            isClosed=True,
            color=(0, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        return frame

    def to_dict(self):
        return {
            "camera_id": self.config.camera_id,
            "roi_name": self.config.roi_name,
            "points": [[x, y] for x, y in self.points],
            "is_full_frame": self.is_full_frame,
        }
