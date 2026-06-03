from __future__ import annotations

import cv2
import numpy as np


class FarnebackFlowAnalyzer:
    """
    ROI 내 군중 수렴도(convergence) 측정.

    수렴도 = ROI 중심 방향으로의 평균 흐름 성분.
    양수(수렴) = 군중이 안쪽으로 밀림 → 위험
    음수(발산) = 군중이 흩어짐 → 안전
    정지      = 0에 가까움

    자동 보정: 초기 CALIBRATION_FRAMES 동안 관측값 수집 →
    95th percentile을 최대 기준으로 설정.
    그 이후부터 0-100 정수로 정규화해 반환.
    """

    CALIBRATION_FRAMES = 60

    def __init__(self):
        self._prev_gray = None
        self._cal_samples: list = []
        self._scale: float | None = None

    def update(self, frame_bgr, roi_polygon) -> int:
        """
        Returns convergence score (0-100 int).
        0 = 발산/정지, 100 = 최대 수렴.
        보정 기간(초기 60프레임) 동안은 0 반환.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None:
            self._prev_gray = gray
            return 0

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        self._prev_gray = gray

        convergence = self._compute_convergence(flow, roi_polygon)

        if self._scale is None:
            self._cal_samples.append(abs(convergence))
            if len(self._cal_samples) >= self.CALIBRATION_FRAMES:
                p95 = float(np.percentile(self._cal_samples, 95))
                self._scale = max(p95, 0.1)
            return 0

        norm = max(0.0, convergence) / self._scale
        return int(min(norm, 1.0) * 100)

    @staticmethod
    def _compute_convergence(flow, roi_polygon) -> float:
        """
        ROI 중심으로 향하는 평균 흐름 성분.
        """
        if roi_polygon is None or len(roi_polygon) < 3:
            return 0.0

        h, w = flow.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [roi_polygon], 255)

        cx = float(roi_polygon[:, 0].mean())
        cy = float(roi_polygon[:, 1].mean())

        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return 0.0

        # 각 픽셀 → 중심 방향 단위 벡터
        to_cx = cx - xs.astype(float)
        to_cy = cy - ys.astype(float)
        norms = np.maximum(np.sqrt(to_cx ** 2 + to_cy ** 2), 1e-6)
        to_cx /= norms
        to_cy /= norms

        # flow: (u=x방향, v=y방향) displacement
        u = flow[ys, xs, 0]
        v = flow[ys, xs, 1]

        # 중심 방향 성분 (양수=수렴, 음수=발산)
        return float((u * to_cx + v * to_cy).mean())

    def reset(self):
        self._prev_gray = None
        self._cal_samples = []
        self._scale = None
