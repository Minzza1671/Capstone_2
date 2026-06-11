"""
ROI 내 군중 보행속도 측정 (Farneback dense optical flow).

이전 버전의 '수렴도(중심방향)' 휴리스틱을 폐기하고,
homography로 픽셀 변위를 바닥평면(m)에 매핑해 실제 속도(m/s)를 낸다.

반환: (mean_speed, var_speed) — Weidmann 기본도 입력.
var_speed 는 향후 Moussaïd 국소 난류(turbulence) 지표용으로 같이 적립.
"""

from __future__ import annotations

import cv2
import numpy as np


class FarnebackFlowAnalyzer:
    SAMPLE_STRIDE = 8  # ROI 내 픽셀 샘플 간격 (연산량 절감)

    def __init__(self):
        self._prev_gray = None

    def update(self, frame_bgr, roi_manager, dt: float) -> tuple[float, float]:
        """
        Returns (mean_speed_m_s, var_speed_m_s).
        homography 없거나 첫 프레임이면 (0.0, 0.0).
        dt: 직전 프레임과의 시간 간격 (초).
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self._prev_gray is None or not roi_manager.has_homography or dt <= 0:
            self._prev_gray = gray
            return 0.0, 0.0

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2,
            flags=0,
        )
        self._prev_gray = gray

        h, w = gray.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if roi_manager.polygon is not None:
            cv2.fillPoly(mask, [roi_manager.polygon], 255)
        else:
            mask[:] = 255

        s = self.SAMPLE_STRIDE
        ys, xs = np.where(mask[::s, ::s] > 0)
        if len(xs) == 0:
            return 0.0, 0.0
        xs = xs * s
        ys = ys * s

        u = flow[ys, xs, 0]
        v = flow[ys, xs, 1]

        # 시작점·도착점 모두 바닥평면 매핑 → 변위(m).
        start = np.stack([xs, ys], axis=1).astype(np.float64)
        end = np.stack([xs + u, ys + v], axis=1).astype(np.float64)
        g_start = roi_manager.image_to_ground(start)
        g_end = roi_manager.image_to_ground(end)

        disp = np.linalg.norm(g_end - g_start, axis=1)  # m
        speed = disp / dt  # m/s

        return float(np.mean(speed)), float(np.var(speed))

    def reset(self):
        self._prev_gray = None
