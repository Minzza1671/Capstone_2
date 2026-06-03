from collections import deque

LEVELS = ["NORMAL", "CAUTION", "WARNING", "DANGER", "CRITICAL"]


class RiskEngine:
    def __init__(self, area_m2: float = 0.0, growth_window: int = 60):
        self.area_m2 = area_m2
        self._history: deque = deque(maxlen=growth_window)

    def compute(self, snapshot: dict) -> dict:
        count = snapshot["roi_person_count"]
        self._history.append(count)
        density_per_m2, density_s = self._density_score(count, self.area_m2)
        flow_s = self._flow_score(snapshot["flow_imbalance"])
        growth_s = self._growth_score(self._history)
        risk_score = round(0.40 * density_s + 0.35 * flow_s + 0.25 * growth_s, 3)
        if risk_score < 0.25:   score_level = "NORMAL"
        elif risk_score < 0.45: score_level = "CAUTION"
        elif risk_score < 0.65: score_level = "WARNING"
        elif risk_score < 0.80: score_level = "DANGER"
        else:                   score_level = "CRITICAL"
        # 안전 override: 압사는 단일 신호(고밀도)만으로 발생 → 가중합이 희석 못 하게
        # density 룰 등급을 floor 로 강제. final = max(가중합 등급, density 등급).
        density_level = self._density_level(density_per_m2)
        risk_level = LEVELS[max(LEVELS.index(score_level),
                                LEVELS.index(density_level))]
        return {**snapshot, "density_per_m2": density_per_m2,
                "risk_score": risk_score, "score_level": score_level,
                "density_level": density_level, "risk_level": risk_level}

    @staticmethod
    def _density_level(density_per_m2: float) -> str:
        """
        행안부 인파 밀집 기준 → 안전 등급 floor (명/㎡).
        가중합과 무관하게 이 등급 이상은 보장. area 미입력(d=0) 시 NORMAL.
        d<1 NORMAL · 1~2 CAUTION · 2~3 WARNING · 3~4 DANGER · 4+ CRITICAL
        """
        d = density_per_m2
        if d < 1.0:   return "NORMAL"
        elif d < 2.0: return "CAUTION"
        elif d < 3.0: return "WARNING"
        elif d < 4.0: return "DANGER"
        else:         return "CRITICAL"

    @staticmethod
    def _density_score(count: int, area_m2: float):
        """
        행안부 인파 밀집 기준 적용 (명/㎡).
        area_m2 미입력(=0) 시 density_score = 0.0 (무시).

        0   ~ 1.0: 여유          → score 0.00 ~ 0.20
        1.0 ~ 2.0: 붐빔          → score 0.20 ~ 0.40
        2.0 ~ 3.0: 불쾌한 밀집  → score 0.40 ~ 0.65
        3.0 ~ 4.0: 위험 (경보)  → score 0.65 ~ 0.85
        4.0+     : 압사 위험     → score 0.85 ~ 1.00
        """
        if area_m2 <= 0:
            return (0.0, 0.0)
        d = count / area_m2
        if d < 1.0:   s = d / 1.0 * 0.20
        elif d < 2.0: s = 0.20 + (d - 1.0) / 1.0 * 0.20
        elif d < 3.0: s = 0.40 + (d - 2.0) / 1.0 * 0.25
        elif d < 4.0: s = 0.65 + (d - 3.0) / 1.0 * 0.20
        else:         s = min(0.85 + (d - 4.0) / 2.0 * 0.15, 1.0)
        return (round(d, 3), round(s, 3))

    @staticmethod
    def _flow_score(convergence: int) -> float:
        """
        수렴도 기반 흐름 위험도.
        convergence = optical_flow.py 에서 0-100 정수로 정규화된 수렴 강도.
        0   = 발산/정지 (위험 아님)
        100 = 최대 수렴 (군중이 중심으로 강하게 밀림)
        """
        return min(convergence / 100.0, 1.0)

    @staticmethod
    def _growth_score(history: deque) -> float:
        """
        최근 growth_window 프레임 동안 인원 증가율.
        5프레임 미만은 데이터 부족으로 0.0 반환.
        """
        if len(history) < 5:
            return 0.0
        delta = history[-1] - history[0]
        return 0.0 if delta <= 0 else min(delta / 20.0, 1.0)
