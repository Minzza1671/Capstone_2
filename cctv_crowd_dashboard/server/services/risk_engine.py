from collections import deque


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
        if risk_score < 0.25:   risk_level = "NORMAL"
        elif risk_score < 0.45: risk_level = "CAUTION"
        elif risk_score < 0.65: risk_level = "WARNING"
        elif risk_score < 0.80: risk_level = "DANGER"
        else:                   risk_level = "CRITICAL"
        return {**snapshot, "density_per_m2": density_per_m2,
                "risk_score": risk_score, "risk_level": risk_level}

    @staticmethod
    def _density_score(count: int, area_m2: float):
        if area_m2 > 0:
            d = count / area_m2
            if d < 0.3:    s = d / 0.3 * 0.2
            elif d < 0.7:  s = 0.2 + (d - 0.3) / 0.4 * 0.2
            elif d < 1.2:  s = 0.4 + (d - 0.7) / 0.5 * 0.2
            elif d < 2.0:  s = 0.6 + (d - 1.2) / 0.8 * 0.2
            else:          s = min(0.8 + (d - 2.0) / 2.0 * 0.2, 1.0)
            return (round(d, 3), round(s, 3))
        else:
            if count <= 20:    s = count / 20 * 0.25
            elif count <= 40:  s = 0.25 + (count - 20) / 20 * 0.25
            elif count <= 65:  s = 0.50 + (count - 40) / 25 * 0.25
            else:              s = min(0.75 + (count - 65) / 35 * 0.25, 1.0)
            return (0.0, round(s, 3))

    @staticmethod
    def _flow_score(imbalance: int) -> float:
        if imbalance <= 0:    return 0.0
        elif imbalance <= 3:  return imbalance / 3 * 0.3
        elif imbalance <= 8:  return 0.3 + (imbalance - 3) / 5 * 0.4
        else:                 return min(0.7 + (imbalance - 8) / 7 * 0.3, 1.0)

    @staticmethod
    def _growth_score(history: deque) -> float:
        if len(history) < 10: return 0.0
        delta = history[-1] - history[0]
        return 0.0 if delta <= 0 else min(delta / 20.0, 1.0)
