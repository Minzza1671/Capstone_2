"""
위험도 산출 엔진 (순수 로직, I/O·프레임 의존 없음).

근거 모델 3종:
  1. Green Guide 전역 밀도   - ROI 전체 평균 명/㎡
  2. Green Guide 격자 국소밀도 - 셀별 명/㎡의 최댓값 (압사는 국소현상)
  3. Weidmann 기본도(속도-밀도) - 같은 밀도에서 관측속도 급락 = 정체/압사 전조

최종 등급 = max(전역등급, 국소등급, Weidmann등급). 국소·전역은 floor로 작동.

입력 snapshot(dict):
    global_density   float  ROI 전역 밀도 (명/㎡)
    local_max_density float 격자 셀 밀도 최댓값 (명/㎡)
    mean_speed       float  ROI 평균 보행속도 (m/s, homography 바닥평면 기준)
  (그 외 키는 그대로 통과시켜 결과 dict에 보존)

출력: snapshot + {global_level, local_level, weidmann_level, risk_level, risk_score}
"""

from __future__ import annotations

import math

LEVELS = ["NORMAL", "CAUTION", "WARNING", "DANGER", "CRITICAL"]

# --- 밀도 위험 기준 (명/㎡) ---------------------------------------------------
# Fruin(1971) / Green Guide LOS 군중 밀도 위험선(문헌 생값).
#   2.0 혼잡 시작 · 3.0 이동제약 · 4.0 위험(LOS D/E) · 5.0 압사위험
DENSITY_REF = {"CAUTION": 2.0, "WARNING": 3.0, "DANGER": 4.0, "CRITICAL": 5.0}

# kNN 국소밀도는 현실(Poisson) 분포에서 근사 불편이나 구간따라 최대 ~17% 과소추정
# (특히 밀집 전이 구간). 경보를 문헌 위험선보다 의도적으로 낮춰 이 불확실성 흡수.
# 추정기 자체는 보정하지 않음(격자에선 +27%, 랜덤에선 ≈0 → 패턴의존이라 단일계수 불가).
ALARM_MARGIN = 0.80  # 예: DANGER 4.0/㎡ → 경보 3.2/㎡

# 실제 적용 임계 = 문헌값 × margin. (상한, LEVELS 인덱스)
_DENSITY_BINS = [
    (DENSITY_REF["CAUTION"] * ALARM_MARGIN, 0),    # <1.6 NORMAL
    (DENSITY_REF["WARNING"] * ALARM_MARGIN, 1),    # 1.6~2.4 CAUTION
    (DENSITY_REF["DANGER"] * ALARM_MARGIN, 2),     # 2.4~3.2 WARNING
    (DENSITY_REF["CRITICAL"] * ALARM_MARGIN, 3),   # 3.2~4.0 DANGER
]
FREE_FLOW_MAX = DENSITY_REF["CAUTION"] * ALARM_MARGIN  # 이 미만은 자유보행(속도신호 무시)

# Weidmann(1993) 보행 기본도 파라미터.
WEIDMANN_V0 = 1.34     # 자유보행 속도 (m/s)
WEIDMANN_RHO_MAX = 5.4  # 최대(정지) 밀도 (명/㎡)
WEIDMANN_GAMMA = 1.913


def density_level(density: float) -> str:
    """Green Guide 밀도 → 안전 등급."""
    for upper, idx in _DENSITY_BINS:
        if density < upper:
            return LEVELS[idx]
    return LEVELS[4]  # >=5.0 CRITICAL


def weidmann_expected_speed(density: float) -> float:
    """Weidmann 기본도상 해당 밀도의 기대 보행속도 (m/s)."""
    if density <= 0:
        return WEIDMANN_V0
    inv = 1.0 / density - 1.0 / WEIDMANN_RHO_MAX
    if inv <= 0:
        return 0.0
    return WEIDMANN_V0 * (1.0 - math.exp(-WEIDMANN_GAMMA * inv))


def weidmann_level(density: float, mean_speed: float) -> str:
    """
    관측 속도가 같은 밀도의 기대속도보다 급락하면 정체로 판정.
    자유보행 구간(FREE_FLOW_MAX 미만)에선 속도 신호 무시(NORMAL).
    jam_ratio = 1 - 관측/기대.  높을수록 비정상 정체.

    주의: density는 국소밀도(local_max), 속도는 ROI 평균(mean_speed)을 쓴다 →
    엄밀히는 같은 구역 값이 아님(불일치). 국소밀집 시 평균속도도 낮게 나올
    개연성이 높아 방향은 맞는 '보수적 근사'. 정밀화(구역별 flow 분리)는 후속.
    """
    if density < FREE_FLOW_MAX:
        return LEVELS[0]
    expected = weidmann_expected_speed(density)
    if expected <= 1e-6:
        return LEVELS[4]  # 밀도가 ρmax 도달 = 정지 = 압사
    jam_ratio = 1.0 - min(mean_speed / expected, 1.0)
    if jam_ratio < 0.4:   return LEVELS[0]
    elif jam_ratio < 0.6: return LEVELS[1]
    elif jam_ratio < 0.8: return LEVELS[2]
    elif jam_ratio < 0.9: return LEVELS[3]
    else:                 return LEVELS[4]


def _level_score(level: str) -> float:
    """등급 → 0~1 연속 점수 (대시보드 게이지용)."""
    return LEVELS.index(level) / (len(LEVELS) - 1)


class RiskEngine:
    def __init__(self):
        pass

    def compute(self, snapshot: dict) -> dict:
        g_density = float(snapshot.get("global_density", 0.0))
        l_density = float(snapshot.get("local_max_density", 0.0))
        speed = float(snapshot.get("mean_speed", 0.0))

        # global_level: 전역 밀도 추세 표시용 (대시보드). 최종등급 max에는 미포함 —
        #   local_max >= global 항등 + density_level 단조 → global은 절대 max를 못 이김.
        # 최종등급은 국소밀도(floor) + Weidmann 정체만으로 결정.
        g_level = density_level(g_density)
        l_level = density_level(l_density)
        w_level = weidmann_level(l_density, speed)  # 국소밀도 사용(보수적 근사)

        final_idx = max(LEVELS.index(l_level), LEVELS.index(w_level))
        risk_level = LEVELS[final_idx]

        return {
            **snapshot,
            "global_level": g_level,
            "local_level": l_level,
            "weidmann_level": w_level,
            "risk_level": risk_level,
            "risk_score": round(_level_score(risk_level), 3),
        }
