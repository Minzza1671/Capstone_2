"""risk_engine 순수 로직 검증. 실행: python -m pytest cctv_crowd_dashboard/tests"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.services.risk_engine import (
    RiskEngine, density_level, weidmann_expected_speed, weidmann_level,
)


def test_density_level_bins():
    # 임계 = Fruin 위험선 × 0.80(margin). <1.6/1.6~2.4/2.4~3.2/3.2~4.0/4.0+
    assert density_level(1.0) == "NORMAL"
    assert density_level(2.0) == "CAUTION"
    assert density_level(3.0) == "WARNING"
    assert density_level(3.5) == "DANGER"
    assert density_level(4.0) == "CRITICAL"


def test_alarm_margin_applied():
    # Fruin DANGER 생값 4.0/㎡는 margin으로 경보 3.2에서 이미 DANGER.
    from server.services.risk_engine import ALARM_MARGIN, DENSITY_REF
    assert density_level(DENSITY_REF["DANGER"] * ALARM_MARGIN) == "DANGER"
    assert density_level(DENSITY_REF["DANGER"]) == "CRITICAL"


def test_weidmann_speed_monotonic():
    # 밀도 높을수록 기대속도 감소, ρmax 도달 시 ~0.
    assert weidmann_expected_speed(0.5) > weidmann_expected_speed(3.0)
    assert weidmann_expected_speed(3.0) > weidmann_expected_speed(5.0)
    assert weidmann_expected_speed(5.4) == 0.0


def test_weidmann_low_density_ignores_speed():
    # 1.5명/㎡ 자유보행 구간 → 속도 0이어도 NORMAL.
    assert weidmann_level(1.5, 0.0) == "NORMAL"


def test_free_flow_normal():
    r = RiskEngine().compute(
        {"global_density": 1.0, "local_max_density": 1.2, "mean_speed": 1.3})
    assert r["risk_level"] == "NORMAL"


def test_local_cluster_floors_global():
    # 전역은 한산(0.3)인데 격자 한 셀이 5명/㎡ → 국소 floor로 CRITICAL.
    r = RiskEngine().compute(
        {"global_density": 0.3, "local_max_density": 5.2, "mean_speed": 1.0})
    assert r["local_level"] == "CRITICAL"
    assert r["risk_level"] == "CRITICAL"


def test_jam_precursor_escalates():
    # 중밀도 2.5명/㎡(density만으론 CAUTION)인데 속도 급락(0.05 m/s) =
    # Weidmann 조기정체 탐지 → WARNING 이상 상향. density 단독보다 빠른 경보.
    r = RiskEngine().compute(
        {"global_density": 2.5, "local_max_density": 2.5, "mean_speed": 0.05})
    assert LEVELS_index(r["weidmann_level"]) >= LEVELS_index("WARNING")
    assert LEVELS_index(r["risk_level"]) >= LEVELS_index("WARNING")


def test_high_density_dangerous_regardless_of_speed():
    # 밀도 4명/㎡면 속도 정상이어도 density floor로 DANGER.
    r = RiskEngine().compute(
        {"global_density": 4.0, "local_max_density": 4.0, "mean_speed": 0.16})
    assert LEVELS_index(r["risk_level"]) >= LEVELS_index("DANGER")


def test_global_level_excluded_from_final():
    # #1 픽스: global_level은 출력되나 최종등급=max(local, weidmann)에만 의존.
    from server.services.risk_engine import LEVELS
    r = RiskEngine().compute(
        {"global_density": 1.0, "local_max_density": 3.5, "mean_speed": 1.0})
    assert "global_level" in r  # 추세표시용으로 여전히 제공
    exp = max(LEVELS.index(r["local_level"]), LEVELS.index(r["weidmann_level"]))
    assert LEVELS.index(r["risk_level"]) == exp


def LEVELS_index(level):
    from server.services.risk_engine import LEVELS
    return LEVELS.index(level)
