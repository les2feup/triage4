import pandas as pd
import pytest

from assessment.triage4_wcrt import (
    AlarmScenario,
    compute_wcrt_alarm,
    compute_wcrt_alarm_aap,
    compute_wcrt_strict_emergency,
    evaluate_scenarios,
    load_results_csv,
    lookup_observed_delay,
)


def test_compute_wcrt_alarm_matches_formula():
    C = 20.0
    sigma_A = 10.0
    rho_A = 5.0
    result = compute_wcrt_alarm(C, sigma_A, rho_A)
    expected = (sigma_A + 1.0) / C + rho_A / (C ** 2)
    assert result == pytest.approx(expected)


def test_compute_wcrt_alarm_aap_matches_formula():
    C = 20.0
    b_P = 45.0
    r_P = 15.0
    result = compute_wcrt_alarm_aap(C, b_P, r_P)
    expected = (b_P + 1.0) / C + r_P / (C ** 2)
    assert result == pytest.approx(expected)


def test_compute_wcrt_strict_emergency_with_interference():
    C = 20.0
    sigma_E = 2.0
    rho_E = 3.0
    interferers = [(1.0, 2.0), (0.5, 1.0)]
    result = compute_wcrt_strict_emergency(C, sigma_E, rho_E, interferers)
    sigma_total = sigma_E + sum(pair[0] for pair in interferers)
    rho_total = rho_E + sum(pair[1] for pair in interferers)
    expected = (sigma_total + 1.0) / C + rho_total / (C ** 2)
    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "C,sigma,rho",
    [
        (0.0, 1.0, 0.5),
        (10.0, -1.0, 0.5),
        (10.0, 1.0, -0.5),
        (10.0, 1.0, 10.0),
    ],
)
def test_compute_wcrt_alarm_validation(C, sigma, rho):
    with pytest.raises(ValueError):
        compute_wcrt_alarm(C, sigma, rho)


def test_compute_wcrt_strict_emergency_validation_total_rate():
    with pytest.raises(ValueError):
        compute_wcrt_strict_emergency(5.0, 1.0, 4.0, [(1.0, 1.0)])


def test_evaluate_scenarios_prints(capsys):
    scenarios = [
        AlarmScenario(name="Base", C=20.0, sigma_A=10.0, rho_A=5.0, observed_delay_s=0.4),
        AlarmScenario(
            name="AAP", C=20.0, b_P=45.0, r_P=15.0, baseline="TRIAGE/4-AAP"
        ),
        AlarmScenario(
            name="Strict",
            C=20.0,
            sigma_A=8.0,
            rho_A=4.0,
            interferers=[(5.0, 3.0)],
            baseline="STRICT",
        ),
    ]
    evaluate_scenarios(scenarios)
    output = capsys.readouterr().out
    assert "Base" in output
    assert "0.4000" in output
    assert "AAP" in output
    assert "Strict" in output


def test_load_results_csv_uses_scenario_key(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "scenario_key": "Base_TRIAGE4",
                "scheduler": "TRIAGE/4",
                "alarm_p95_latency_mean": 0.4,
                "alarm_p95_latency_ci_upper": 0.5,
            }
        ]
    ).to_csv(path, index=False)

    df = load_results_csv(str(path))
    assert "scenario_name" in df.columns
    assert df.loc[0, "scenario_name"] == "Base_TRIAGE4"


def test_lookup_observed_delay_mean_and_ci(tmp_path):
    path = tmp_path / "results.csv"
    pd.DataFrame(
        [
            {
                "scenario_name": "Base_TRIAGE4",
                "scheduler": "TRIAGE/4",
                "alarm_p95_latency_mean": 0.4,
                "alarm_p95_latency_ci_upper": 0.5,
            },
            {
                "scenario_name": "Strict_Emergency",
                "scheduler": "Strict",
                "alarm_p95_latency_mean": 0.7,
                "alarm_p95_latency_ci_upper": 0.8,
            },
        ]
    ).to_csv(path, index=False)

    df = load_results_csv(str(path))
    base = AlarmScenario(
        name="Base_TRIAGE4", C=20.0, sigma_A=10.0, rho_A=5.0, baseline="TRIAGE/4"
    )
    strict = AlarmScenario(
        name="Strict_Emergency", C=20.0, sigma_A=10.0, rho_A=5.0, baseline="STRICT"
    )

    mean_val = lookup_observed_delay(df, base, metric="p95_mean")
    ci_val = lookup_observed_delay(df, strict, metric="p95_ci_upper")

    assert mean_val == pytest.approx(0.4)
    assert ci_val == pytest.approx(0.8)
