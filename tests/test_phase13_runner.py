from edge_validation.phase13_runner import run_phase13, DEFAULT_METRICS


def test_phase13_runner_deterministic():
    # Run twice with the same metrics and assert identical structured output (except timestamp)
    r1 = run_phase13(metrics=DEFAULT_METRICS, save_json_path=None, console=False)
    r2 = run_phase13(metrics=DEFAULT_METRICS, save_json_path=None, console=False)

    # generated_at will differ; remove for deterministic comparison
    r1.pop('generated_at', None)
    r2.pop('generated_at', None)

    assert r1 == r2
    assert 'decisions' in r1 and 'capital_report' in r1
    # ensure metrics snapshot preserved and sorted keys
    assert list(r1['metrics_snapshot'].keys()) == sorted(DEFAULT_METRICS.keys())
