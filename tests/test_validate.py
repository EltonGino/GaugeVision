from gaugevision.measurement.validate import (
    run_validation_sweep,
    summarize,
)


def test_validation_sweep_runs_and_produces_low_diameter_error():
    records = run_validation_sweep(designations=("M6",), conditions=("clean",))
    assert len(records) == 2  # 2 estimators x 1 size x 1 condition
    for r in records:
        assert r.diameter_abs_error_mm < 0.5
        assert r.confidence >= 0.0


def test_summarize_aggregates_by_estimator_and_condition():
    records = run_validation_sweep(designations=("M3", "M6"), conditions=("clean", "blur"))
    summaries = summarize(records)
    keys = {(s.estimator_name, s.condition) for s in summaries}
    assert keys == {
        ("PeakPitchEstimator", "clean"),
        ("PeakPitchEstimator", "blur"),
        ("FFTPitchEstimator", "clean"),
        ("FFTPitchEstimator", "blur"),
    }
    for s in summaries:
        assert s.n_cases == 2  # M3, M6
        assert s.diameter_mae_mm < 0.5
