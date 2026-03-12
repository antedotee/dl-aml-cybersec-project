import math

from cps_ad.phase1 import run_phase1


def test_run_phase1_synthetic_end_to_end() -> None:
    results, meta = run_phase1(synthetic=True, random_state=2, gmm_components_search=range(2, 6))
    assert meta["dataset"] == "synthetic_intrusion_tabular"
    assert meta["synthetic"] is True
    assert meta["gmm_selected_k"] >= 2
    assert len(results) == 4
    for r in results:
        assert not math.isnan(r.test.roc_auc)
