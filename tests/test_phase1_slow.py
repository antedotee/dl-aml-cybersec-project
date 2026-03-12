import math

import pytest


@pytest.mark.slow
def test_run_phase1_downloads_and_runs() -> None:
    from cps_ad.phase1 import run_phase1

    results, meta = run_phase1(percent10=True, random_state=0)
    assert meta["gmm_selected_k"] >= 2
    assert len(results) == 4
    for r in results:
        assert not math.isnan(r.test.roc_auc)
