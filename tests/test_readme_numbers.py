"""The headline numbers the README prints are the ones the notebooks stored in data/processed/results.json."""
import json
import re

import pytest

from src import config as cfg

LEDGER = cfg.DATA_PROC / "results.json"
README = cfg.ROOT / "README.md"


def _duck_evening(L):
    w = next(x for x in L["nb08"]["conditional_reserve_windows"] if x["regime"] == "DUCK" and x["hours"] == "16-20")
    return w["shortfall_flat"], w["shortfall_conditional"]


# (what the README prints, how it is computed from the ledger)
HEADLINES = [
    ("{:,.0f} MW", lambda L: L["nb04"]["per_horizon_skill"]["24"]["model_mae"]),          # day-ahead MAE, common rows
    ("{:.1f}%", lambda L: 100 * L["nb04"]["per_horizon_skill"]["24"]["skill_vs_naive"]),  # vs seasonal naive
    ("{:.1f}%", lambda L: 100 * L["nb04"]["skill_vs_df_composite"]),                      # vs the EIA composite
    ("{:.1f}%", lambda L: 100 * L["nb04"]["per_horizon_skill"]["24"]["skill_vs_v2"]),     # vs the strengthened composite
    ("{:.1f}%", lambda L: 100 * L["nb06"]["coverage"]["cqr 90%"]),                         # 90% band coverage
    ("{:.3f}", lambda L: L["nb05"]["ari_stability"]),                                     # regime stability (ARI)
    ("{:.1f}%", lambda L: 100 * _duck_evening(L)[0]),                                     # duck-evening shortfall, flat
    ("{:.1f}%", lambda L: 100 * _duck_evening(L)[1]),                                     # ... and conditional
]


@pytest.mark.skipif(not LEDGER.exists(), reason="no results.json")
@pytest.mark.parametrize("fmt,value", HEADLINES)
def test_readme_headline_matches_the_ledger(fmt, value):
    L = json.loads(LEDGER.read_text(encoding="utf-8"))
    printed = fmt.format(value(L))
    text = README.read_text(encoding="utf-8")
    assert re.search(rf"(?<![\d.,]){re.escape(printed)}", text), f"README does not show {printed}"
