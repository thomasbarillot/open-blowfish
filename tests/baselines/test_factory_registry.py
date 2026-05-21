"""Phase 2 — BaselineHooks registry."""

from blowfish.baselines import ALL_BASELINE_IDS, BaselineHooks


def test_factory_has_all_ten_baselines():
    for bid in ALL_BASELINE_IDS:
        cls = getattr(BaselineHooks, bid)
        assert cls.name == bid


def test_baseline_names_unique():
    names = [getattr(BaselineHooks, bid).name for bid in ALL_BASELINE_IDS]
    assert len(set(names)) == len(names)
