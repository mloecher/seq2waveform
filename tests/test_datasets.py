"""Check that bundled datasets are found and match any external corpus copies.

These tests fail on missing bundled data rather than silently skipping.
"""

import pytest

from conftest import BUNDLED, TESTDATA, dataset_root, datasets, h5_path, seq_path

#: The bundled datasets, read off the directory rather than hard-coded, so adding one needs no
#: edit here -- but the test still fails if the directory itself goes missing.
BUNDLED_NAMES = sorted(d.name for d in BUNDLED.iterdir() if d.is_dir()) if BUNDLED.is_dir() else []


@pytest.mark.no_corpus
def test_the_bundled_datasets_are_discoverable():
    """`tests/data/` must be found, with a `.seq` and an `.h5` per dataset.

    The one test in the suite that cannot skip. If it fails, the rest of the run is skips and
    means nothing, whatever colour it comes out.
    """
    assert BUNDLED_NAMES, f"no bundled datasets under {BUNDLED}; the suite can only skip now"

    found = datasets(".seq", ".h5")
    missing = [n for n in BUNDLED_NAMES if n not in found]
    assert not missing, (
        f"{missing} are in {BUNDLED} but discovery did not find them with both a .seq and an "
        f".h5; corpus resolved to {TESTDATA}"
    )


@pytest.mark.no_corpus
def test_the_bundled_datasets_carry_no_dsv_export():
    """A DSV export is a vendor simulator's output and must not reach the repo.

    `conftest` promises the bundled sets have none, `test_golden` excuses the DSV columns on that
    basis, and the reason for the promise is a licensing one -- so it is asserted, not assumed.
    """
    for name in BUNDLED_NAMES:
        stray = sorted(p.name for p in (BUNDLED / name).rglob("*.dsv"))
        assert not stray, f"{name}: {stray} should not be committed"


@pytest.mark.parametrize("name", BUNDLED_NAMES)
def test_the_bundled_copy_matches_the_corpus(name):
    """A bundled file must be byte-identical to the corpus file it was copied from.

    Only runs where both exist. The corpus wins the lookup, so locally the bundled copy is never
    read and could drift unnoticed until CI compares it against a snapshot regenerated from the
    corpus. Compared as bytes: these are inputs, so anything but equality is drift.
    """
    if dataset_root(name) == BUNDLED:
        pytest.skip(f"{name} is not in the corpus at {TESTDATA}, so there is nothing to compare")

    for path in (seq_path(name), str(h5_path(name))):
        suffix = path.rsplit(".", 1)[-1]
        mine = BUNDLED / name / f"{name}.{suffix}"
        assert mine.read_bytes() == open(path, "rb").read(), (
            f"{mine} differs from the corpus copy at {path}; re-copy it, and regenerate the "
            f"snapshot if the corpus file changed on purpose"
        )
