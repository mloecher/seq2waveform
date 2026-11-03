"""Test load() dispatch, DSV discovery and argument validation.

Discovery tests use empty files because paths and arguments are checked before
parsing. Data comparisons use the bundled files or external corpus.
"""

from __future__ import annotations

import numpy as np
import pytest

import seq2waveform as s2w
from conftest import BUNDLED, TESTDATA, datasets, dsv_dir, dsv_paths, seq_path

PREFIX = "SimulationProtocol"
#: Everything a real Siemens export holds beyond the five channels that matter here.
OTHER_CHANNELS = ("ADC", "CON", "ERR", "M0X", "M0Y", "M0Z", "NC1", "RF2", "RF3", "SLL")


def _export(directory, prefix=PREFIX, channels=s2w._DSV_CHANNELS, extras=OTHER_CHANNELS):
    """Touch an export's worth of empty files; returns the directory."""
    directory.mkdir(parents=True, exist_ok=True)
    for channel in tuple(channels) + tuple(extras):
        (directory / f"{prefix}_{channel}.dsv").touch()
    (directory / "SimulationProtocol.pro").touch()
    (directory / "SequenceSimulation.log").touch()
    return directory


# ---------------------------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------------------------


@pytest.mark.no_corpus
def test_a_directory_resolves_to_the_five_channels_in_order(tmp_path):
    got = s2w._dsv_paths(_export(tmp_path / "exp"))
    assert [p.rsplit("_", 1)[1] for p in got] == ["GRX.dsv", "GRY.dsv", "GRZ.dsv", "RFD.dsv",
                                                  "INF.dsv"]


@pytest.mark.no_corpus
def test_the_other_channels_are_ignored(tmp_path):
    """A real export holds ADC, CON, ERR, M0X/Y/Z and more; only five are wanted."""
    got = s2w._dsv_paths(_export(tmp_path / "exp"))
    assert len(got) == 5
    assert not any(ch in p for p in got for ch in OTHER_CHANNELS)


@pytest.mark.no_corpus
@pytest.mark.parametrize("named", ["GRX", "GRY", "INF", "RFD"])
def test_any_member_file_identifies_its_own_set(tmp_path, named):
    directory = _export(tmp_path / "exp")
    assert s2w._dsv_paths(directory / f"{PREFIX}_{named}.dsv") == s2w._dsv_paths(directory)


@pytest.mark.no_corpus
def test_discovery_is_case_insensitive(tmp_path):
    """Siemens writes capitals, but an archive or a rename can arrive with anything."""
    directory = _export(tmp_path / "exp", channels=(), extras=())
    for channel in s2w._DSV_CHANNELS:
        (directory / f"{PREFIX}_{channel}.dsv".lower()).touch()
    assert len(s2w._dsv_paths(directory)) == 5


@pytest.mark.no_corpus
def test_two_exports_in_one_directory_is_an_error_naming_both(tmp_path):
    directory = tmp_path / "exp"
    _export(directory, prefix="scan_a")
    _export(directory, prefix="scan_b")
    with pytest.raises(ValueError, match="holds 2 DSV exports") as exc:
        s2w._dsv_paths(directory)
    assert "scan_a" in str(exc.value) and "scan_b" in str(exc.value)
    # ...and naming a member file is the documented way out of it.
    assert len(s2w._dsv_paths(directory / "scan_b_GRX.dsv")) == 5


@pytest.mark.no_corpus
def test_a_missing_channel_says_which(tmp_path):
    directory = _export(tmp_path / "exp")
    (directory / f"{PREFIX}_RFD.dsv").unlink()
    with pytest.raises(FileNotFoundError, match=r"missing SimulationProtocol_RFD\.dsv"):
        s2w._dsv_paths(directory)


@pytest.mark.no_corpus
def test_a_directory_with_no_export_suggests_naming_the_seq(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="named directly"):
        s2w._dsv_paths(tmp_path / "empty")


# ---------------------------------------------------------------------------------------------
# Dispatch and argument checks, all before any parsing happens
# ---------------------------------------------------------------------------------------------


@pytest.mark.no_corpus
def test_a_missing_path_is_a_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="no such path"):
        s2w.load(tmp_path / "nope.seq")


@pytest.mark.no_corpus
def test_an_unrecognised_suffix_says_what_is_accepted(tmp_path):
    other = tmp_path / "notes.txt"
    other.touch()
    with pytest.raises(ValueError, match=r"expected a \.seq file"):
        s2w.load(other)


@pytest.mark.no_corpus
def test_gamma_is_rejected_for_a_seq_rather_than_ignored(tmp_path):
    """Sequence amplitudes are already in Hz/m."""
    empty = tmp_path / "x.seq"
    empty.touch()
    with pytest.raises(ValueError, match="gamma does not apply"):
        s2w.load(empty, gamma=42.0)


@pytest.mark.no_corpus
@pytest.mark.parametrize("mode", ["pulseq", "centers"])
def test_a_non_dsv_mode_is_rejected_for_a_dsv_export(tmp_path, mode):
    """A DSV export cannot be reconstructed in another mode."""
    with pytest.raises(ValueError, match="does not apply to a DSV export"):
        s2w.load(_export(tmp_path / "exp"), mode=mode)


@pytest.mark.no_corpus
def test_mode_dsv_is_accepted_for_a_dsv_export(tmp_path):
    """The mode check succeeds; the empty file then fails during parsing."""
    with pytest.raises(Exception) as exc:  # noqa: PT011 -- the point is *which* error it is not
        s2w.load(_export(tmp_path / "exp"), mode="dsv")
    assert "does not apply" not in str(exc.value)


# ---------------------------------------------------------------------------------------------
# On the real corpus
# ---------------------------------------------------------------------------------------------

SEQ_NAMES = datasets(".seq")


@pytest.mark.skipif(not SEQ_NAMES, reason=f"no datasets in {TESTDATA} or {BUNDLED}")
@pytest.mark.parametrize("name", SEQ_NAMES)
def test_load_defaults_a_seq_to_dsv_mode(name):
    """load() defaults to the gradient representation shared with DSV exports."""
    wf = s2w.load(seq_path(name))
    assert isinstance(wf, s2w.Sequence)
    assert wf.gradient_mode == "dsv"

    explicit = s2w.Sequence(seq_path(name), gradient_mode="dsv")
    for axis in "xyz":
        got, want = wf.gradient(axis), explicit.gradient(axis)
        assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(got, want))


@pytest.mark.skipif(not SEQ_NAMES, reason=f"no datasets in {TESTDATA} or {BUNDLED}")
@pytest.mark.parametrize("name", SEQ_NAMES)
def test_load_opens_a_real_dsv_export_from_the_directory(name):
    paths = dsv_paths(name)
    if paths is None:
        pytest.skip(f"{name} has no DSV export")
    directory = str(dsv_dir(name))
    wf = s2w.load(directory)
    assert isinstance(wf, s2w.DsvSet)
    assert wf.gamma == s2w.SCAN_GAMMA
    assert wf.duration > 0.0
    # The same object the explicit constructor builds, and discovery found the same five files.
    assert s2w._dsv_paths(directory) == tuple(paths)
    assert wf.duration == s2w.DsvSet(*paths).duration


@pytest.mark.skipif(not SEQ_NAMES, reason=f"no datasets in {TESTDATA} or {BUNDLED}")
def test_both_sources_answer_the_shared_surface(request):
    """Both loaded source types support the shared sampling API."""
    name = next((n for n in SEQ_NAMES if dsv_paths(n) is not None), None)
    if name is None:
        pytest.skip("no dataset has both a .seq and a DSV export")
    for src in (seq_path(name), str(dsv_dir(name))):
        wf = s2w.load(src)
        assert wf.duration > 0.0
        t, v = wf.gradient("z")
        assert len(t) == len(v)
        assert isinstance(wf.adc_events(), list)
        assert len(wf.sampler().gradients(np.array([0.0, 1e-3]))) == 4
