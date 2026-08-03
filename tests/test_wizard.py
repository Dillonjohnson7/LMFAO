import builtins

import pytest

from lmfao.wizard import _STEP_AUGS, _build_sweep, parse_hf_link


def test_build_sweep_bidirectional_counts_and_magnitudes():
    specs, mags = _build_sweep("lighting.brightness", _STEP_AUGS["lighting.brightness"], 3)
    assert len(specs) == 6  # 3 steps x 2 directions
    assert mags == ["±5%", "±10%", "±15%"]
    factors = sorted(s["pipeline"][0]["params"]["factor"] for s in specs)
    assert factors == [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]


def test_build_sweep_unidirectional_noise():
    specs, mags = _build_sweep("noise.gaussian", _STEP_AUGS["noise.gaussian"], 3)
    assert len(specs) == 3  # magnitude-only, one per step
    assert mags == ["2%", "4%", "6%"]


@pytest.mark.parametrize("text, expected", [
    ("Dillonjohnson/pick_place_v2", "Dillonjohnson/pick_place_v2"),
    ("https://huggingface.co/datasets/Dillonjohnson/pick_place_v2", "Dillonjohnson/pick_place_v2"),
    ("huggingface.co/datasets/lerobot/aloha", "lerobot/aloha"),
    ("https://huggingface.co/datasets/owner/name?ref=main", "owner/name"),
    ("hf.co/datasets/a/b", "a/b"),
])
def test_parse_hf_link_accepts_valid_refs(text, expected):
    assert parse_hf_link(text) == expected


@pytest.mark.parametrize("text", ["demo", "not a link", "a/b/c", "", "justone"])
def test_parse_hf_link_rejects_non_refs(text):
    assert parse_hf_link(text) is None


def _script(answers):
    """Return an input() replacement that yields the scripted answers in order."""
    it = iter(answers)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:  # pragma: no cover - means the flow asked more than expected
            raise EOFError from None

    return fake_input


def test_wizard_drives_augment_on_demo(tmp_path, monkeypatch):
    pytest.importorskip("pyarrow")
    pytest.importorskip("av")
    from lmfao.datasets import read_lerobot_dataset
    from lmfao.wizard import run_wizard

    out = tmp_path / "wiz"
    answers = [
        "demo",        # data source
        "1",           # operation: augment
        str(out),      # output dir
        "5",           # seed
        "1,3",         # effects: brightness, color_temperature
        "2",           # brightness steps -> 4 videos (±5%, ±10%)
        "1",           # color_temperature steps -> 2 videos (±0.2)
        "n",           # keep originals?
        "0",           # limit (all)
        "y",           # run it?
    ]
    monkeypatch.setattr(builtins, "input", _script(answers))
    rc = run_wizard()
    assert rc == 0
    back = read_lerobot_dataset(out)
    # 3 demo episodes x (4 brightness + 2 color) = 18
    assert len(back) == 18
    assert all(e.metadata.get("augmented") for e in back)
    assert {e.metadata["sweep"] for e in back} >= {"lighting.brightness +5%", "lighting.color_temperature +0.2"}


def test_wizard_cancels_cleanly_on_eof(monkeypatch):
    from lmfao.wizard import run_wizard

    def raise_eof(prompt=""):
        raise EOFError

    monkeypatch.setattr(builtins, "input", raise_eof)
    assert run_wizard() == 130
