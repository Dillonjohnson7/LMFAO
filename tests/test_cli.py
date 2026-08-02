import json

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("av")

from lmfao.cli import main
from lmfao.datasets import read_lerobot_dataset


def test_generate_demo_writes_dataset(tmp_path):
    cfg = {
        "miniworld": {"enabled": True, "n_synthetic": 4, "seed": 7},
        "pipeline": [{"name": "lighting.brightness", "params": {"factor": 1.1}, "probability": 1.0}],
    }
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "out"

    rc = main(["generate", "--demo", "--config", str(cfg_path), "--seed", "42", "--output", str(out)])
    assert rc == 0
    assert (out / "meta" / "info.json").exists()

    back = read_lerobot_dataset(out)
    assert len(back) == 7  # 3 demo real + 4 synthetic
    assert sum(e.is_synthetic for e in back) == 4


def test_generate_disabled_is_pure_adjust(tmp_path):
    cfg = {"miniworld": {"enabled": False}, "pipeline": []}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(cfg))
    out = tmp_path / "out"

    rc = main(["generate", "--demo", "--config", str(cfg_path), "--output", str(out)])
    assert rc == 0
    back = read_lerobot_dataset(out)
    assert len(back) == 3  # only the real demo episodes
    assert all(not e.is_synthetic for e in back)


def test_generate_requires_input_without_demo(tmp_path):
    rc = main(["generate", "--output", str(tmp_path / "out")])
    assert rc == 2  # missing --input
