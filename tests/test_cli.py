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


def _run(tmp_path, cfg_text, capsys=None):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(cfg_text)
    return main(["generate", "--demo", "--config", str(cfg_path), "--output", str(tmp_path / "out")])


def test_malformed_json_config_is_clean_error(tmp_path, capsys):
    rc = _run(tmp_path, '{"miniworld": {,,}')
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_non_object_config_is_clean_error(tmp_path, capsys):
    rc = _run(tmp_path, "[1, 2, 3]")
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_unknown_toplevel_config_key_is_rejected(tmp_path, capsys):
    rc = _run(tmp_path, json.dumps({"minworld": {"enabled": True}, "pipeline": []}))
    assert rc == 2
    assert "minworld" in capsys.readouterr().err


def test_unknown_miniworld_field_is_clean_error(tmp_path, capsys):
    rc = _run(tmp_path, json.dumps({"miniworld": {"enabled": True, "foo": 1}}))
    assert rc == 2
    assert "foo" in capsys.readouterr().err


def test_unknown_augmenter_is_clean_error(tmp_path, capsys):
    rc = _run(tmp_path, json.dumps({"pipeline": [{"name": "lighting.nope", "params": {}}]}))
    assert rc == 2
    assert "lighting.nope" in capsys.readouterr().err


def test_missing_config_file_is_clean_error(tmp_path, capsys):
    rc = main(["generate", "--demo", "--config", str(tmp_path / "nope.json"),
               "--output", str(tmp_path / "out")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_inspect_summarizes_dataset(tmp_path, capsys):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"miniworld": {"enabled": True, "n_synthetic": 2, "seed": 1}}))
    out = tmp_path / "out"
    assert main(["generate", "--demo", "--config", str(cfg), "--output", str(out)]) == 0
    capsys.readouterr()

    assert main(["inspect", str(out), "--episodes"]) == 0
    text = capsys.readouterr().out
    assert "episodes:  5 on disk" in text
    assert "2 synthetic / 3 real" in text
    assert "episode 4:" in text


def test_inspect_non_dataset_is_clean_error(tmp_path, capsys):
    rc = main(["inspect", str(tmp_path)])
    assert rc == 2
    assert "meta/info.json" in capsys.readouterr().err


def test_inspect_malformed_info_json_is_clean_error(tmp_path, capsys):
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text("{not json")
    rc = main(["inspect", str(tmp_path)])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize("flag,value", [("--seed", "-1"), ("--limit", "-1"), ("--max-frames", "0")])
def test_generate_rejects_bad_numeric_flags(tmp_path, capsys, flag, value):
    out = tmp_path / "out"
    rc = main(["generate", "--demo", flag, value, "--output", str(out)])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_generate_refuses_existing_dataset_without_overwrite(tmp_path, capsys):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"miniworld": {"enabled": False}, "pipeline": []}))
    out = tmp_path / "out"
    assert main(["generate", "--demo", "--config", str(cfg), "--output", str(out)]) == 0
    capsys.readouterr()
    rc = main(["generate", "--demo", "--config", str(cfg), "--output", str(out)])
    assert rc == 2
    assert "already contains a dataset" in capsys.readouterr().err
    # With --overwrite it succeeds and stays self-consistent.
    assert main(["generate", "--demo", "--config", str(cfg), "--output", str(out), "--overwrite"]) == 0
    assert len(read_lerobot_dataset(out)) == 3


def test_negative_seed_in_config_is_clean_error(tmp_path, capsys):
    cfg = tmp_path / "c.json"
    cfg.write_text(json.dumps({"miniworld": {"enabled": True, "n_synthetic": 1, "seed": -3}}))
    rc = main(["generate", "--demo", "--config", str(cfg), "--output", str(tmp_path / "out")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_augment_demo_writes_variants(tmp_path):
    cfg = tmp_path / "pipe.json"
    cfg.write_text(json.dumps([
        {"name": "lighting.brightness", "params": {"factor": 0.7}, "probability": 1.0},
    ]))
    out = tmp_path / "out"
    rc = main(["augment", "--demo", "--config", str(cfg), "--variants", "2",
               "--include-original", "--seed", "3", "--output", str(out)])
    assert rc == 0
    back = read_lerobot_dataset(out)
    assert len(back) == 9  # 3 demo originals + 3*2 variants
    assert sum(1 for e in back if e.metadata.get("augmented")) == 6
    assert all(not e.is_synthetic for e in back)  # augment never synthesizes


def test_augment_accepts_bare_list_or_pipeline_object(tmp_path):
    obj = tmp_path / "obj.json"
    obj.write_text(json.dumps({"pipeline": [{"name": "noise.gaussian", "params": {"sigma": 0.02}}]}))
    rc = main(["augment", "--demo", "--config", str(obj), "--output", str(tmp_path / "o1")])
    assert rc == 0


def test_augment_requires_config(tmp_path, capsys):
    rc = main(["augment", "--demo", "--output", str(tmp_path / "o")])
    assert rc == 2
    assert "config" in capsys.readouterr().err


def test_augment_rejects_bad_variants(tmp_path, capsys):
    cfg = tmp_path / "p.json"
    cfg.write_text(json.dumps([{"name": "noise.gaussian", "params": {}}]))
    rc = main(["augment", "--demo", "--config", str(cfg), "--variants", "0",
               "--output", str(tmp_path / "o")])
    assert rc == 2
    assert "variants" in capsys.readouterr().err
