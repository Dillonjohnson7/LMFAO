"""Interactive terminal wizard for the ``lmfao`` CLI.

Launched by running ``lmfao`` with no arguments (or ``lmfao wizard``). It walks a
newcomer through the whole flow in plain prompts: paste a Hugging Face dataset
link (or a local path, or use the built-in demo), pick what to do, choose an
augmentation preset, and it runs the same code paths the flag-driven commands
use. No external dependencies -- just ANSI-colored ``input()`` prompts that fall
back to plain text when the terminal has no color.

The flag-driven subcommands remain the scriptable interface; this is the
friendly front door.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------- styling

_USE_COLOR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None and os.environ.get("TERM") != "dumb"


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _USE_COLOR else s


def _bold(s: str) -> str: return _c("1", s)
def _dim(s: str) -> str: return _c("2", s)
def _cyan(s: str) -> str: return _c("36", s)
def _green(s: str) -> str: return _c("32", s)
def _yellow(s: str) -> str: return _c("33", s)
def _red(s: str) -> str: return _c("31", s)


def _banner() -> None:
    line = _dim("─" * 52)
    print()
    print(line)
    print(f"  {_bold('LMFAO')} {_dim('· augment robot demos into training data')}")
    print(line)
    print(f"  {_dim('Answer a few questions. Enter accepts the [default].')}")
    print(f"  {_dim('Ctrl-C to cancel at any time.')}")
    print()


# ---------------------------------------------------------------- prompts


def _prompt(text: str) -> str:
    return input(text)


def ask_text(label: str, *, default: str | None = None, required: bool = False,
             validate=None, help: str | None = None) -> str:
    if help:
        print(f"{_cyan('?')} {_bold(label)}  {_dim(help)}")
    else:
        print(f"{_cyan('?')} {_bold(label)}")
    hint = f" {_dim('[' + default + ']')}" if default else ""
    while True:
        raw = _prompt(f"{_green('›')}{hint} ").strip()
        if not raw and default is not None:
            raw = default
        if not raw and required:
            print(_red("  Please enter a value."))
            continue
        if validate is not None:
            err = validate(raw)
            if err:
                print(_red(f"  {err}"))
                continue
        return raw


def ask_choice(label: str, options: list[tuple[str, str]], default: int = 0) -> str:
    """options: list of (value, description). Returns the chosen value."""
    print(f"{_cyan('?')} {_bold(label)}")
    for i, (val, desc) in enumerate(options):
        mark = _green("❯") if i == default else " "
        tail = f"  {_dim(desc)}" if desc else ""
        print(f"  {mark} {_bold(str(i + 1))}. {val}{tail}")
    while True:
        raw = _prompt(f"{_green('›')} choose 1-{len(options)} {_dim('[' + str(default + 1) + ']')} ").strip()
        if not raw:
            return options[default][0]
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1][0]
        print(_red(f"  Enter a number between 1 and {len(options)}."))


def ask_yesno(label: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    while True:
        raw = _prompt(f"{_cyan('?')} {_bold(label)} {_dim('[' + d + ']')} ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print(_red("  Please answer y or n."))


def ask_int(label: str, *, default: int, minimum: int = 0) -> int:
    while True:
        raw = _prompt(f"{_cyan('?')} {_bold(label)} {_dim('[' + str(default) + ']')} ").strip()
        if not raw:
            return default
        try:
            v = int(raw)
        except ValueError:
            print(_red("  Enter a whole number."))
            continue
        if v < minimum:
            print(_red(f"  Must be at least {minimum}."))
            continue
        return v


def ask_multi(label: str, options: list[tuple[str, str]], defaults: list[int]) -> list[str]:
    """Multi-select. User types comma-separated numbers, 'all', or Enter for defaults."""
    print(f"{_cyan('?')} {_bold(label)}  {_dim('(comma-separated numbers, or ' + chr(39) + 'all' + chr(39) + ')')}")
    for i, (val, desc) in enumerate(options):
        pre = _green("✓") if i in defaults else " "
        tail = f"  {_dim(desc)}" if desc else ""
        print(f"  {pre} {_bold(str(i + 1))}. {val}{tail}")
    default_label = ",".join(str(i + 1) for i in defaults) or "none"
    while True:
        raw = _prompt(f"{_green('›')} {_dim('[' + default_label + ']')} ").strip().lower()
        if not raw:
            return [options[i][0] for i in defaults]
        if raw == "all":
            return [v for v, _ in options]
        try:
            idx = [int(p) - 1 for p in raw.replace(" ", "").split(",") if p]
        except ValueError:
            print(_red("  Enter numbers like 1,3,4."))
            continue
        if all(0 <= i < len(options) for i in idx) and idx:
            return [options[i][0] for i in idx]
        print(_red(f"  Use numbers between 1 and {len(options)}."))


# ---------------------------------------------------------------- HF links


def parse_hf_link(text: str) -> str | None:
    """Return an ``owner/name`` repo id if ``text`` looks like a Hugging Face
    dataset reference, else None."""
    t = text.strip()
    for prefix in (
        "https://huggingface.co/datasets/",
        "http://huggingface.co/datasets/",
        "huggingface.co/datasets/",
        "hf.co/datasets/",
        "https://huggingface.co/",
    ):
        if t.startswith(prefix):
            t = t[len(prefix):]
            break
    t = t.strip("/").split("?")[0].split("#")[0]
    parts = t.split("/")
    if len(parts) == 2 and all(parts) and " " not in t:
        return t
    return None


# ---------------------------------------------------------------- magnitude sweep

# Effects with a clean magnitude axis. Each step k adds inc*k to the base param.
# Bidirectional effects emit both directions (+ and -) per step, so N steps => 2N
# videos; magnitude-only effects (noise) emit N.
# ``pos`` / ``neg`` name the direction of a + / - step so labels read plainly
# (e.g. color temperature + is cooler, - is warmer, matching the augmenter).
_STEP_AUGS: dict[str, dict] = {
    "lighting.brightness": dict(param="factor", base=1.0, inc=0.05, bidir=True, scale=100, unit="%",
                                pos="brighter", neg="darker", desc="brighter / darker"),
    "lighting.contrast": dict(param="factor", base=1.0, inc=0.10, bidir=True, scale=100, unit="%",
                              pos="more", neg="less", desc="more / less contrast"),
    "lighting.color_temperature": dict(param="shift", base=0.0, inc=0.2, bidir=True, scale=1, unit="",
                                       pos="cooler", neg="warmer", desc="cooler / warmer"),
    "noise.gaussian": dict(param="sigma", base=0.0, inc=0.02, bidir=False, scale=100, unit="%",
                           desc="sensor grain"),
    "noise.uniform": dict(param="amplitude", base=0.0, inc=0.02, bidir=False, scale=100, unit="%",
                          desc="quantisation noise"),
}


def _fmt(value: float) -> str:
    return f"{value:g}"


def _build_sweep(name: str, d: dict, n_steps: int) -> tuple[list[dict], list[str]]:
    """Return (specs, magnitude_labels) for ``n_steps`` of effect ``name``."""
    param, base, inc = d["param"], d["base"], d["inc"]
    specs: list[dict] = []
    mags: list[str] = []
    for k in range(1, n_steps + 1):
        mag = inc * k
        disp = f"{_fmt(mag * d['scale'])}{d['unit']}"
        mags.append(("±" if d["bidir"] else "") + disp)
        if d["bidir"]:
            specs.append(_one(name, param, base + mag, f"+{disp} ({d['pos']})"))
            specs.append(_one(name, param, base - mag, f"-{disp} ({d['neg']})"))
        else:
            specs.append(_one(name, param, base + mag, disp))
    return specs, mags


def _one(name: str, param: str, value: float, tag: str) -> dict:
    return {"label": f"{name} {tag}",
            "pipeline": [{"name": name, "params": {param: round(value, 4)}, "probability": 1.0}]}


def _episode_count(src: dict, limit: int) -> int | None:
    """Best-effort source-episode count for the projection (no video decode)."""
    if src["kind"] == "demo":
        n = 3
    else:
        try:
            import pyarrow.parquet as pq

            from lmfao.datasets.lerobot import _episode_records
            n = len(_episode_records(Path(src["path"]), pq))
        except Exception:  # noqa: BLE001 - projection is advisory only
            return None
    return min(n, limit) if limit > 0 else n


# ---------------------------------------------------------------- augment presets

_PRESETS: dict[str, list[dict]] = {
    "light": [
        {"name": "lighting.brightness", "params": {}, "probability": 1.0},
        {"name": "lighting.color_temperature", "params": {}, "probability": 1.0},
    ],
    "standard": [
        {"name": "lighting.brightness", "params": {}, "probability": 1.0},
        {"name": "lighting.color_temperature", "params": {}, "probability": 1.0},
        {"name": "noise.gaussian", "params": {"sigma": 0.03}, "probability": 1.0},
    ],
    "heavy": [
        {"name": "lighting.brightness", "params": {}, "probability": 1.0},
        {"name": "lighting.contrast", "params": {}, "probability": 1.0},
        {"name": "lighting.color_temperature", "params": {}, "probability": 1.0},
        {"name": "noise.gaussian", "params": {"sigma": 0.04}, "probability": 0.8},
        {"name": "occlusion.moving_box", "params": {}, "probability": 0.5},
        {"name": "spatial.random_crop", "params": {"pad": 8}, "probability": 0.7},
    ],
}


def _choose_pipeline() -> list[dict]:
    choice = ask_choice(
        "Which augmentations?",
        [
            ("standard", "brightness + color + a little noise (recommended)"),
            ("light", "just relighting (brightness + color)"),
            ("heavy", "relight + noise + occlusion + crop"),
            ("custom", "pick each one yourself"),
        ],
        default=0,
    )
    if choice != "custom":
        return _PRESETS[choice]

    from lmfao.registry import list_augmenter_info
    info = list_augmenter_info()
    names = sorted(info) if isinstance(info, dict) else sorted(a["name"] for a in info)
    opts = [(n, "") for n in names]
    default_idx = [i for i, n in enumerate(names)
                   if n in ("lighting.brightness", "lighting.color_temperature", "noise.gaussian")]
    picked = ask_multi("Select augmentations to apply", opts, default_idx)
    return [{"name": n, "params": {}, "probability": 1.0} for n in picked]


# ---------------------------------------------------------------- data source


def _resolve_source() -> dict:
    """Return {'kind': 'demo'|'dataset', 'path': str|None, 'video_key': str|None}."""
    print(f"{_cyan('?')} {_bold('Where is your data?')}")
    print(f"  {_dim('Paste a Hugging Face dataset link, e.g.')}")
    print(f"  {_dim('  https://huggingface.co/datasets/owner/name   or   owner/name')}")
    print(f"  {_dim('or a local dataset folder, or type')} {_bold('demo')} {_dim('to try the built-in scene.')}")
    while True:
        raw = _prompt(f"{_green('›')} {_dim('[demo]')} ").strip()
        if not raw or raw.lower() == "demo":
            return {"kind": "demo", "path": None, "video_key": None}

        # A local dataset folder wins over HF parsing, since an id like
        # "owner/name" and a relative path "data/name" look identical.
        p = Path(raw).expanduser()
        if (p / "meta" / "info.json").exists():
            return _pick_camera(p)

        repo = parse_hf_link(raw)
        if repo is not None:
            path = _download_from_hf(repo)
            if path is None:
                continue  # download failed; ask again
            return _pick_camera(path)

        if p.exists():
            print(_red(f"  {p} exists but is not a LeRobot dataset (no meta/info.json)."))
        else:
            print(_red(f"  Not a Hugging Face link and no folder at {p}."))
        print(_dim("  Try again, or type 'demo'."))


def _download_from_hf(repo: str) -> Path | None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(_red("  Downloading from Hugging Face needs huggingface_hub."))
        print(_dim("  Install it: pip install huggingface_hub"))
        return None
    dest = Path("data") / repo.split("/")[-1]
    print(_dim(f"  Fetching metadata for {repo} ..."))
    try:
        snapshot_download(repo, repo_type="dataset", local_dir=str(dest),
                          allow_patterns=["meta/*", "data/*"])
    except Exception as e:  # noqa: BLE001 - surface any hub error as a friendly line
        print(_red(f"  Could not fetch {repo}: {e}"))
        return None
    info_path = dest / "meta" / "info.json"
    if not info_path.exists():
        print(_red(f"  {repo} does not look like a LeRobot dataset (no meta/info.json)."))
        return None

    info = json.loads(info_path.read_text())
    keys = [k for k, f in info.get("features", {}).items() if f.get("dtype") == "video"]
    on_disk = [k for k in keys if (dest / "videos" / k).exists()]
    missing = [k for k in keys if k not in on_disk]
    if missing:
        key = missing[0] if not on_disk else _pick_key_to_download(keys, on_disk)
        print(_yellow(f"  Downloading video for '{key}' (this can be large) ..."))
        try:
            snapshot_download(repo, repo_type="dataset", local_dir=str(dest),
                              allow_patterns=[f"videos/{key}/**"])
        except Exception as e:  # noqa: BLE001
            print(_red(f"  Video download failed: {e}"))
            return None
    print(_green(f"  Ready: {dest}"))
    return dest


def _pick_key_to_download(keys: list[str], on_disk: list[str]) -> str:
    opts = [(k, "already downloaded" if k in on_disk else "") for k in keys]
    return ask_choice("Which camera stream to use?", opts, default=0)


def _pick_camera(path: Path) -> dict:
    info = json.loads((path / "meta" / "info.json").read_text())
    keys = [k for k, f in info.get("features", {}).items() if f.get("dtype") == "video"]
    on_disk = [k for k in keys if (path / "videos" / k).exists()]
    if len(on_disk) <= 1:
        return {"kind": "dataset", "path": str(path), "video_key": on_disk[0] if on_disk else None}
    key = ask_choice("Which camera stream?", [(k, "") for k in on_disk], default=0)
    return {"kind": "dataset", "path": str(path), "video_key": key}


# ---------------------------------------------------------------- run


def _write_config(pipeline: list[dict], miniworld: dict | None = None) -> str:
    cfg: dict = {"pipeline": pipeline}
    if miniworld is not None:
        cfg["miniworld"] = miniworld
    return _write_temp(cfg)


def _write_sweep_config(specs: list[dict]) -> str:
    return _write_temp({"sweep": specs})


def _write_temp(cfg: dict) -> str:
    fd, name = tempfile.mkstemp(prefix="lmfao_wizard_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(cfg, f)
    return name


def _summary(lines: list[tuple[str, str]]) -> None:
    print()
    print(_bold("  Summary"))
    width = max(len(k) for k, _ in lines)
    for k, v in lines:
        print(f"    {_dim(k.ljust(width))}  {v}")
    print()


def run_wizard() -> int:
    # Imported here so the wizard module stays importable even if the datasets
    # extra is missing until the user actually runs something.
    from lmfao.cli import _augment, _generate, _inspect

    _banner()
    try:
        src = _resolve_source()

        op = ask_choice(
            "What do you want to do?",
            [
                ("augment", "season real frames at full resolution (recommended)"),
                ("generate", "also synthesize novel camera views (experimental, low-res)"),
                ("inspect", "just summarize the dataset"),
            ],
            default=0,
        )

        if op == "inspect":
            if src["kind"] == "demo":
                print(_yellow("\n  Nothing to inspect for the demo scene. Try augment or generate."))
                return 0
            ns = argparse.Namespace(dataset=src["path"], episodes=True)
            print()
            return _inspect(ns)

        output = ask_text("Where should the output dataset go?",
                          default="output/augmented" if op == "augment" else "output/generated",
                          required=True)
        seed = ask_int("Random seed (for reproducibility)", default=7)

        if op == "augment":
            names = list(_STEP_AUGS)
            opts = [(n, _STEP_AUGS[n]["desc"]) for n in names]
            default_idx = [names.index(n) for n in
                           ("lighting.brightness", "lighting.color_temperature", "noise.gaussian")]
            chosen = ask_multi("Which effects to sweep?", opts, default_idx)

            print(_dim("\n  For each effect, choose how many magnitude steps."))
            specs: list[dict] = []
            rows: list[tuple[str, int, int, list[str]]] = []
            for name in chosen:
                d = _STEP_AUGS[name]
                arrow = "+/-" if d["bidir"] else "+"
                n = ask_int(f"  {name} ({arrow}, {_fmt(d['inc'] * d['scale'])}{d['unit']} per step) — steps?",
                            default=3, minimum=1)
                aug_specs, mags = _build_sweep(name, d, n)
                specs.extend(aug_specs)
                rows.append((name, n, len(aug_specs), mags))

            keep = ask_yesno("Also keep the original (un-augmented) episodes?", default=False)
            limit = ask_int("Limit to N source episodes (0 = all)", default=0, minimum=0)
            max_frames = ask_int("Quick test: cap frames per episode (0 = full clip)", default=0, minimum=0)

            per_ep = sum(r[2] for r in rows)
            n_ep = _episode_count(src, limit)
            print()
            print(_bold("  Projected output"))
            for name, nsteps, count, mags in rows:
                d = _STEP_AUGS[name]
                dir_note = _dim(f"  +{d['pos']} / -{d['neg']}") if d["bidir"] else ""
                print(f"    {name:26} {nsteps} step(s){'  ±' if d['bidir'] else '   '}  "
                      f"{_green(str(count))} videos/ep   {_dim(', '.join(mags))}{dir_note}")
            print(_dim(f"    {'─' * 60}"))
            total_line = f"    {per_ep} variations/episode"
            if n_ep is not None:
                total = per_ep * n_ep + (n_ep if keep else 0)
                total_line += f"  ×  {n_ep} episode(s)  =  {_bold(_green(str(per_ep * n_ep)))} augmented videos"
                if keep:
                    total_line += _dim(f"  (+ {n_ep} originals = {total} total)")
            else:
                total_line += _dim("  (episode count unknown until read)")
            print(total_line)
            print()

            if not ask_yesno("Run it?", default=True):
                print(_dim("  Cancelled."))
                return 0
            cfg_path = _write_sweep_config(specs)
            ns = argparse.Namespace(
                input=src["path"], output=output, config=cfg_path,
                demo=(src["kind"] == "demo"), variants=1,
                include_original=keep, seed=seed, video_key=src["video_key"],
                write_video_key=None, limit=(limit or None),
                max_frames=(max_frames or None), overwrite=True,
            )
            print()
            return _augment(ns)

        # generate
        pipeline = _choose_pipeline()
        n_synth = ask_int("How many synthetic novel-view episodes per source?", default=2, minimum=1)
        work = ask_int("Working resolution for the reference renderer (px)", default=96, minimum=16)
        assume = True if src["kind"] != "demo" else False
        limit = ask_int("Limit to N source episodes (0 = all)", default=1, minimum=0)
        miniworld = {"enabled": True, "n_synthetic": n_synth, "seed": seed,
                     "object_pose_region": [[-0.05, -0.05], [0.05, 0.05]]}
        cfg_path = _write_config(pipeline, miniworld=miniworld)
        _summary([
            ("source", "demo scene" if src["kind"] == "demo" else src["path"]),
            ("camera", src["video_key"] or "-"),
            ("operation", "generate (miniworld + augment)"),
            ("synthetic", f"{n_synth} per source episode"),
            ("working res", f"{work}px"),
            ("assume poses", "yes (real data has none)" if assume else "no (demo has real poses)"),
            ("augmentations", ", ".join(s["name"] for s in pipeline)),
            ("episodes", "all" if limit == 0 else str(limit)),
            ("output", output),
        ])
        print(_dim("  Note: generate uses the low-res reference renderer; output is a preview,"))
        print(_dim("  not training-grade. Use augment for full-resolution training data."))
        if not ask_yesno("Run it?", default=True):
            print(_dim("  Cancelled."))
            return 0
        ns = argparse.Namespace(
            input=src["path"], output=output, config=cfg_path,
            demo=(src["kind"] == "demo"), seed=seed, video_key=src["video_key"],
            write_video_key="observation.images.render", limit=(limit or None),
            max_frames=None, assume_poses=assume, work_size=work, overwrite=True,
        )
        print()
        return _generate(ns)

    except (KeyboardInterrupt, EOFError):
        print(_dim("\n\n  Cancelled. Nothing was written."))
        return 130
