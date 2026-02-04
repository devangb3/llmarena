"""
Generate a "rich" leaderboard summary JSON for a single competition.

This script reads per-problem run JSONs under:
  outputs/<comp>/<provider>/<model>/<problem_idx>.json

and produces a consolidated JSON (default):
  outputs/<comp>/leaderboard_summary_rich.json

Compared to scripts/extraction/leaderboard.py, this export also includes:
  - avg_pass_at_1
  - avg_pass_at_4
  - avg_output_tokens
  - avg_latency_sec
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import yaml


def _mean(xs: list[Optional[float]]) -> Optional[float]:
    vals = [x for x in xs if x is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _pass_at_k(correct: list[bool], k: int) -> Optional[float]:
    """Compute pass@k from a list of booleans (one per run)."""
    n = len(correct)
    if n == 0:
        return None
    c = sum(1 for x in correct if x)
    if n < k:
        return 1.0 if c > 0 else 0.0
    if c == 0:
        return 0.0
    ic = n - c
    if ic < k:
        return 1.0
    # Standard pass@k formula: 1 - C(n - c, k) / C(n, k) == 1 - C(ic, k) / C(n, k)
    return 1.0 - (math.comb(ic, k) / math.comb(n, k))


def _load_yaml_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _iter_model_dirs(comp_dir: Path) -> list[tuple[str, str, Path]]:
    """Return [(provider, model, model_dir)] for outputs/<comp>/<provider>/<model>/."""
    out: list[tuple[str, str, Path]] = []
    if not comp_dir.exists():
        return out
    for provider_dir in sorted([p for p in comp_dir.iterdir() if p.is_dir()]):
        provider = provider_dir.name
        for model_dir in sorted([m for m in provider_dir.iterdir() if m.is_dir()]):
            out.append((provider, model_dir.name, model_dir))
    return out


def _numeric_jsons(model_dir: Path) -> list[Path]:
    files = []
    for p in model_dir.glob("*.json"):
        if p.stem.isdigit():
            files.append(p)
    files.sort(key=lambda x: int(x.stem))
    return files


def generate_rich_leaderboard(
    comp: str,
    output_folder: str = "outputs",
    configs_folder: str = "configs/models",
    export_json: Optional[str] = None,
) -> Path:
    outputs_root = Path(output_folder)
    comp_dir = outputs_root / comp

    if export_json is None:
        export_path = comp_dir / "leaderboard_summary_rich.json"
    else:
        export_path = Path(export_json)

    entries: list[dict[str, Any]] = []
    model_dirs = _iter_model_dirs(comp_dir)

    for provider, model, model_dir in model_dirs:
        config_path = f"{provider}/{model}"

        # Load model config if present (used for display name + metadata)
        cfg_file = Path(configs_folder) / provider / f"{model}.yaml"
        model_cfg = _load_yaml_if_exists(cfg_file)

        prob_files = _numeric_jsons(model_dir)
        if not prob_files:
            continue

        per_problem_p1: list[Optional[float]] = []
        per_problem_p4: list[Optional[float]] = []
        all_correct: list[Optional[float]] = []
        all_out_tokens: list[Optional[float]] = []
        all_latency: list[Optional[float]] = []
        total_cost_like_leaderboard = 0.0
        any_cost = False

        for pf in prob_files:
            with pf.open("r", encoding="utf-8") as f:
                d = json.load(f)

            correct = d.get("correct", []) or []

            # Per-problem pass@1
            if correct:
                p1 = sum(1 for x in correct if x) / len(correct)
                per_problem_p1.append(float(p1))
                all_correct.extend([1.0 if x else 0.0 for x in correct])
            else:
                per_problem_p1.append(None)

            # Per-problem pass@4 (or k=4 formula; if fewer runs, reduces to any-correct)
            p4 = _pass_at_k([bool(x) for x in correct], 4)
            per_problem_p4.append(float(p4) if p4 is not None else None)

            # Costs / tokens / latency from detailed_costs
            dcs = d.get("detailed_costs", []) or []
            run_costs: list[Optional[float]] = []
            for dc in dcs:
                if not isinstance(dc, dict):
                    continue
                run_costs.append(dc.get("cost"))
                all_out_tokens.append(dc.get("output_tokens"))
                all_latency.append(dc.get("time"))
            mrc = _mean(run_costs)
            if mrc is not None:
                total_cost_like_leaderboard += float(mrc)
                any_cost = True

        avg_score = _mean(all_correct)
        if avg_score is None:
            continue

        entry = {
            "model_display_name": model_cfg.get("human_readable_id", model),
            "config_path": config_path,
            "avg_score": 100.0 * float(avg_score),
            "avg_pass_at_1": (100.0 * float(_mean(per_problem_p1))) if _mean(per_problem_p1) is not None else None,
            "avg_pass_at_4": (100.0 * float(_mean(per_problem_p4))) if _mean(per_problem_p4) is not None else None,
            "avg_output_tokens": _mean([float(x) if x is not None else None for x in all_out_tokens]),
            "avg_latency_sec": _mean([float(x) if x is not None else None for x in all_latency]),
            "avg_cost": float(total_cost_like_leaderboard) if any_cost else None,
            "rank": None,
            "date": model_cfg.get("date", "2000-01-01"),
            "config": model_cfg,
            "per_competition_scores": {comp: 100.0 * float(avg_score)},
        }
        entries.append(entry)

    entries.sort(key=lambda x: x["avg_score"], reverse=True)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    with export_path.open("w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2)

    return export_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leaderboard_summary JSON for a single competition")
    parser.add_argument("--comp", type=str, required=True, help="Competition name (e.g., gpqa/gpqa_diamond)")
    parser.add_argument("--output-folder", type=str, default="outputs", help="Outputs folder (default: outputs)")
    parser.add_argument("--configs-folder", type=str, default="configs/models", help="Configs folder (default: configs/models)")
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Path to write JSON (default: outputs/<comp>/leaderboard_summary_rich.json)",
    )
    args = parser.parse_args()

    out_path = generate_rich_leaderboard(
        comp=args.comp,
        output_folder=args.output_folder,
        configs_folder=args.configs_folder,
        export_json=args.export_json,
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    # Ensure we run from repo root: script assumes relative paths by default.
    os.chdir(Path(__file__).resolve().parents[2])
    main()

