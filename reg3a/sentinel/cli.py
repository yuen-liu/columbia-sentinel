"""Console entry point for the Reg3A interpretability pipeline.

Each subcommand runs one stage script from the repository's numbered stage
directories. Arguments after the stage name are passed through untouched, so

    sentinel train --data-dir reg3a_layer23_paired --output-dir checkpoints/run1

is equivalent to

    python 03_train/train.py --data-dir reg3a_layer23_paired --output-dir checkpoints/run1

and ``sentinel <stage> --help`` shows that stage's own options.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

# stage name -> (path relative to the repository root, one-line summary)
STAGES: dict[str, tuple[str, str]] = {
    "manifest": (
        "00_data/build_manifest.py",
        "Build the design table from a Boltz-Prot-1 campaign tree",
    ),
    "layer-sweep": (
        "01_embed/layer_sweep.py",
        "Score candidate ESM-C layers and record the chosen one",
    ),
    "embed": (
        "01_embed/embed_esmc_paired.py",
        "Co-attend binder with target and write per-residue activations",
    ),
    "train": (
        "03_train/train.py",
        "Train the TopK sparse autoencoder (k-annealing, AuxK, resumable)",
    ),
    "interface": (
        "05_interpret/interface_activations.py",
        "Compute interface geometry and per-feature enrichment",
    ),
    "label": (
        "05_interpret/label_features.py",
        "Label features from their max-activating examples",
    ),
    "steer-motif": (
        "06_steering/steer_recover_motif.py",
        "Motif-recovery readout for a single feature direction",
    ),
    "steer-v2": (
        "06_steering/steer_v2.py",
        "Calibrated clamp with multi-layer injection schedules",
    ),
    "steer-canvas": (
        "06_steering/steer_canvas.py",
        "Synthetic-canvas activation-profile matching",
    ),
    "cross-injection": (
        "06_steering/cross_injection_test.py",
        "The specificity test: does the RIGHT feature win?",
    ),
}

# sentinel/cli.py -> sentinel/ -> repository root
REPO_ROOT = Path(__file__).resolve().parent.parent


def _build_parser() -> argparse.ArgumentParser:
    width = max(len(name) for name in STAGES)
    stage_help = "\n".join(
        f"  {name:<{width}}  {summary}" for name, (_, summary) in STAGES.items()
    )
    parser = argparse.ArgumentParser(
        prog="sentinel",
        description=(
            "Sparse-autoencoder interpretability for AI-designed protein binders.\n\n"
            "Stages run in the order listed and each one is resumable. Arguments\n"
            "after the stage name are forwarded to that stage unchanged; use\n"
            "`sentinel <stage> --help` for its own options.\n\n"
            f"Stages:\n{stage_help}"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "stage",
        choices=list(STAGES),
        metavar="STAGE",
        help="pipeline stage to run (see list above)",
    )
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to the stage script",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help"}:
        _build_parser().print_help()
        return 0

    parsed = _build_parser().parse_args(argv)
    relative_path, _ = STAGES[parsed.stage]
    script = REPO_ROOT / relative_path

    if not script.is_file():
        print(
            f"sentinel: cannot find {relative_path}\n"
            f"  looked under: {REPO_ROOT}\n"
            "  The stage scripts live in the repository, so install from a\n"
            "  checkout with `pip install -e .` rather than from a wheel.",
            file=sys.stderr,
        )
        return 2

    # Stage scripts are __main__-guarded and read sys.argv directly.
    sys.argv = [str(script), *parsed.args]
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exit_signal:  # argparse and explicit sys.exit in stages
        return int(exit_signal.code or 0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
