"""
plot_rq2_probing.py — RQ2 probing figure runner (§5.2).

Reads:  results/rq2_probing/accuracy_metrics_corrected.csv
Outputs: results/figures/rq2/accuracy_curves.{png,html}

Figure 2: accuracy-vs-layer curves (reuses probing_viz.plot_accuracy_curves) with
per-property emergence markers (first layer with accuracy > 0.7, E-G-03 operational
definition) and a shaded span over the parity plateau→jump (the largest single-layer
accuracy increase). Both are computed from the data, never hardcoded.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.viz.probing_viz import plot_accuracy_curves

EMERGENCE_THRESHOLD = 0.7


def compute_emergence_layers(df: pd.DataFrame, threshold: float = EMERGENCE_THRESHOLD) -> dict[str, int | None]:
    """First layer whose accuracy strictly exceeds threshold, per property (None if never)."""
    out: dict[str, int | None] = {}
    for prop in df["property"].unique():
        sub = df[df["property"] == prop].sort_values("layer")
        over = sub[sub["accuracy"] > threshold]
        out[prop] = int(over["layer"].iloc[0]) if len(over) else None
    return out


def compute_jump_span(df: pd.DataFrame, prop: str = "parity") -> tuple[int, int] | None:
    """(layer-1, layer) of the largest single-layer accuracy increase for prop."""
    sub = df[df["property"] == prop].sort_values("layer")
    if len(sub) < 2:
        return None
    deltas = sub["accuracy"].diff()
    jump_layer = int(sub.loc[deltas.idxmax(), "layer"])
    return (jump_layer - 1, jump_layer)


def main(results_dir: str = "results/rq2_probing", out_dir: str = "results/figures/rq2") -> None:
    results = Path(results_dir)
    acc_file = results / "accuracy_metrics_corrected.csv"
    if not acc_file.exists():
        raise FileNotFoundError(f"RQ2 data missing: {acc_file} — run run_rq2.py first.")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(acc_file).sort_values("layer")
    emergence = compute_emergence_layers(df)
    jump_span = compute_jump_span(df, "parity")
    jump_label = None
    if jump_span is not None:
        jump_label = f"parity jump L{jump_span[0]}→L{jump_span[1]}"

    plot_accuracy_curves(
        df, model_name="Pythia-1.4B",
        output_png=out / "accuracy_curves.png",
        output_html=out / "accuracy_curves.html",
        emergence_layers=emergence, jump_span=jump_span, jump_label=jump_label,
    )
    print(f"RQ2 accuracy figure saved: {out/'accuracy_curves.html'} | emergence={emergence}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2 probing figures.")
    parser.add_argument("--results_dir", default="results/rq2_probing")
    parser.add_argument("--out_dir", default="results/figures/rq2")
    args = parser.parse_args()
    main(args.results_dir, args.out_dir)
