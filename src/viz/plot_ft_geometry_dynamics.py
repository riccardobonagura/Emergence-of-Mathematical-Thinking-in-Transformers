"""plot_ft_geometry_dynamics.py — Supplementary dashboard (exploratory).

Visualizes how RQ1's descriptive geometry (cross-temporal CKA, inter-category CKA,
ΔIso) co-moves with GSM8K across QLoRA checkpoints. This is an exploratory bridge
between RQ1 and RQ3 — NOT confirmatory. Reads results/rq1_emergence/dynamic/rq1_dynamics.csv.
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ft_geometry_viz")

DYN_CSV = Path("results/rq1_emergence/dynamic/rq1_dynamics.csv")
ANN_CSV = Path("results/rq1_emergence/cka_results_annotated.csv")
TRAJ_CSV = Path("results/rq4_drift/trajectories_probing.csv")
NF4_JSON = Path("results/nf4_degradation/summary.json")
OUT_HTML = Path("results/figures/supplementary_ft_dynamics.html")

STEP_COLORS = ["#1f2937", "#2563eb", "#0891b2", "#059669", "#d97706", "#dc2626",
               "#7c3aed", "#db2777", "#65a30d", "#0d9488"]


def _intro(title: str, body_html: str) -> str:
    """One panel introduction block (detailed prose + caveats)."""
    return (
        f'<div style="max-width:1360px;margin:28px auto 6px auto;font-family:system-ui,'
        f'Segoe UI,Arial;color:#1f2937;line-height:1.5">'
        f'<h2 style="margin-bottom:4px;color:#0f172a">{title}</h2>{body_html}</div>'
    )


def main() -> None:
    if not DYN_CSV.exists():
        logger.error(f"Missing {DYN_CSV}. Run run_rq1_dynamics.py first.")
        raise FileNotFoundError(DYN_CSV)

    df = pd.read_csv(DYN_CSV)
    steps = sorted(df["step"].unique())
    layers = sorted(df["layer"].unique())

    # GSM8K per step (read live), assert coverage of the dynamics steps.
    gsm = {}
    if TRAJ_CSV.exists():
        traj = pd.read_csv(TRAJ_CSV)
        if "gsm8k_acc" in traj.columns:
            gsm = traj.groupby("step")["gsm8k_acc"].first().dropna().to_dict()
    missing = [s for s in steps if s not in gsm]
    if missing:
        logger.warning(f"GSM8K missing for steps {missing}; overlay will skip those points.")
    gsm_x = [s for s in steps if s in gsm]
    gsm_y = [gsm[s] for s in gsm_x]

    # T16 NF4 non-learning floor (read live; key confirmed mean_frobenius_relative).
    t16 = None
    if NF4_JSON.exists():
        t16 = json.load(open(NF4_JSON, encoding="utf-8")).get("mean_frobenius_relative")

    # Within-model baseline band (Panel B only) from existing RQ1 reviewer baselines.
    band = None
    if ANN_CSV.exists():
        ann = pd.read_csv(ANN_CSV)
        if {"cka_ctrl_neu_vs_num", "cka_math_template_baseline"}.issubset(ann.columns):
            band = ann.set_index("layer")[["cka_ctrl_neu_vs_num", "cka_math_template_baseline"]]

    gsm_caveat = (
        "<b>Reading the GSM8K overlay (descriptive only).</b> With <b>n = "
        f"{len(steps)}</b> checkpoints this is a visual trend, <b>not</b> a correlation: "
        "no coefficient or significance is claimable (E-G-04, E-M-02, E-O-04). MetaMath "
        "(fine-tuning) and GSM8K share distributional overlap — a third-variable confound, "
        "so co-movement is not causal. The trajectory is within ~1 epoch (~12k steps) and "
        "may be pre-convergence, so the endpoint is not a saturated geometry (E-F-02). "
        "Report magnitude, not coefficients."
    )

    html_parts = ['<html><head><meta charset="utf-8"><title>Supplementary — FT geometry '
                  'dynamics</title></head><body style="background:#f8fafc">']
    html_parts.append(_intro(
        "Supplementary — Fine-tuning geometry dynamics (exploratory bridge between RQ1 and RQ3)",
        "<p><b>This is a supplementary, post-hoc exploratory analysis — not part of either "
        "RQ's confirmatory design.</b> It recomputes RQ1's descriptive geometry on the QLoRA "
        "checkpoints and shows how it co-moves with GSM8K. Evolutionary (layer-to-layer) CKA, "
        "the third RQ1 CKA notion, is intentionally out of scope here. Step 0 = base model; "
        f"steps {steps[1:]} are checkpoints (final_adapter = {steps[-1]}). All metrics use the "
        "same 3000 stimuli in identical order (hash-verified against the base).</p>"
        f"<p style='background:#fef9c3;padding:8px 12px;border-radius:6px'>{gsm_caveat}</p>"))

    first = True

    def add_fig(fig: go.Figure) -> None:
        nonlocal first
        html_parts.append(fig.to_html(full_html=False,
                                      include_plotlyjs="cdn" if first else False))
        first = False

    # ── Panel A — Cross-temporal CKA drift (base → checkpoint) ────────────────
    pivot = df.pivot(index="layer", columns="step", values="cka_vs_base")
    drift_pivot = 1.0 - pivot
    mean_drift = drift_pivot.mean(axis=0)

    figA = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42],
                         subplot_titles=("Drift heatmap: 1 − CKA(base, ckpt)",
                                         "Mean drift vs step (+ GSM8K)"),
                         specs=[[{"type": "heatmap"}, {"secondary_y": True}]])
    figA.add_trace(go.Heatmap(z=drift_pivot.values, x=[str(s) for s in drift_pivot.columns],
                              y=drift_pivot.index, colorscale="Inferno",
                              colorbar=dict(title="1−CKA", x=0.46),
                              hovertemplate="Step %{x}<br>Layer %{y}<br>Drift %{z:.4f}<extra></extra>"),
                   row=1, col=1)
    figA.add_trace(go.Scatter(x=[str(s) for s in mean_drift.index], y=mean_drift.values,
                              mode="lines+markers", name="mean 1−CKA drift",
                              line=dict(color="#dc2626")), row=1, col=2, secondary_y=False)
    if gsm_x:
        figA.add_trace(go.Scatter(x=[str(s) for s in gsm_x], y=gsm_y, mode="lines+markers",
                                  name="GSM8K 0-shot", line=dict(color="black", dash="dash")),
                       row=1, col=2, secondary_y=True)
    figA.update_xaxes(title_text="Training step", row=1, col=1)
    figA.update_xaxes(title_text="Training step", row=1, col=2)
    figA.update_yaxes(title_text="Layer", row=1, col=1)
    figA.update_yaxes(title_text="Mean 1−CKA drift", secondary_y=False, row=1, col=2)
    figA.update_yaxes(title_text="GSM8K acc", secondary_y=True, row=1, col=2)
    figA.update_layout(height=520, width=1360, template="plotly_white",
                       title="Panel A — Cross-temporal CKA drift")
    t16_txt = f"{t16:.3f}" if t16 is not None else "n/a"
    html_parts.append(_intro(
        "Panel A — Cross-temporal CKA drift (base → checkpoint)",
        "<p>For each layer L, <code>CKA(H_base[L], H_ckpt[L])</code> measures how much "
        "fine-tuning <i>restructured</i> that layer's representation. Linear CKA is invariant "
        "to rotation and isotropic scaling — that invariance is the <b>feature</b> that lets "
        "it isolate genuine restructuring from mere rotation/rescaling, making it the "
        "complement to RQ3's Frobenius drift (E-G-02). <b>CKA is a similarity, not a "
        "divergence.</b></p>"
        f"<p><b>Noise reference (not a band):</b> step 0 is 1.0 by construction (base vs base). "
        f"There is <b>no apples-to-apples CKA noise band</b> for a cross-temporal comparison, "
        f"so the within-model floors of Panel B do not apply here. As an external, "
        f"<i>different-unit</i> reference (E-F-03), the T16 NF4 quantization degradation is "
        f"mean relative Frobenius = <b>{t16_txt}</b> — a non-learning floor expressed in "
        f"Frobenius, NOT in 1−CKA, so compare only qualitatively. Observed max mean drift here "
        f"is ≈{mean_drift.max():.3f} (1−CKA): most of RQ3's larger Frobenius drift is therefore "
        f"rotation/scaling, not restructuring.</p>"))
    add_fig(figA)

    # ── Panel B — Inter-category CKA(math, ctrl) ──────────────────────────────
    figB = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42],
                         subplot_titles=("CKA(math, ctrl) per layer — one line per step",
                                         "Terminal-layer inter-CKA vs step (+ GSM8K)"),
                         specs=[[{"type": "scatter"}, {"secondary_y": True}]])
    if band is not None:
        lo = band.min(axis=1).reindex(layers)
        hi = band.max(axis=1).reindex(layers)
        figB.add_trace(go.Scatter(x=layers, y=hi.values, mode="lines",
                                  line=dict(width=0), showlegend=False, hoverinfo="skip"),
                       row=1, col=1)
        figB.add_trace(go.Scatter(x=layers, y=lo.values, mode="lines", fill="tonexty",
                                  fillcolor="rgba(148,163,184,0.30)", line=dict(width=0),
                                  name="within-model noise band"), row=1, col=1)
    for i, s in enumerate(steps):
        d = df[df["step"] == s].sort_values("layer")
        figB.add_trace(go.Scatter(x=d["layer"], y=d["cka_inter"], mode="lines+markers",
                                  name=f"step {s}", line=dict(color=STEP_COLORS[i % len(STEP_COLORS)])),
                       row=1, col=1)
    term = layers[-1]
    term_series = df[df["layer"] == term].sort_values("step")
    figB.add_trace(go.Scatter(x=[str(s) for s in term_series["step"]], y=term_series["cka_inter"],
                              mode="lines+markers", name=f"inter-CKA @L{term}",
                              line=dict(color="#0891b2")), row=1, col=2, secondary_y=False)
    if gsm_x:
        figB.add_trace(go.Scatter(x=[str(s) for s in gsm_x], y=gsm_y, mode="lines+markers",
                                  name="GSM8K 0-shot", line=dict(color="black", dash="dash")),
                       row=1, col=2, secondary_y=True)
    figB.update_xaxes(title_text="Layer", row=1, col=1)
    figB.update_xaxes(title_text="Training step", row=1, col=2)
    figB.update_yaxes(title_text="Inter-category CKA", row=1, col=1)
    figB.update_yaxes(title_text=f"inter-CKA @L{term}", secondary_y=False, row=1, col=2)
    figB.update_yaxes(title_text="GSM8K acc", secondary_y=True, row=1, col=2)
    figB.update_layout(height=520, width=1360, template="plotly_white",
                       title="Panel B — Inter-category CKA(math, ctrl)")
    html_parts.append(_intro(
        "Panel B — Inter-category CKA (math vs control), within model",
        "<p>Per layer, <code>CKA(math stimuli, control stimuli)</code> inside one model. "
        "A <b>high</b> value means math and control occupy a <b>similar</b> relational geometry "
        "— it must <b>not</b> be read as 'divergence' (E-G-02). The line family shows whether "
        "fine-tuning shifts that separation curve.</p>"
        "<p><b>Within-model noise band (shaded):</b> the expected-variance floor from the RQ1 "
        "reviewer baselines — CKA(CTRL-NEU, CTRL-NUM) and within-math across-template CKA. A "
        "change <b>inside</b> this band is noise, not signal. (These floors are within-model and "
        "do not transfer to Panel A.) <b>Caveat — positional asymmetry:</b> math states are read "
        "at the '=' token, control states at the terminal word/'.', so inter-category CKA "
        "conflates content with this positional gap.</p>"))
    add_fig(figB)

    # ── Panel C — Isotropy ΔIso(math − ctrl) ──────────────────────────────────
    figC = make_subplots(rows=1, cols=2, column_widths=[0.58, 0.42],
                         subplot_titles=("ΔIso(math − ctrl) per layer — one line per step",
                                         "min ΔIso vs step (+ GSM8K)"),
                         specs=[[{"type": "scatter"}, {"secondary_y": True}]])
    for i, s in enumerate(steps):
        d = df[df["step"] == s].sort_values("layer")
        figC.add_trace(go.Scatter(x=d["layer"], y=d["delta_iso"], mode="lines+markers",
                                  name=f"step {s}", line=dict(color=STEP_COLORS[i % len(STEP_COLORS)])),
                       row=1, col=1)
    figC.add_hline(y=0, line=dict(color="#94a3b8", dash="dot"), row=1, col=1)
    min_diso = df.groupby("step")["delta_iso"].min().reindex(steps)
    figC.add_trace(go.Scatter(x=[str(s) for s in min_diso.index], y=min_diso.values,
                              mode="lines+markers", name="min ΔIso", line=dict(color="#7c3aed")),
                   row=1, col=2, secondary_y=False)
    if gsm_x:
        figC.add_trace(go.Scatter(x=[str(s) for s in gsm_x], y=gsm_y, mode="lines+markers",
                                  name="GSM8K 0-shot", line=dict(color="black", dash="dash")),
                       row=1, col=2, secondary_y=True)
    figC.update_xaxes(title_text="Layer", row=1, col=1)
    figC.update_xaxes(title_text="Training step", row=1, col=2)
    figC.update_yaxes(title_text="ΔIso (math − ctrl)", row=1, col=1)
    figC.update_yaxes(title_text="min ΔIso", secondary_y=False, row=1, col=2)
    figC.update_yaxes(title_text="GSM8K acc", secondary_y=True, row=1, col=2)
    figC.update_layout(height=520, width=1360, template="plotly_white",
                       title="Panel C — Isotropy ΔIso(math − ctrl)")
    html_parts.append(_intro(
        "Panel C — Isotropy drift, ΔIso(math − ctrl)",
        "<p>ΔIso = mean off-diagonal cosine of math minus that of control, per layer. It is a "
        "<b>relative</b> measure (E-G-01): a negative ΔIso means math is more anisotropic than "
        "control, but <b>anisotropy ≠ semantic richness</b> — it can reflect corpus token "
        "density (Ethayarajh 2019), not structure. The line family shows whether the base dip "
        "(≈ −0.106 at L15) migrates or deepens with fine-tuning. <b>Caveat — positional "
        "asymmetry</b> applies here too: ΔIso is also a math-vs-control contrast (math at '=', "
        "control at the terminal token).</p>"))
    add_fig(figC)

    html_parts.append("</body></html>")
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text("\n".join(html_parts), encoding="utf-8")
    logger.info(f"Dashboard written: {OUT_HTML}")
    logger.info(f"Panels: A (cross-temporal, T16 ref={t16}), B (inter-cat, band={'yes' if band is not None else 'no'}), "
                f"C (ΔIso). GSM8K steps covered: {gsm_x}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        sys.exit(1)
