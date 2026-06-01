"""pca_umap_viz.py — PCA/UMAP dimensionality reduction and 2-D scatter plots."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from plotly.graph_objects import Figure

try:
    import torch
except ImportError:
    torch = None


def compute_pca_embeddings(
    representations: "torch.Tensor | np.ndarray",
    n_components: int = 2,
    whiten: bool = False,
) -> np.ndarray:
    """PCA projection; returns float64 array of shape (n, n_components)."""
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise ImportError("scikit-learn required: pip install scikit-learn") from exc

    X = _to_numpy_2d(representations)
    if not (0 < n_components <= X.shape[1]):
        raise ValueError(
            f"n_components={n_components} must be in (0, {X.shape[1]}]"
        )
    return PCA(n_components=n_components, whiten=whiten, svd_solver="auto").fit_transform(X)


def compute_umap_embeddings(
    representations: "torch.Tensor | np.ndarray",
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    metric: str = "euclidean",
) -> np.ndarray:
    """UMAP non-linear projection; returns float64 array of shape (n, n_components)."""
    try:
        import umap  # type: ignore
    except ImportError as exc:
        raise ImportError("umap-learn required: pip install umap-learn") from exc

    X = _to_numpy_2d(representations)
    return umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    ).fit_transform(X)


def scatter_2d(
    embeddings_2d: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    *,
    title: str | None = None,
    palette: Optional[Sequence[str]] = None,
    backend: str = "matplotlib",
) -> "Figure":
    """2-D scatter coloured by integer label; supports matplotlib/seaborn/plotly."""
    X = np.asarray(embeddings_2d)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError(f"Expected shape (n, 2), got {X.shape}")
    if labels is not None and len(labels) != X.shape[0]:
        raise ValueError("labels length must match embeddings_2d length")

    backend = backend.lower().strip()

    if backend in {"matplotlib", "seaborn"}:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 6))
        if labels is None:
            ax.scatter(X[:, 0], X[:, 1], s=20, alpha=0.8)
        else:
            labels_arr = np.asarray(labels)
            cmap       = palette or ([None] * len(np.unique(labels_arr)))
            for idx, lab in enumerate(np.unique(labels_arr)):
                mask = labels_arr == lab
                ax.scatter(X[mask, 0], X[mask, 1], s=20, alpha=0.8,
                           c=cmap[idx] if idx < len(cmap) else None,
                           label=str(lab))
            ax.legend(title="label", loc="best", frameon=False)
        ax.set_xlabel("dim_1"); ax.set_ylabel("dim_2")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return fig

    if backend == "plotly":
        import plotly.graph_objects as go
        if labels is None:
            traces = [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                                 marker={"size": 6, "opacity": 0.85}, name="points")]
        else:
            labels_arr = np.asarray(labels)
            traces = [
                go.Scatter(x=X[labels_arr == lab, 0], y=X[labels_arr == lab, 1],
                           mode="markers", marker={"size": 6, "opacity": 0.85},
                           name=str(lab))
                for lab in np.unique(labels_arr)
            ]
        fig = go.Figure(data=traces)
        fig.update_layout(title=title or "2D scatter",
                          xaxis_title="dim_1", yaxis_title="dim_2",
                          template="plotly_white")
        return fig

    raise ValueError(f"backend must be one of: matplotlib, seaborn, plotly")


LABEL_MAP_4WAY = {"CAT-SIGN": 0, "CAT-PARITY": 1, "CTRL-NEU": 2, "CTRL-NUM": 3}
CATEGORY_2CLASS = {"CAT-SIGN": "math", "CAT-PARITY": "math",
                   "CTRL-NEU": "ctrl", "CTRL-NUM": "ctrl"}


def two_class_labels(categories: Sequence[str]) -> list[str]:
    """Collapse the 4 dataset categories to math vs ctrl."""
    return [CATEGORY_2CLASS[c] for c in categories]


def reduce_embeddings(representations, reducer: str = "pca", n_components: int = 2) -> np.ndarray:
    """Dispatch to PCA or (import-guarded) UMAP; returns (n, n_components)."""
    reducer = reducer.lower().strip()
    if reducer == "pca":
        return compute_pca_embeddings(representations, n_components=n_components)
    if reducer == "umap":
        return compute_umap_embeddings(representations, n_components=n_components)
    raise ValueError("reducer must be one of: pca, umap")


def plot_layer_category_figures(
    representations,
    categories: Sequence[str],
    layer: int,
    out_dir,
    reducer: str = "pca",
) -> "Figure":
    """Per-layer figures: the existing 4-way static PNG plus a 2-class (math vs ctrl)
    interactive HTML. At the terminal layer the title notes inter-category CKA≈0.01 ≈
    the matched baselines — i.e. no math-specific geometry; any visible separation in
    the top-2 components tracks the high-variance terminal-'=' positional axis (RQ1
    confound), not mathematical content. Returns the 2-class plotly figure (testability)."""
    from pathlib import Path
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    emb = reduce_embeddings(representations, reducer)
    tag = reducer.lower().strip()

    # 4-way static PNG (preserves the original behavior).
    labels4 = [LABEL_MAP_4WAY[c] for c in categories]
    fig4 = scatter_2d(emb, labels4, title=f"{tag.upper()} 4-way — Layer {layer:02d}",
                      backend="matplotlib")
    fig4.savefig(out_dir / f"{tag}_4way_layer_{layer:02d}.png", dpi=150)
    plt.close(fig4)

    # 2-class interactive HTML (math vs ctrl).
    title = f"{tag.upper()} math vs ctrl — Layer {layer:02d}"
    if int(layer) == 23:
        title += (" — inter-category CKA≈0.01 ≈ matched baselines: no math-specific "
                  "geometry (separation tracks the terminal-'=' positional axis)")
    fig2 = scatter_2d(emb, two_class_labels(categories), title=title, backend="plotly")
    fig2.write_html(str(out_dir / f"{tag}_2class_layer_{layer:02d}.html"))
    return fig2


def _to_numpy_2d(x: "torch.Tensor | np.ndarray") -> np.ndarray:
    """Convert tensor or array to float64 2-D numpy array."""
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    X = np.asarray(x)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D array [n, d], got shape {X.shape}")
    return X.astype(np.float64, copy=False)


if __name__ == "__main__":
    import argparse, json, sys
    from pathlib import Path
    from src.probing.io_utils import load_hidden_states

    parser = argparse.ArgumentParser(description="Math-vs-ctrl PCA/UMAP scatter per layer.")
    # Default emphasizes the terminal layer (figure 3: the inter-category overlap at L23).
    parser.add_argument("--layers", nargs="+", type=int, default=[23])
    parser.add_argument("--proc_dir", default="data/processed/pythia-1.4b")
    parser.add_argument("--out_dir", default="results/figures/pca")
    parser.add_argument("--reducer", default="pca", choices=["pca", "umap"])
    args = parser.parse_args()

    proc  = Path(args.proc_dir)
    out   = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    with open(proc / "metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    cats  = meta["categories"]

    for layer in args.layers:
        H = load_hidden_states(proc / f"layer_{layer:02d}.pt")
        plot_layer_category_figures(H, cats, layer, out, reducer=args.reducer)
        print(f"[OK] Layer {layer:02d} → {out} ({args.reducer}: 4-way PNG + 2-class HTML)")