"""pca_umap_viz.py — PCA/UMAP dimensionality reduction and 2-D scatter plots."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

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


def _to_numpy_2d(x: "torch.Tensor | np.ndarray") -> np.ndarray:
    """Convert tensor or array to float64 2-D numpy array."""
    if torch is not None and isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    X = np.asarray(x)
    if X.ndim != 2:
        raise ValueError(f"Expected 2-D array [n, d], got shape {X.shape}")
    return X.astype(np.float64, copy=False)