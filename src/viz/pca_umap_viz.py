"""
Funzioni di supporto per visualizzazioni PCA/UMAP e scatter plot 2D.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - dipendenza opzionale
    torch = None


def compute_pca_embeddings(
    representations: "torch.Tensor | np.ndarray",
    n_components: int = 2,
    whiten: bool = False,
) -> np.ndarray:
    """
    Applica PCA alle rappresentazioni per ottenere un embedding a bassa
    dimensione (tipicamente 2D o 3D).
    """
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover
        raise ImportError("scikit-learn non installato. Esegui: pip install scikit-learn") from exc

    X = _to_numpy_2d(representations)
    if n_components <= 0:
        raise ValueError("n_components deve essere > 0")
    if n_components > X.shape[1]:
        raise ValueError(
            f"n_components={n_components} > dimensione feature={X.shape[1]}"
        )

    model = PCA(n_components=n_components, whiten=whiten, svd_solver="auto")
    return model.fit_transform(X)


def compute_umap_embeddings(
    representations: "torch.Tensor | np.ndarray",
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
    metric: str = "euclidean",
) -> np.ndarray:
    """
    Applica UMAP per ottenere una proiezione non lineare.
    """
    try:
        import umap  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError("umap-learn non installato. Esegui: pip install umap-learn") from exc

    X = _to_numpy_2d(representations)
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    return reducer.fit_transform(X)


def scatter_2d(
    embeddings_2d: np.ndarray,
    labels: Optional[Sequence[int]] = None,
    *,
    title: str | None = None,
    palette: Optional[Sequence[str]] = None,
    backend: str = "matplotlib",
) -> "Figure":
    """
    Genera uno scatter plot 2D colorato per etichetta.
    """
    X = np.asarray(embeddings_2d)
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("embeddings_2d deve avere forma [n, 2]")

    if labels is not None and len(labels) != X.shape[0]:
        raise ValueError("labels deve avere la stessa lunghezza di embeddings_2d")

    backend_norm = backend.lower().strip()
    if backend_norm in {"matplotlib", "seaborn"}:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        if labels is None:
            ax.scatter(X[:, 0], X[:, 1], s=20, alpha=0.8)
        else:
            labels_arr = np.asarray(labels)
            unique = np.unique(labels_arr)
            cmap = palette if palette is not None else [None] * len(unique)
            for idx, lab in enumerate(unique):
                mask = labels_arr == lab
                color = cmap[idx] if idx < len(cmap) else None
                ax.scatter(
                    X[mask, 0],
                    X[mask, 1],
                    s=20,
                    alpha=0.8,
                    c=color,
                    label=str(lab),
                )
            ax.legend(title="label", loc="best", frameon=False)

        ax.set_xlabel("dim_1")
        ax.set_ylabel("dim_2")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return fig

    if backend_norm == "plotly":
        import plotly.graph_objects as go

        if labels is None:
            trace = go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode="markers",
                marker={"size": 6, "opacity": 0.85},
                name="points",
            )
            fig = go.Figure(data=[trace])
        else:
            labels_arr = np.asarray(labels)
            traces = []
            for lab in np.unique(labels_arr):
                mask = labels_arr == lab
                traces.append(
                    go.Scatter(
                        x=X[mask, 0],
                        y=X[mask, 1],
                        mode="markers",
                        marker={"size": 6, "opacity": 0.85},
                        name=str(lab),
                    )
                )
            fig = go.Figure(data=traces)

        fig.update_layout(
            title=title or "2D scatter",
            xaxis_title="dim_1",
            yaxis_title="dim_2",
            template="plotly_white",
        )
        return fig

    raise ValueError("backend deve essere uno tra: matplotlib, seaborn, plotly")


def _to_numpy_2d(representations: "torch.Tensor | np.ndarray") -> np.ndarray:
    """Converte input a ndarray 2D float64."""
    if torch is not None and isinstance(representations, torch.Tensor):
        X = representations.detach().cpu().numpy()
    else:
        X = np.asarray(representations)

    if X.ndim != 2:
        raise ValueError(f"Atteso array 2D [n, d], ricevuto shape={X.shape}")
    return X.astype(np.float64, copy=False)

