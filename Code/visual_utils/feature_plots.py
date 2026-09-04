
### Visualization functions for feature analysis of transition datasets.

# Ali Zare (zareali@msu.edu, ali.zr1983@gmail.com, ali.zare@duke.edu)



from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# Internal Helpers
# ============================================================

def _create_figure(ax = None, figsize = (7, 5)):
    """
    Create a figure if ax is None.

    Returns:
    fig, ax
    """

    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)
    else:
        fig = ax.figure

    return fig, ax


def _finalize_plot(fig, save_path: Optional[str] = None,
    show: bool = True,
    dpi: int = 300
    ):

    """
    Finalize a matplotlib figure.
    """

    fig.tight_layout()

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(parents = True, exist_ok = True)

        fig.savefig(save_path, dpi = dpi, bbox_inches = "tight")

    if show:
        plt.show(block = False)

    return fig


def _validate_feature(df: pd.DataFrame, feature: str):
    """
    Ensure that a feature exists and is numeric.
    """

    if feature not in df.columns:

        raise ValueError(f"'{feature}' not found in dataframe.")

    if not pd.api.types.is_numeric_dtype(df[feature]):

        raise ValueError(f"'{feature}' must be numeric.")


def _transition_labels(df, current_col = "current_state", next_col = "next_state"):

    """
    Return transition labels.

    Example:
    CN to MCI
    """

    return df[current_col].astype(str) + " to " + df[next_col].astype(str)


# ============================================================
# Feature Distribution
# ============================================================

def plot_feature_distribution(transition_df: pd.DataFrame,
    feature: str,
    bins = "auto",
    color = "steelblue",
    edgecolor = "black",
    alpha = 0.8,
    density = False,
    fontsize = 12,
    ax = None,
    figsize = (7, 5),
    save_path = None,
    show = True
    ):

    """
    Plot the distribution of a feature.

    Inputs:
    transition_df : DataFrame

    feature : str

    density : bool
        If True, plot density instead of counts.
    """

    _validate_feature(transition_df, feature)

    fig, ax = _create_figure(ax, figsize)

    values = transition_df[feature].dropna().values

    ax.hist(
        values,
        bins = bins,
        density = density,
        color = color,
        edgecolor = edgecolor,
        alpha = alpha)

    ax.set_xlabel(feature, fontsize = fontsize)

    ax.set_ylabel("Density" if density else "Count",
        fontsize = fontsize)

    ax.set_title(f"Distribution of {feature}", fontsize = fontsize + 2)

    ax.grid(alpha = 0.3)

    _finalize_plot(fig, save_path, show)

    return fig, ax


# ============================================================
# Feature by Transition
# ============================================================

def plot_feature_by_transition(transition_df: pd.DataFrame,
    feature: str,
    current_col = "current_state",
    next_col = "next_state",
    kind = "box",
    fontsize = 12,
    ax = None,
    figsize = (10, 5),
    save_path = None,
    show = True
    ):

    """
    Plot a feature grouped by transition type.

    Inputs:
    kind : {"box","violin"}

    Example:
    CN to CN
    CN to MCI
    MCI to MCI
    MCI to AD
    """

    _validate_feature(transition_df, feature)

    fig, ax = _create_figure(ax, figsize)

    df = transition_df.copy()

    df["transition"] = _transition_labels(df, current_col, next_col)

    order = sorted(df["transition"].unique())

    data = [
        df.loc[df["transition"] == t, feature].dropna().values
        for t in order
        ]

    if kind.lower() == "box":

        ax.boxplot(data, tick_labels = order, patch_artist = True)

    elif kind.lower() == "violin":

        parts = ax.violinplot(data, showmeans = False, showmedians = True)

        ax.set_xticks(np.arange(1, len(order)+1))

        ax.set_xticklabels(order)

    else:
        raise ValueError("kind must be 'box' or 'violin'.")

    ax.set_xlabel("Transition", fontsize = fontsize)

    ax.set_ylabel(feature, fontsize = fontsize)

    ax.set_title(f"{feature} by Transition Type",
        fontsize = fontsize + 2)

    plt.setp(ax.get_xticklabels(), rotation = 30, ha = "right")

    ax.grid(alpha = 0.3)

    _finalize_plot(fig, save_path, show)

    return fig, ax



# ============================================================
# Feature Correlation Heatmap
# ============================================================

def plot_feature_correlations(transition_df: pd.DataFrame,
    feature_cols,
    method: str = "pearson",
    annotate: bool = True,
    decimals: int = 2,
    cmap: str = "coolwarm",
    fontsize: int = 12,
    ax = None,
    figsize = (8, 7),
    save_path = None,
    show = True
    ):

    """
    Plot the correlation matrix of selected features.

    Inputs:
    feature_cols : list[str]

    method : {"pearson", "spearman", "kendall"}

    """

    missing = [f for f in feature_cols if f not in transition_df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    corr = transition_df[feature_cols].corr(method = method)

    fig, ax = _create_figure(ax, figsize)

    image = ax.imshow(
        corr.values,
        cmap = cmap,
        vmin = -1,
        vmax = 1,
        aspect = "auto")

    fig.colorbar(image, ax = ax)

    ax.set_xticks(np.arange(len(feature_cols)))

    ax.set_yticks(np.arange(len(feature_cols)))

    ax.set_xticklabels(feature_cols, rotation = 45, ha = "right")

    ax.set_yticklabels(feature_cols)

    ax.set_title(f"{method.capitalize()} Correlation Matrix",
                 fontsize = fontsize + 2)

    if annotate:

        for i in range(corr.shape[0]):

            for j in range(corr.shape[1]):

                value = corr.iloc[i, j]

                color = "white" if abs(value) > 0.5 else "black"

                ax.text(j, i,
                    f"{value:.{decimals}f}",
                    ha = "center",
                    va = "center",
                    color = color,
                    fontsize = fontsize - 2)

    _finalize_plot(fig, save_path, show)

    return fig, ax


# ============================================================
# Missing Value Heatmap
# ============================================================

def plot_missing_values(transition_df: pd.DataFrame,
    feature_cols=None,
    cmap = "Greys",
    fontsize = 12,
    ax = None,
    figsize = (10, 6),
    save_path = None,
    show = True
    ):

    """
    Visualize missing values.

    White = present
    Black = missing
    """

    if feature_cols is None:
        df = transition_df

    else:

        missing = [f for f in feature_cols
            if f not in transition_df.columns]

        if missing:
            raise ValueError(f"Missing columns: {missing}")

        df = transition_df[feature_cols]

    fig, ax = _create_figure(ax, figsize)

    image = ax.imshow(
        df.isna(),
        cmap = cmap,
        aspect = "auto",
        interpolation = "nearest")

    fig.colorbar(image, ax = ax)

    ax.set_title("Missing Values", fontsize = fontsize + 2)

    ax.set_xlabel("Features", fontsize = fontsize)

    ax.set_ylabel("Transition", fontsize = fontsize)

    ax.set_xticks(np.arange(df.shape[1]))

    ax.set_xticklabels(df.columns, rotation = 45, ha = "right")

    _finalize_plot(fig, save_path, show)

    return fig, ax


# ============================================================
# Scatter Matrix
# ============================================================

def plot_scatter_matrix(transition_df: pd.DataFrame,
    feature_cols,
    figsize = (10, 10),
    diagonal = "hist",
    alpha = 0.6,
    marker = "o",
    color = "steelblue",
    save_path = None,
    show = True
    ):

    """
    Scatter matrix of selected features.

    Inputs:
    feature_cols : list[str]

    diagonal : {"hist", "kde"}
    """

    missing = [f for f in feature_cols if f not in transition_df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    axes = pd.plotting.scatter_matrix(
        transition_df[feature_cols],
        figsize = figsize,
        diagonal = diagonal,
        alpha = alpha,
        marker = marker,
        color = color
        )

    fig = axes[0, 0].figure

    _finalize_plot(fig, save_path, show)

    return fig, axes



# ============================================================
# Internal Helper
# ============================================================

from sklearn.preprocessing import StandardScaler


def _prepare_embedding_data(transition_df: pd.DataFrame,
                            feature_cols,
                            color_by = None):

    """
    Prepare data for PCA/UMAP/t-SNE.

    Returns:
    X : ndarray
        Standardized feature matrix.

    labels : ndarray or None
        Labels used for coloring.
    """

    missing = [c for c in feature_cols
        if c not in transition_df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    X = transition_df[feature_cols].dropna()

    indices = X.index

    scaler = StandardScaler()

    X = scaler.fit_transform(X)

    labels = None

    if color_by is not None:
        if color_by not in transition_df.columns:
            raise ValueError(f"{color_by} not found.")

        labels = transition_df.loc[indices, color_by].values

    return X, labels


# ============================================================
# PCA
# ============================================================

def plot_pca(transition_df: pd.DataFrame,
    feature_cols,
    color_by = None,
    point_size = 40,
    alpha = 0.8,
    fontsize = 12,
    ax = None,
    figsize = (7, 6),
    save_path = None,
    show = True
    ):

    """
    PCA projection.

    Returns:
    fig, ax, pca

    """

    from sklearn.decomposition import PCA

    X, labels = _prepare_embedding_data(
        transition_df,
        feature_cols,
        color_by)

    pca = PCA(n_components = 2, random_state = 42)

    Z = pca.fit_transform(X)

    fig, ax = _create_figure(ax, figsize)

    if labels is None:
        ax.scatter(Z[:, 0], Z[:, 1], s = point_size, alpha = alpha)

    else:

        unique = np.unique(labels)

        for value in unique:
            mask = labels == value
            ax.scatter(Z[mask, 0], Z[mask, 1],
                s = point_size,
                alpha = alpha,
                label = str(value))

        ax.legend()

    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")

    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")

    ax.set_title("Principal Component Analysis", fontsize = fontsize + 2)

    ax.grid(alpha = 0.3)

    _finalize_plot(fig, save_path, show)

    return fig, ax, pca


# ============================================================
# PCA Loadings
# ============================================================

def plot_pca_loadings(pca, feature_cols,
    component = 1,
    fontsize = 12,
    ax = None,
    figsize = (8, 5),
    save_path = None,
    show = True
    ):

    """
    Plot feature loadings for one principal component.
    """

    if component < 1:
        raise ValueError("component must be >= 1")

    index = component - 1

    if index >= pca.components_.shape[0]:
        raise ValueError("PCA component does not exist.")

    fig, ax = _create_figure(ax, figsize)

    loadings = pca.components_[index]

    order = np.argsort(np.abs(loadings))[::-1]

    ax.bar(np.array(feature_cols)[order],
        loadings[order])

    ax.set_xticklabels(
        np.array(feature_cols)[order],
        rotation = 45,
        ha = "right")

    ax.set_ylabel("Loading")

    ax.set_title(f"PCA Component {component} Loadings",
        fontsize = fontsize + 2)

    ax.grid(alpha = 0.3)

    _finalize_plot(fig, save_path, show)

    return fig, ax


# ============================================================
# UMAP
# ============================================================

def plot_umap(transition_df, feature_cols,
    color_by = None,
    n_neighbors = 15,
    min_dist = 0.1,
    point_size = 40,
    alpha = 0.8,
    fontsize = 12,
    ax = None,
    figsize = (7,6),
    save_path = None,
    show = True
    ):

    """
    UMAP projection.
    """

    try:
        import umap
    except ImportError:
        raise ImportError("Install umap-learn to use plot_umap().")

    X, labels = _prepare_embedding_data(transition_df, feature_cols, color_by)

    reducer = umap.UMAP(n_neighbors = n_neighbors, min_dist = min_dist, random_state = 42)

    Z = reducer.fit_transform(X)

    fig, ax = _create_figure(ax, figsize)

    if labels is None:
        ax.scatter(Z[:,0], Z[:,1], s = point_size, alpha = alpha)

    else:
        for value in np.unique(labels):
            mask = labels == value
            ax.scatter(Z[mask,0], Z[mask,1],
                s = point_size,
                alpha = alpha,
                label = str(value))

        ax.legend()

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")

    ax.set_title("UMAP Projection", fontsize = fontsize+2)

    ax.grid(alpha = 0.3)

    _finalize_plot(fig, save_path, show)

    return fig, ax


# ============================================================
# t-SNE
# ============================================================

def plot_tsne(transition_df, feature_cols,
    color_by = None,
    perplexity = 30,
    point_size = 40,
    alpha = 0.8,
    fontsize = 12,
    ax = None,
    figsize = (7,6),
    save_path = None,
    show = True
    ):
    """
    t-SNE projection.
    """

    from sklearn.manifold import TSNE

    X, labels = _prepare_embedding_data(
        transition_df,
        feature_cols,
        color_by)

    tsne = TSNE(n_components = 2, perplexity = perplexity, random_state = 42)

    Z = tsne.fit_transform(X)

    fig, ax = _create_figure(ax, figsize)

    if labels is None:
        ax.scatter(Z[:,0], Z[:,1], s = point_size, alpha = alpha)

    else:
        for value in np.unique(labels):
            mask = labels == value
            ax.scatter(Z[mask,0], Z[mask,1],
                s = point_size,
                alpha = alpha,
                label = str(value))

        ax.legend()

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    ax.set_title("t-SNE Projection", fontsize = fontsize + 2)

    ax.grid(alpha = 0.3)

    _finalize_plot(fig, save_path, show)

    return fig, ax