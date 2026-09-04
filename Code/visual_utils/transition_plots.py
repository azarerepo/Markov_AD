"""
transition_plots.py

Visualization functions for transition datasets.

Ali Zare (zareali@msu.edu, ali.zr1983@gmail.com, ali.zare@duke.edu)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from general_utils.transition_utils import (
    compute_transition_table,
    compute_state_distribution
    )

from pathlib import Path

import networkx as nx
import plotly.graph_objects as go

# ============================================================
# Internal Helpers
# ============================================================

def _create_figure(ax = None, figsize = (6, 5)):
    """
    Create a figure if one does not already exist.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize = figsize)
    else:
        fig = ax.figure

    return fig, ax



def _finalize_plot(fig, save_path = None, show = True, dpi = 300):
    """
    Finalize a figure.

    Inputs:
    fig : matplotlib.figure.Figure

    save_path : str or pathlib.Path, optional
        If provided, save the figure.

    show : bool, default=True
        Whether to display the figure.

    dpi : int, default=300
        Resolution used when saving.
    """

    fig.tight_layout()

    if save_path is not None:

        save_path = Path(save_path)

        save_path.parent.mkdir(parents = True, exist_ok = True)

        fig.savefig(save_path, dpi = dpi, bbox_inches = "tight")

    if show:
        plt.show(block = False)

    return fig


def _plot_heatmap(table, title, xlabel, ylabel,
    cmap = "Blues",
    annotate = True,
    decimals = None,
    colorbar = True,
    vmin = None,
    vmax = None,
    fontsize = 12,
    ax = None,
    figsize = (6, 5),
    save_path = None,
    show = True
    ):

    """
    Generic heatmap plotting routine.

    Inputs:
    table : pandas.DataFrame

    decimals : int or None
        None -> integers
        int  -> floating point
    """

    fig, ax = _create_figure(ax, figsize)

    image = ax.imshow(table.values,
        cmap = cmap,
        aspect = "auto",
        vmin = vmin,
        vmax = vmax)

    if colorbar:
        fig.colorbar(image, ax = ax)

    ax.set_xticks(np.arange(table.shape[1]))
    ax.set_yticks(np.arange(table.shape[0]))

    ax.set_xticklabels(table.columns)
    ax.set_yticklabels(table.index)

    ax.set_xlabel(xlabel, fontsize = fontsize)
    ax.set_ylabel(ylabel, fontsize = fontsize)

    ax.set_title(title, fontsize = fontsize + 2)

    if annotate:

        for i in range(table.shape[0]):

            for j in range(table.shape[1]):

                value = table.iloc[i, j]

                if decimals is None:
                    text = f"{int(value)}"
                else:
                    text = f"{value:.{decimals}f}"

                if vmax is None:
                    threshold = table.values.max() / 2
                else:
                    threshold = vmax / 2

                color = "white" if value > threshold else "black"

                ax.text(j, i, text, ha = "center", va = "center",
                    color = color,
                    fontsize = fontsize - 1)

    _finalize_plot(fig, save_path = save_path, show = show)

    return fig, ax


# ============================================================
# Transition Count Heatmap
# ============================================================

def plot_transition_counts(transition_df,
    allowed_states = ("CN", "MCI", "AD"),
    cmap = "Blues",
    annotate = True,
    fontsize = 12,
    ax = None,
    figsize = (6, 5),
    save_path = None,
    show = True
    ):

    """
    Plot transition counts.
    """

    table = compute_transition_table(transition_df,
        allowed_states = allowed_states,
        normalize = False)

    return _plot_heatmap(table = table,
        title = "Observed Transition Counts",
        xlabel = "Next State",
        ylabel = "Current State",
        cmap = cmap,
        annotate = annotate,
        decimals = None,
        fontsize = fontsize,
        ax = ax,
        figsize = figsize,
        save_path = save_path,
        show = show)


# ============================================================
# Transition Probability Matrix
# ============================================================

def plot_transition_matrix(transition_df,
    allowed_states = ("CN", "MCI", "AD"),
    cmap = "Blues",
    annotate = True,
    decimals = 3,
    fontsize = 12,
    ax = None,
    figsize = (6, 5),
    save_path = None,
    show = True
    ):

    """
    Plot empirical Markov transition matrix.
    """

    table = compute_transition_table(
        transition_df,
        allowed_states = allowed_states,
        normalize = True)

    return _plot_heatmap(
        table = table,
        title = "Empirical Transition Matrix",
        xlabel = "Next State",
        ylabel = "Current State",
        cmap = cmap,
        annotate = annotate,
        decimals = decimals,
        fontsize = fontsize,
        ax = ax,
        figsize = figsize,
        save_path = save_path,
        show = show,
        vmin = 0,
        vmax = 1)


# ============================================================
# State Distribution
# ============================================================

def plot_state_distribution(transition_df,
    figsize = (10, 4),
    current_color = "steelblue",
    next_color = "darkorange",
    fontsize = 12,
    save_path = None,
    show = True
    ):

    """
    Plot current and next state distributions.
    """

    distrib = compute_state_distribution(transition_df)

    current = distrib["current"]
    nxt = distrib["next"]

    fig, axes = plt.subplots(1, 2, figsize = figsize)

    # ------------------------
    # Current state
    # ------------------------

    axes[0].bar(current.index, current.values, color = current_color)

    axes[0].set_title("Current State")

    axes[0].set_ylabel("Count")

    # ------------------------
    # Next state
    # ------------------------

    axes[1].bar(nxt.index, nxt.values, color = next_color)

    axes[1].set_title("Next State")

    axes[1].set_ylabel("Count")

    fig.suptitle("State Distribution", fontsize = fontsize + 2)

    _finalize_plot(fig, save_path = save_path, show = show)

    return fig, axes


# ============================================================
# Delta-T Histogram
# ============================================================

def plot_delta_t_histogram(transition_df,
    delta_col = "delta_t_months",
    bins = "auto",
    color = "steelblue",
    edgecolor = "black",
    alpha = 0.8,
    fontsize = 12,
    ax = None,
    figsize = (7, 5),
    save_path = None,
    show = True
    ):

    """
    Histogram of the time interval between visits.
    """

    fig, ax = _create_figure(ax, figsize)

    values = transition_df[delta_col].dropna()

    ax.hist(values, bins = bins, color = color,
            edgecolor = edgecolor, alpha = alpha)

    ax.set_xlabel("Time Between Visits (months)", fontsize = fontsize)
    ax.set_ylabel("Number of Transitions", fontsize = fontsize)

    ax.set_title("Distribution of Time Between Visits",
                 fontsize = fontsize + 2)

    ax.grid(alpha = 0.3)

    _finalize_plot(fig, save_path = save_path, show = show)

    return fig, ax


# ============================================================
# Delta-T by Transition Type
# ============================================================

def plot_delta_t_by_transition(transition_df,
    delta_col = "delta_t_months",
    current_col = "current_state",
    next_col = "next_state",
    fontsize = 12,
    ax = None,
    figsize = (9, 5),
    save_path = None,
    show = True
    ):

    """
    Boxplot of delta_t grouped by transition type.
    """

    fig, ax = _create_figure(ax, figsize)

    df = transition_df.copy()

    df["transition"] = df[current_col].astype(str) + " to " + df[next_col].astype(str)

    order = sorted(df["transition"].unique())

    data = [df.loc[df["transition"] == t,
                   delta_col].dropna().values for t in order]

    ax.boxplot(data, tick_labels = order, patch_artist = True)

    ax.set_xlabel("Transition Type", fontsize = fontsize)

    ax.set_ylabel("Time Between Visits (months)", fontsize = fontsize)

    ax.set_title("Time Between Visits by Transition Type", fontsize = fontsize + 2)

    ax.grid(alpha = 0.3)

    plt.setp(ax.get_xticklabels(), rotation = 30, ha = "right")

    _finalize_plot(fig, save_path = save_path, show = show)

    return fig, ax


# ============================================================
# Subject Transition Counts
# ============================================================

def plot_subject_transition_counts(transition_df,
    subject_col = "subject_id",
    bins = "auto",
    color = "darkorange",
    edgecolor = "black",
    alpha = 0.8,
    fontsize = 12,
    ax = None,
    figsize = (7, 5),
    save_path = None,
    show = True
    ):

    """
    Histogram of the number of transitions contributed
    by each subject.
    """

    fig, ax = _create_figure(ax, figsize)

    counts = transition_df.groupby(subject_col).size()

    ax.hist(counts.values, bins = bins,
        color = color,
        edgecolor = edgecolor,
        alpha = alpha)

    ax.set_xlabel("Transitions per Subject", fontsize = fontsize)

    ax.set_ylabel("Number of Subjects", fontsize = fontsize)

    ax.set_title("Subject Contribution", fontsize = fontsize + 2)

    ax.grid(alpha = 0.3)

    _finalize_plot(fig, save_path = save_path, show = show)

    return fig, ax


# ============================================================
# Regression Statistics
# ============================================================

def plot_regression_statistics(transition_df,
    current_state_col = "current_state_id",
    next_state_col = "next_state_id",
    fontsize = 12,
    colors = ("steelblue", "seagreen", "firebrick"),
    ax = None,
    figsize = (6, 5),
    save_path = None,
    show = True
    ):

    """
    Plot the numbers of stable, progressive,
    and regressive transitions.
    """

    required = {current_state_col, next_state_col}

    missing = required.difference(transition_df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    current = transition_df[current_state_col]
    nxt = transition_df[next_state_col]

    stable = np.sum(current == nxt)
    progressive = np.sum(nxt > current)
    regressive = np.sum(nxt < current)

    fig, ax = _create_figure(ax, figsize)

    labels = ["Stable", "Progressive", "Regressive"]

    values = [stable, progressive, regressive]

    ax.bar(labels, values, color = colors)

    for i, value in enumerate(values):
        ax.text(i, value, str(value), ha = "center", va = "bottom",
            fontsize = fontsize - 1)

    ax.set_ylabel("Number of Transitions", fontsize = fontsize)

    ax.set_title("Transition Categories", fontsize = fontsize + 2)

    ax.grid(alpha = 0.3)

    _finalize_plot(fig, save_path, show)

    return fig, ax


# ============================================================
# Transition Network
# ============================================================

def plot_transition_network(transition_df,
    allowed_states = ("CN", "MCI", "AD"),
    normalize = False,
    layout = "circular",
    node_size = 2500,
    edge_scale = 8,
    node_color = "lightsteelblue",
    font_size = 12,
    figsize = (6, 6),
    save_path = None,
    show = True
    ):

    """
    Draw a directed transition graph.
    """

    table = compute_transition_table(transition_df,
        allowed_states = allowed_states,
        normalize = normalize)

    G = nx.DiGraph()

    for state in allowed_states:
        G.add_node(state)

    for source in allowed_states:

        for target in allowed_states:

            weight = table.loc[source, target]

            if weight > 0:
                G.add_edge(source, target, weight=weight)

    if layout == "spring":
        pos = nx.spring_layout(G, seed = 42)

    elif layout == "shell":
        pos = nx.shell_layout(G)

    else:
        pos = nx.circular_layout(G)

    # fig, ax = plt.subplots(figsize = figsize)
    fig, ax = _create_figure(ax = None, figsize = figsize)

    nx.draw_networkx_nodes(G, pos, node_size = node_size,
        node_color = node_color,
        ax = ax)

    edge_widths = [
        edge_scale * G[u][v]["weight"]
        if normalize
        else
        np.sqrt(G[u][v]["weight"])
        for u, v in G.edges()
        ]

    nx.draw_networkx_edges(G, pos, width = edge_widths,
        arrows = True,
        arrowsize = 25,
        connectionstyle = "arc3,rad = 0.08",
        ax = ax)

    nx.draw_networkx_labels(G, pos, font_size = font_size, ax = ax)

    edge_labels = {
        (u, v): f"{G[u][v]['weight']:.2f}"
        if normalize else str(int(G[u][v]["weight"]))
        for u, v in G.edges()
        }

    nx.draw_networkx_edge_labels(G, pos, edge_labels = edge_labels,
        font_size = font_size - 2,
        ax = ax)

    ax.set_title("Transition Network", fontsize = font_size + 2)

    ax.axis("off")

    _finalize_plot(fig, save_path, show)

    return fig, ax


# ============================================================
# Sankey Diagram
# ============================================================

def plot_sankey(transition_df,
    allowed_states = ("CN", "MCI", "AD"),
    normalize = False,
    title = "Transition Sankey Diagram"
    ):

    """
    Interactive Sankey diagram.

    Returns:
    plotly.graph_objects.Figure
    """

    table = compute_transition_table(transition_df,
        allowed_states = allowed_states,
        normalize = normalize)

    node_names = list(allowed_states)

    source = []
    target = []
    value = []

    for i, current in enumerate(allowed_states):

        for j, nxt in enumerate(allowed_states):

            weight = table.loc[current, nxt]

            if weight == 0:
                continue

            source.append(i)
            target.append(j)
            value.append(weight)

    fig = go.Figure(
        go.Sankey(node = dict(pad = 20, thickness = 25, label = node_names),
            link = dict(source = source, target = target, value = value))
            )

    fig.update_layout(title = title, font_size = 12)

    fig.show(block = False)

    return fig, None
