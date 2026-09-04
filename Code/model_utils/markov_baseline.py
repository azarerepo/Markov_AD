### This module contains functions for the baseline Markov model

### functions included:

# estimate_transition_matrix()
# compute_transition_confidence_intervals()
# plot_transition_matrix()
# predict_markov()


# Ali Zare (zareali@msu.edu, ali.zr1983@gmail.com, ali.zare@duke.edu)

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def estimate_transition_matrix(transition_df: pd.DataFrame,
    current_state_col: str = "current_state_id",
    next_state_col: str = "next_state_id",
    states: Sequence[Any] = (0, 1, 2),
    state_labels: Sequence[str] | None = ("CN", "MCI", "AD"),
    smoothing: float = 0.0
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

    """
    Estimate a population-level transition matrix from observed transitions.

    For states i and j:

        P[i, j] = count(i to j) / total transitions originating from i

    Inputs:
    transition_df
        Transition dataset containing one row per observed transition.

    current_state_col
        Column containing the current state.

    next_state_col
        Column containing the next state.

    states
        Ordered state values used in the transition columns. For integer-
        encoded states, this would normally be (0, 1, 2).

    state_labels
        Human-readable labels corresponding to `states`. Set to None to use
        the state values directly.

    smoothing
        Additive smoothing applied to every transition count. A value of 0
        gives the empirical maximum-likelihood estimates. For example,
        smoothing=1 applies Laplace smoothing.


    Returns:
    probability_df
        Row-normalized transition probability matrix.

    count_df
        Observed transition count matrix before smoothing.

    """

    if not isinstance(transition_df, pd.DataFrame):
        raise TypeError("transition_df must be a pandas DataFrame.")

    required_cols = {current_state_col, next_state_col}
    missing_cols = required_cols.difference(transition_df.columns)

    if missing_cols:
        raise ValueError(
            f"Missing required columns: {sorted(missing_cols)}")

    if smoothing < 0:
        raise ValueError("smoothing must be nonnegative.")

    states = list(states)

    if len(states) == 0:
        raise ValueError("states must contain at least one state.")

    if len(set(states)) != len(states):
        raise ValueError("states must not contain duplicates.")

    if state_labels is None:
        labels = list(states)
    else:
        labels = list(state_labels)

        if len(labels) != len(states):
            raise ValueError(
                "state_labels must have the same length as states.")

    observed_current = set(
        transition_df[current_state_col].dropna().unique())
    observed_next = set(
        transition_df[next_state_col].dropna().unique())
    unknown_states = (observed_current | observed_next) - set(states)

    if unknown_states:
        raise ValueError(
            "The transition data contain states not included in `states`: "
            f"{sorted(unknown_states, key = str)}")

    # Observed transition counts.
    count_table = pd.crosstab(
        transition_df[current_state_col],
        transition_df[next_state_col])

    count_table = count_table.reindex(
        index = states,
        columns = states,
        fill_value = 0).astype(int)

    count_df = pd.DataFrame(
        count_table.to_numpy(),
        index = labels,
        columns = labels)
    count_df.index.name = "current_state"
    count_df.columns.name = "next_state"

    # Smoothing is used only for probability estimation. Returned counts
    # remain the actual observed counts.
    adjusted_counts = count_table.to_numpy(dtype = float) + smoothing
    row_totals = adjusted_counts.sum(axis = 1, keepdims = True)

    probabilities = np.divide(
        adjusted_counts,
        row_totals,
        out = np.full_like(adjusted_counts, np.nan, dtype = float),
        where = row_totals > 0)

    probability_df = pd.DataFrame(
        probabilities,
        index = labels,
        columns = labels)
    probability_df.index.name = "current_state"
    probability_df.columns.name = "next_state"

    return probability_df, count_df

#####################################################
#####################################################

def compute_transition_confidence_intervals(
    data: pd.DataFrame | np.ndarray,
    method: Literal["wilson", "bootstrap"] = "wilson",
    confidence_level: float = 0.95,
    current_state_col: str = "current_state_id",
    next_state_col: str = "next_state_id",
    subject_col: str = "subject_id",
    states: Sequence[Any] = (0, 1, 2),
    state_labels: Sequence[str] | None = ("CN", "MCI", "AD"),
    n_bootstrap: int = 2000,
    random_state: int | None = 42
    ) -> pd.DataFrame:

    """
    Compute confidence intervals for transition probabilities.

    Inputs:
    data
        For method="wilson":
            Either a square count matrix or a transition DataFrame.

        For method="bootstrap":
            Must be the transition DataFrame because subjects are resampled.

    method
        "wilson" or "bootstrap".

    confidence_level
        Desired confidence level, such as 0.95.

    current_state_col, next_state_col
        Current-state and next-state columns in the transition DataFrame.

    subject_col
        Subject identifier used for subject-level bootstrap resampling.

    states
        Ordered state values, for example (0, 1, 2).

    state_labels
        Human-readable state labels.

    n_bootstrap
        Number of bootstrap samples.

    random_state
        Random seed.

    Returns:
    pd.DataFrame
        One row per possible transition, containing its observed count,
        probability estimate, and confidence interval.
    """

    if method not in {"wilson", "bootstrap"}:
        raise ValueError("method must be either 'wilson' or 'bootstrap'.")

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1.")

    states = list(states)

    if len(states) == 0:
        raise ValueError("states must not be empty.")

    if len(set(states)) != len(states):
        raise ValueError("states must not contain duplicates.")

    if state_labels is None:
        labels = list(states)
    else:
        labels = list(state_labels)

        if len(labels) != len(states):
            raise ValueError("state_labels must have the same length as states.")

    state_to_position = {state: index
        for index, state in enumerate(states)}

    ####################################################
    # Construct the observed transition count matrix
    ####################################################
    if isinstance(data, pd.DataFrame):
        required = {current_state_col, next_state_col}

        if method == "bootstrap":
            required.add(subject_col)

        missing = required.difference(data.columns)

        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        observed_states = set(
            data[current_state_col].dropna().unique()
            ) | set(
            data[next_state_col].dropna().unique())

        unknown_states = observed_states - set(states)

        if unknown_states:
            raise ValueError("The data contain unknown states: "
                f"{sorted(unknown_states, key = str)}")

        count_table = pd.crosstab(
            data[current_state_col],
            data[next_state_col]).reindex(
                index = states,
                columns = states,
                fill_value = 0)

        observed_counts = count_table.to_numpy(dtype = int)

    else:
        if method == "bootstrap":
            raise TypeError(
                "Bootstrap intervals require the transition DataFrame, "
                "not only a count matrix.")

        observed_counts = np.asarray(data)

        if (observed_counts.ndim != 2
            or observed_counts.shape[0] != observed_counts.shape[1]):
            raise ValueError("The count matrix must be square.")

        if observed_counts.shape[0] != len(states):
            raise ValueError("The count matrix size must match the number of states.")

        if np.isnan(observed_counts).any():
            raise ValueError("The count matrix contains missing values.")

        if (observed_counts < 0).any():
            raise ValueError("Transition counts must be nonnegative.")

        if not np.allclose(observed_counts, np.round(observed_counts)):
            raise ValueError("Transition counts must be integer-valued.")

        observed_counts = observed_counts.astype(int)

    row_totals = observed_counts.sum(axis = 1)

    observed_probabilities = np.divide(
        observed_counts,
        row_totals[:, None],
        out = np.full(observed_counts.shape, np.nan, dtype = float),
        where = row_totals[:, None] > 0)

    alpha = 1.0 - confidence_level

    ####################################################
    # Wilson confidence intervals
    ####################################################
    if method == "wilson":
        z = norm.ppf(1.0 - alpha / 2.0)

        lower_bounds = np.full_like(observed_probabilities, np.nan)
        upper_bounds = np.full_like(observed_probabilities, np.nan)

        for i in range(len(states)):
            n = row_totals[i]

            if n == 0:
                continue

            for j in range(len(states)):
                probability = observed_probabilities[i, j]

                denominator = 1.0 + z**2 / n

                center = (probability + z**2 / (2.0 * n)) / denominator

                margin = z * np.sqrt(probability *
                                      (1.0 - probability) / n
                                      + z**2 / (4.0 * n**2) ) / denominator

                lower_bounds[i, j] = max(0.0, center - margin)

                upper_bounds[i, j] = min(1.0, center + margin)

    ####################################################
    # Subject-level bootstrap confidence intervals
    ####################################################
    else:
        if n_bootstrap < 2:
            raise ValueError("n_bootstrap must be at least 2.")

        bootstrap_df = data[
            [subject_col, current_state_col, next_state_col]
        ].dropna().copy()

        subjects = bootstrap_df[subject_col].unique()

        if len(subjects) < 2:
            raise ValueError(
                "At least two subjects are required for bootstrap "
                "confidence intervals.")

        subject_groups = {
            subject: group
            for subject, group in bootstrap_df.groupby(
                subject_col,
                sort = False)
            }

        rng = np.random.default_rng(random_state)

        bootstrap_probabilities = np.full(
            (n_bootstrap, len(states), len(states)),
            np.nan,
            dtype = float)

        for bootstrap_index in range(n_bootstrap):
            sampled_subjects = rng.choice(
                subjects,
                size = len(subjects),
                replace = True)

            bootstrap_counts = np.zeros((len(states), len(states)), dtype = int)

            # A subject selected more than once contributes its complete
            # transition history more than once.
            for subject in sampled_subjects:
                group = subject_groups[subject]

                for current_state, next_state in zip(
                    group[current_state_col],
                    group[next_state_col]):
                    i = state_to_position[current_state]
                    j = state_to_position[next_state]

                    bootstrap_counts[i, j] += 1

            bootstrap_row_totals = bootstrap_counts.sum(axis = 1)

            bootstrap_probabilities[bootstrap_index] = np.divide(
                bootstrap_counts, bootstrap_row_totals[:, None],
                out = np.full(bootstrap_counts.shape, np.nan, dtype = float),
                where = bootstrap_row_totals[:, None] > 0)

        lower_percentile = 100.0 * alpha / 2.0
        upper_percentile = 100.0 * (1.0 - alpha / 2.0)

        lower_bounds = np.nanpercentile(
            bootstrap_probabilities,
            lower_percentile,
            axis = 0)

        upper_bounds = np.nanpercentile(
            bootstrap_probabilities,
            upper_percentile,
            axis = 0)

    ####################################################
    # Long-format results
    ####################################################
    results = []

    for i, current_label in enumerate(labels):
        for j, next_label in enumerate(labels):
            valid_bootstrap_samples = np.nan

            if method == "bootstrap":
                valid_bootstrap_samples = int(
                    np.isfinite(bootstrap_probabilities[:, i, j]).sum())

            results.append(
                {"current_state": current_label,
                 "next_state": next_label,
                 "count": int(observed_counts[i, j]),
                 "row_total": int(row_totals[i]),
                 "probability": observed_probabilities[i, j],
                 "lower_bound": lower_bounds[i, j],
                 "upper_bound": upper_bounds[i, j],
                 "confidence_level": confidence_level,
                 "method": method,
                 "valid_bootstrap_samples":
                        valid_bootstrap_samples
                })

    return pd.DataFrame(results)

#####################################################
#####################################################

def plot_transition_matrix(
    matrix: pd.DataFrame | np.ndarray,
    state_labels: Sequence[str] | None = None,
    matrix_type: Literal["probability", "count"] = "probability",
    title: str | None = None,
    annotate: bool = True,
    decimals: int = 3,
    cmap: str = "Blues",
    ax: plt.Axes | None = None,
    show: bool = True
    ) -> tuple[plt.Figure, plt.Axes]:

    """
    This function plots a transition probability or
    count matrix as a heatmap.

    Inputs:
    matrix
        Square probability or count matrix.

    state_labels
        Labels to use when `matrix` is a NumPy array. If `matrix` is a
        DataFrame, its index and columns are used by default.

    matrix_type
        Either "probability" or "count".

    title
        Optional plot title.

    annotate
        Whether to write values inside the cells.

    decimals
        Number of decimal places for probability annotations.

    cmap
        Matplotlib colormap.

    ax
        Optional existing Matplotlib axes.

    show
        Whether to call plt.show().

    
    Returns:
    fig, ax
        Matplotlib figure and axes.

    """

    if matrix_type not in {"probability", "count"}:
        raise ValueError("matrix_type must be 'probability' or 'count'.")

    if isinstance(matrix, pd.DataFrame):
        values = matrix.to_numpy(dtype = float)
        row_labels = [str(label) for label in matrix.index]
        col_labels = [str(label) for label in matrix.columns]
    else:
        values = np.asarray(matrix, dtype = float)

        if state_labels is None:
            state_labels = [str(i) for i in range(values.shape[0])]

        row_labels = [str(label) for label in state_labels]
        col_labels = [str(label) for label in state_labels]

    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("matrix must be a square two-dimensional matrix.")

    if len(row_labels) != values.shape[0]:
        raise ValueError("The number of state labels must match the matrix size.")

    if matrix_type == "probability":
        finite_values = values[np.isfinite(values)]

        if finite_values.size and (
            finite_values.min() < 0 or finite_values.max() > 1
            ):
            raise ValueError("A probability matrix must contain values from 0 to 1.")

    if ax is None:
        fig, ax = plt.subplots(
            figsize = (1.7 * values.shape[1] + 2, 1.4 * values.shape[0] + 1)
            )
    else:
        fig = ax.figure

    if matrix_type == "probability":
        image = ax.imshow(values, cmap = cmap, vmin = 0, vmax = 1)
    else:
        image = ax.imshow(values, cmap = cmap)

    ax.set_xticks(np.arange(values.shape[1]))
    ax.set_yticks(np.arange(values.shape[0]))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")

    if title is None:
        title = ("Transition Probability Matrix"
            if matrix_type == "probability"
            else "Transition Count Matrix")

    ax.set_title(title)

    colorbar = fig.colorbar(image, ax = ax)

    if matrix_type == "probability":
        colorbar.set_label("Probability")
    else:
        colorbar.set_label("Count")

    if annotate:
        finite_values = values[np.isfinite(values)]
        threshold = (float(finite_values.max()) / 2.0
            if finite_values.size
            else 0.0)

        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]

                if np.isnan(value):
                    text = "NA"
                elif matrix_type == "count":
                    text = f"{int(round(value))}"
                else:
                    text = f"{value:.{decimals}f}"

                text_color = ("white"
                    if np.isfinite(value) and value > threshold
                    else "black")

                ax.text(j, i, text,
                    ha = "center",
                    va = "center",
                    color = text_color)

    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor = True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor = True)
    ax.grid(which = "minor",
        color = "white",
        linestyle = "-",
        linewidth = 1.5)
    ax.tick_params(which = "minor", bottom = False, left = False)

    fig.tight_layout()

    if show:
        plt.show(block = False)

    return fig, ax


#####################################################
#####################################################

def plot_transition_matrix_with_ci(
    ci_df: pd.DataFrame,
    state_labels = None,
    title = None,
    decimals = 3,
    cmap = "Blues",
    ax = None,
    show = True
    ):

    """
    Plot a transition probability matrix with confidence intervals
    displayed inside each heatmap cell.

    Each cell is shown as:

        probability
        [lower, upper]

    Inputs: 
    ci_df : pd.DataFrame
        Output from compute_transition_confidence_intervals().
        Must contain columns:

            current_state
            next_state
            probability
            lower_bound
            upper_bound

    state_labels : sequence, optional
        Desired ordering of states, e.g.
            ("CN", "MCI", "AD")

        If None, the order is inferred from ci_df.

    title : str, optional
        Plot title.

    decimals : int
        Number of decimal places shown.

    cmap : str
        Matplotlib colormap.

    ax : matplotlib.axes.Axes, optional
        Existing axes.

    show : bool
        Whether to call plt.show().

    Returns:
    fig, ax

    """

    ###########################################################
    # Validate required columns
    required_cols = {
        "current_state",
        "next_state",
        "probability",
        "lower_bound",
        "upper_bound"
        }

    missing = required_cols.difference(ci_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    ###########################################################
    # Determine state ordering
    if state_labels is None:
        # Preserve order of appearance
        state_labels = list(
            pd.unique(
                pd.concat(
                    [ci_df["current_state"], ci_df["next_state"]],
                    ignore_index = True)))

    else:
        state_labels = list(state_labels)

    n_states = len(state_labels)

    ###########################################################
    # Convert long-form table into matrices
    probability_matrix = (ci_df.pivot(index = "current_state",
            columns = "next_state",
            values = "probability"
            ).reindex(index = state_labels, columns = state_labels))

    lower_matrix = (ci_df.pivot(index = "current_state",
            columns = "next_state",
            values = "lower_bound"
            ).reindex(index = state_labels, columns = state_labels))

    upper_matrix = (ci_df.pivot(index = "current_state",
            columns = "next_state",
            values = "upper_bound"
            ).reindex(index = state_labels, columns = state_labels))

    probabilities = probability_matrix.to_numpy(dtype = float)
    lower = lower_matrix.to_numpy(dtype = float)
    upper = upper_matrix.to_numpy(dtype = float)

    ###########################################################
    # Create figure
    if ax is None:
        fig, ax = plt.subplots(figsize = (2.2*n_states + 2, 1.8*n_states + 1))
    else:
        fig = ax.figure

    ###########################################################
    # Heatmap
    image = ax.imshow(probabilities, cmap = cmap, vmin = 0, vmax = 1)

    ###########################################################
    # Axis labels
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(n_states))
    ax.set_xticklabels(state_labels)
    ax.set_yticklabels(state_labels)
    ax.set_xlabel("Next state")
    ax.set_ylabel("Current state")

    if title is None:

        method = None

        if ("method" in ci_df.columns
            and ci_df["method"].nunique() == 1):
            method = ci_df["method"].iloc[0]

        confidence = None

        if ("confidence_level" in ci_df.columns
            and ci_df["confidence_level"].nunique() == 1):
            confidence = (100 * ci_df["confidence_level"].iloc[0])

        if (method is not None and confidence is not None):

            title = ("Transition Probability Matrix\n"
                f"{confidence:.0f}% "
                f"{method.capitalize()} Confidence Intervals")

        else:

            title = ("Transition Probability Matrix "
                "with Confidence Intervals")

    ax.set_title(title)

    ###########################################################
    # Color bar
    colorbar = fig.colorbar(image, ax = ax)
    colorbar.set_label("Transition probability")

    ###########################################################
    # Cell annotations
    for i in range(n_states):
        for j in range(n_states):
            p = probabilities[i, j]
            lo = lower[i, j]
            hi = upper[i, j]

            if np.isnan(p):
                text = "NA"

            elif (np.isnan(lo) or np.isnan(hi)):
                text = (f"{p:.{decimals}f}\n" "[NA, NA]")

            else:
                text = (f"{p:.{decimals}f}\n"
                    f"[{lo:.{decimals}f}, "
                    f"{hi:.{decimals}f}]")

            # Contrast text against background
            if np.isfinite(p) and p > 0.5:
                text_color = "white"
            else:
                text_color = "black"

            ax.text(j, i, text,
                ha = "center",
                va = "center",
                color = text_color,
                fontsize = 10)

    ###########################################################
    # Grid separating cells
    ax.set_xticks(np.arange(-0.5, n_states, 1), minor = True)
    ax.set_yticks(np.arange(-0.5, n_states, 1), minor = True)
    ax.grid(which = "minor", color = "white",
        linestyle = "-", linewidth = 1.5)
    ax.tick_params(which = "minor", bottom = False, left = False)

    fig.tight_layout()

    if show:
        plt.show(block = False)

    return fig, ax


#####################################################
#####################################################

def predict_markov(
    current_states: Sequence[Any] | pd.Series | np.ndarray,
    transition_matrix: pd.DataFrame | np.ndarray,
    states: Sequence[Any] | None = None,
    return_classes: bool = False
    ) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:

    """
    Baseline Markov prediction: For each transition, the simple
    Markov model returns the row of the transition matrix
    corresponding to the subject's current state.

    Goal: Predict next-state probabilities using a baseline Markov matrix.

    All observations with the same current state receive the same probability
    distribution.

    Inputs:
    current_states
        Current state for each observation. These may be integer state IDs
        or string labels, as long as they match the transition matrix.

    transition_matrix
        Row-normalized transition probability matrix.

    states
        State order when transition_matrix is a NumPy array. Not needed for
        a labeled DataFrame.

    return_classes
        If True, also return the most likely predicted next state.

    Returns:
    probability_df
        Predicted probability distribution for every observation.

    predicted_classes
        Returned only if return_classes = True.

    """
    current_states_array = np.asarray(current_states)

    if current_states_array.ndim != 1:
        raise ValueError("current_states must be one-dimensional.")

    if isinstance(transition_matrix, pd.DataFrame):
        probability_matrix = transition_matrix.to_numpy(dtype = float)
        matrix_states = list(transition_matrix.index)
        next_state_labels = list(transition_matrix.columns)

        if matrix_states != next_state_labels:
            raise ValueError(
                "The transition matrix index and columns must use "
                "the same ordered states.")
    else:
        probability_matrix = np.asarray(transition_matrix, dtype = float)

        if states is None:
            states = list(range(probability_matrix.shape[0]))

        matrix_states = list(states)
        next_state_labels = list(states)

    if (probability_matrix.ndim != 2
        or probability_matrix.shape[0] != probability_matrix.shape[1]):
        raise ValueError("Transition_matrix must be a square two-dimensional matrix.")

    if len(matrix_states) != probability_matrix.shape[0]:
        raise ValueError("The number of states must match the transition matrix size.")

    if np.isnan(probability_matrix).any():
        raise ValueError(
            "Transition_matrix contains missing probabilities. This may "
            "occur when a state has no observed outgoing transitions.")

    if (probability_matrix < 0).any():
        raise ValueError("Transition_matrix cannot contain negative probabilities.")

    row_sums = probability_matrix.sum(axis = 1)

    if not np.allclose(row_sums, 1.0):
        raise ValueError("Every transition matrix row must add up to 1.")

    state_to_row = {
        state: index for index, state in enumerate(matrix_states)
        }

    unknown_states = set(current_states_array) - set(matrix_states)

    if unknown_states:
        raise ValueError(
            "current_states contain values not represented in the "
            f"transition matrix: {sorted(unknown_states, key = str)}")

    row_indices = np.array(
        [state_to_row[state] for state in current_states_array],
        dtype = int)

    predicted_probabilities = probability_matrix[row_indices]

    probability_columns = [f"P_next_{state}" for state in next_state_labels]

    probability_df = pd.DataFrame(
        predicted_probabilities,
        columns = probability_columns,
        index = (current_states.index
            if isinstance(current_states, pd.Series)
            else None)
        )

    if not return_classes:
        return probability_df

    predicted_indices = np.argmax(predicted_probabilities, axis = 1)

    predicted_classes = np.asarray(
        [next_state_labels[index] for index in predicted_indices]
    )

    return probability_df, predicted_classes


