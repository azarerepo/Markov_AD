### Some utility functions for calculations on the transition dataset

# Ali Zare (zareali@msu.edu, ali.zr1983@gmail.com, ali.zare@duke.edu)


import numpy as np
import pandas as pd


# ============================================================
# Transition Table
# ============================================================

def compute_transition_table(transition_df: pd.DataFrame,
    current_col: str = "current_state",
    next_col: str = "next_state",
    allowed_states = ("CN", "MCI", "AD"),
    normalize: bool = False
    ):

    """
    Compute the transition count/probability matrix.

    Returns:
    pandas.DataFrame
        Rows = current states
        Columns = next states
    """

    table = pd.crosstab(transition_df[current_col],
        transition_df[next_col],
        dropna = False)

    table = table.reindex(index = allowed_states,
        columns = allowed_states,
        fill_value = 0)

    if normalize:
        table = table.div(table.sum(axis = 1).replace(0, np.nan), axis = 0).fillna(0)

    return table


# ============================================================
# Transition Statistics
# ============================================================

def compute_transition_statistics(transition_df, allowed_states = ("CN", "MCI", "AD")):

    """
    Compute transition counts and probabilities.
    """

    counts = compute_transition_table(transition_df,
        allowed_states = allowed_states,
        normalize = False)

    probs = compute_transition_table(transition_df,
        allowed_states = allowed_states,
        normalize = True)

    return {"counts": counts, "probabilities": probs}


# ============================================================
# State Distribution
# ============================================================

def compute_state_distribution(transition_df,
    current_col = "current_state",
    next_col = "next_state"
    ):

    current = transition_df[current_col].value_counts().sort_index()

    nxt = transition_df[next_col].value_counts().sort_index()

    return {"current": current, "next": nxt}


# ============================================================
# Delta-T Statistics
# ============================================================

def compute_delta_t_statistics(transition_df, delta_col = "delta_t_months"):

    return {
        "mean": transition_df[delta_col].mean(),
        "median": transition_df[delta_col].median(),
        "std": transition_df[delta_col].std(),
        "min": transition_df[delta_col].min(),
        "max": transition_df[delta_col].max(),
        "values": transition_df[delta_col],
    }


# ============================================================
# Subject Statistics
# ============================================================

def compute_subject_statistics(transition_df, subject_col = "subject_id"):

    counts = transition_df.groupby(subject_col).size()

    return {
        "counts": counts,
        "mean": counts.mean(),
        "median": counts.median(),
        "std": counts.std(),
        "min": counts.min(),
        "max": counts.max()
        }


# ============================================================
# Regression Statistics
# ============================================================

def compute_regression_statistics(transition_df):

    if "is_regression" not in transition_df.columns:
        raise ValueError("Column 'is_regression' not found.")

    n_regression = transition_df["is_regression"].sum()

    n_total = len(transition_df)

    n_progressive = (transition_df["current_state_id"] < transition_df["next_state_id"]).sum()

    n_stable = (transition_df["current_state_id"] == transition_df["next_state_id"]).sum()

    return {
        "stable": n_stable,
        "progressive": n_progressive,
        "regressive": n_regression,
        "total": n_total
        }