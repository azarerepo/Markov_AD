# This module contains helper functions to:
# find common subjects among longitudinal and phenotype files,
# build a transition dataset from longitudinal data,
# work with sets (find intersection, build powerset, ...) 

# Ali Zare (zareali@msu.edu, ali.zr1983@gmail.com, ali.zare@duke.edu)

import numpy as np
from itertools import chain, combinations
from pathlib import Path
import pandas as pd
from tqdm import tqdm


def load_subjects(csv_file, big_flag = True, id_col = "NACCID"):
    """
    Read a CSV file and return the set of unique subject IDs.
    """
    if csv_file.lower().endswith((".csv")):
        if big_flag: # if it is UDS data
            df = pd.read_csv(csv_file, low_memory = False)
            df = filter_and_label_diagnosis(df)

        else:
            df = pd.read_csv(csv_file)

    elif csv_file.lower().endswith((".xlsx")):
        df = pd.read_excel(csv_file)
    
    return set(df[id_col].dropna().unique())

###############################################################

def load_subject_sets(files, id_col = "NACCID"):
    """
    Read multiple CSV files.

    Returns:
    dict
        {
            filename_without_extension : set(subject IDs),
            ...
        }
    """
    subject_sets = {}

    for f in files:
        name = Path(f).stem
        subject_sets[name] = load_subjects(f, False, id_col)

    return subject_sets

###############################################################

def powerset(items):
    """
    Generate all non-empty subsets of a set.

    Example:
    ['A','B','C'] outputs
        ('A',)
        ('B',)
        ('C',)
        ('A','B')
        ('A','C')
        ('B','C')
        ('A','B','C')
    """
    return chain.from_iterable(
        combinations(items, r)
        for r in range(1, len(items) + 1))

###############################################################

def intersection_of_sets(sets):
    """
    Compute the intersection of a collection of sets.
    """
    sets = list(sets)

    if not sets:
        return set()

    common = sets[0].copy()

    for s in sets[1:]:
        common &= s

    return common

###############################################################

def find_common_subjects(uds_subjects, phenotype_sets):
    """
    Find common subjects between UDS and every subset
    of the phenotype files.

    Inputs:
    uds_subjects : set

    phenotype_sets : dict
        {
            file_name : set(subject IDs),
            ...
        }

    Returns:
    list of dict

    """

    results = []

    names = list(phenotype_sets.keys())

    for subset in powerset(names):

        common = intersection_of_sets(
            [uds_subjects] +
            [phenotype_sets[name] for name in subset])

        results.append({
            "subset": subset,
            "subjects": common,
            "count": len(common)
            })

    return results

###############################################################

def print_results(results):
    for r in results:
        print(", ".join(r["subset"]))
        print(f"Subjects : {r['count']}")
        print()

###############################################################

def summarize_visits(uds_df,
                     common_subjects,
                     min_vis = 1,
                     id_col = "NACCID",
                     visit_col = "NACCAVST"):
    """
    Return the total number of visits for each common subject.

    Inputs:
    uds_df : pandas.DataFrame
        Longitudinal UDS data.
    common_subjects : set
        Set of common subject IDs.

    Returns:
    pandas.DataFrame
        Columns:
            NACCID
            Number_of_Visits
    """

    summary = (
        uds_df.loc[uds_df[id_col].isin(common_subjects),
                   [id_col, visit_col]]
        .drop_duplicates(subset = [id_col])
        .rename(columns = {visit_col: "Number_of_Visits"})
        .sort_values(id_col)
        .reset_index(drop = True))
    summary = summary[summary["Number_of_Visits"] >= min_vis]

    return summary.sort_values(id_col).reset_index(drop = True)

###############################################################

def visit_statistics(visit_summary):
    """
    Compute statistics of the number of visits.
    """

    visits = visit_summary["Number_of_Visits"]

    return {
        "subjects": len(visit_summary),
        "total_visits": visits.sum(),
        "mean_visits": visits.mean(),
        "median_visits": visits.median(),
        "std_visits": visits.std(),
        "min_visits": visits.min(),
        "max_visits": visits.max()
        }

###############################################################

# def save_results(results, output_dir = "common_subjects",
#                  id_col = "NACCID"):

#     output_dir = Path(output_dir)
#     output_dir.mkdir(exist_ok = True)

#     for r in results:

#         filename = "_".join(r["subset"]) + ".csv"

#         pd.DataFrame(
#             sorted(r["subjects"]),
#             columns = [id_col]
#         ).to_csv(output_dir / filename, index = False)

def save_results(results, uds_df, output_dir = "common_subjects", id_col = "NACCID"):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok = True)

    for r in results:

        subset_name = "_".join(r["subset"])

        visit_summary = summarize_visits(uds_df, r["subjects"])

        stats = visit_statistics(visit_summary)

        print(f"Subset: {' + '.join(r['subset'])}")
        print(f"Subjects      : {stats['subjects']}")
        print(f"Total visits  : {stats['total_visits']}")
        print(f"Mean visits   : {stats['mean_visits']:.2f}")
        print(f"Median visits : {stats['median_visits']:.1f}")
        print(f"Min visits    : {stats['min_visits']}")
        print(f"Max visits    : {stats['max_visits']}")
        print()

        visit_summary.to_csv(output_dir / f"{subset_name}.csv", index = False)

###############################################################

def final_results(results, uds_df, min_vis = 1):

    final = []

    for r in results:

        visit_summary = summarize_visits(uds_df, r["subjects"], min_vis = min_vis)

        # # Keep only subjects with at least min_vis visits
        # visit_summary = visit_summary[visit_summary["Number_of_Visits"] >= min_vis]

        stats = visit_statistics(visit_summary)

        final.append({
            "subset": r["subset"],
            "subjects": set(visit_summary["NACCID"]),
            "visit_summary": visit_summary,
            "stats": stats
        })

    return final

###############################################################

def filter_and_label_diagnosis(df):
    """
    Keep only CN, MCI, and AD visits defined based on criteria
    and create a DIAGNOSIS column.

    Label definitions:
    
    CN:
        NACCALZD = 8
        NACCUDSD = 1
        NORMCOG = 1
        DEMENTED = 0

    MCI:
        NACCALZD = 1
        NACCUDSD = 3
        NORMCOG = 0
        DEMENTED = 0

    AD:
        NACCALZD = 1
        NACCUDSD = 4
        NORMCOG = 0
        DEMENTED = 1
    """

    # Masks
    cn = ((df["NACCALZD"] == 8) & (df["NACCUDSD"] == 1) &
        (df["NORMCOG"] == 1) & (df["DEMENTED"] == 0))

    mci = ((df["NACCALZD"] == 1) & (df["NACCUDSD"] == 3) &
        (df["NORMCOG"] == 0) & (df["DEMENTED"] == 0))

    ad = ((df["NACCALZD"] == 1) & (df["NACCUDSD"] == 4) &
        (df["NORMCOG"] == 0) & (df["DEMENTED"] == 1))

    # Keep only valid rows
    df = df[cn | mci | ad].copy()

    # Create diagnosis column (hard-coded)
    # df["DIAGNOSIS"] = ""
    # df.loc[cn.loc[df.index], "DIAGNOSIS"] = "CN"
    # df.loc[mci.loc[df.index], "DIAGNOSIS"] = "MCI"
    # df.loc[ad.loc[df.index], "DIAGNOSIS"] = "AD"

    # Create diagnosis column
    df["DIAGNOSIS"] = np.select(
        [cn.loc[df.index], mci.loc[df.index], ad.loc[df.index]],
        ["CN", "MCI", "AD"],
        default = "")

    print("\nThe data were filtered to keep only CN, MCI, "
    "and AD subjects, and a DIAGNOSIS column was created "
    "with labels CN, MCI, and AD.")

    return df


###############################################################

def int_to_state(statetoint):
    """
    statetoint: dictionary mapping states to integers
    """
    inttostate = {i: state for state, i in statetoint.items()}
    return inttostate

###############################################################

def build_transition_dataset(
    longitudinal_df: pd.DataFrame,
    feature_cols: list,
    subject_col: str = "NACCID",
    diagnosis_col: str = "DIAGNOSIS",
    months_col: str = "months_from_baseline",
    allowed_states = ("CN", "MCI", "AD"),
    allow_regression = True,
    regression_action = "flag",   # "keep", "flag", "remove"
    drop_missing = False,
    return_summary = False
    ):

    """
    Build a transition dataset from longitudinal visit data.

    Each output row corresponds to one transition:

        Visit j  --->  Visit j+1

    Features are copied from Visit j.
    The target is the diagnosis at Visit j+1.


    Inputs:

    longitudinal_df : pd.DataFrame
        Longitudinal dataset sorted by subject and visit.

    feature_cols : list
        Feature columns copied from the current visit.

    subject_col : str

    diagnosis_col : str

    months_col : str

    allowed_states : tuple

    allow_regression : boolean
        Whether regressive transitions are allowed.

    regression_action : str
        "keep"   : keep all transitions
        "flag"   : keep and add is_regression column
        "remove" : remove regressive transitions

    drop_missing : boolean
        Remove transitions with missing feature values.

    return_summary : boolean
        Return summary dictionary.

    Returns:
    transition_df
    summary (optional)
    """

    #################################
    # Check columns
    #################################

    required = {subject_col, diagnosis_col, months_col, *feature_cols}

    missing = required.difference(longitudinal_df.columns)

    if len(missing) > 0:
        raise ValueError(f"Missing required columns:\n{sorted(missing)}")

    if regression_action not in {"keep", "flag", "remove"}:
        raise ValueError("regression_action must be 'keep', 'flag', or 'remove'.")

    # state_to_int = {"CN": 0, "MCI": 1, "AD": 2}
    state_to_int = {state: i for i, state in enumerate(allowed_states)}

    transition_rows = []

    transition_counts = {}

    n_subjects = 0
    n_skipped_subjects = 0

    transition_id = 0

    #################################
    ### Process each subject
    #################################

    for subject, df_sub in tqdm(longitudinal_df.groupby(subject_col, sort = False), desc = "Building Transition Dataset"):

        df_sub = df_sub.reset_index(drop = True)

        if len(df_sub) < 2:
            n_skipped_subjects += 1
            continue

        n_subjects += 1

        months = df_sub[months_col].values

        if np.any(np.diff(months) < 0):
            raise ValueError(f"{subject}: {months_col} is not sorted.")

        for j in range(len(df_sub) - 1):

            current = df_sub.iloc[j]
            nxt = df_sub.iloc[j + 1]

            current_state = current[diagnosis_col]
            next_state = nxt[diagnosis_col]

            if current_state not in state_to_int:
                raise ValueError(f"Unknown diagnosis: {current_state}")

            if next_state not in state_to_int:
                raise ValueError(f"Unknown diagnosis: {next_state}")

            is_regression = state_to_int[next_state] < state_to_int[current_state]

            if is_regression and not allow_regression:
                continue

            if (is_regression and regression_action == "remove"):
                continue

            row = {
                "transition_id": transition_id,
                "subject_id": subject,

                "current_visit_index": j,
                "next_visit_index": j + 1,

                "current_state": current_state,
                "next_state": next_state,

                "current_state_id": state_to_int[current_state],
                "next_state_id": state_to_int[next_state],

                "current_month":
                    current[months_col],
                "next_month":
                    nxt[months_col],

                "delta_t_months":
                    nxt[months_col] - current[months_col]
                }

            if regression_action == "flag":
                row["is_regression"] = is_regression

            # copy features from current visit
            for col in feature_cols:
                row[col] = current[col]

            if drop_missing:
                if pd.isnull(pd.Series([row[c] for c in feature_cols])).any():
                    continue

            transition_rows.append(row)

            key = (current_state, next_state)

            transition_counts[key] = transition_counts.get(key, 0) + 1

            transition_id += 1

    transition_df = pd.DataFrame(transition_rows)

    if not return_summary:
        return transition_df

    #################################
    ### Summary
    #################################

    transition_frequencies = {}

    for state in allowed_states:

        total = sum(transition_counts.get((state, s), 0)
            for s in allowed_states)

        if total == 0:
            continue

        transition_frequencies[state] = {
            s: transition_counts.get((state, s), 0) / total
            for s in allowed_states
            }

    summary = {
        "n_subjects": n_subjects,
        "n_skipped_subjects": n_skipped_subjects,
        "n_transitions": len(transition_df),
        "transition_counts": transition_counts,
        "transition_frequencies": transition_frequencies
            }

    return transition_df, summary


###############################################################

