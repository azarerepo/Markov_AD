### This modeule contains the cross-validation and evaluation functions
# for the baseline Markov model

# Ali Zare (zareali@msu.edu, ali.zr1983@gmail.com, ali.zare@duke.edu)


import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score
    )

import time

from sklearn.model_selection import (
    GroupKFold,
    StratifiedGroupKFold
    )

from model_utils import markov_baseline as mb


# ============================================================
# Input validation
# ============================================================

def _validate_probability_inputs(y_true, y_prob, states):

    """
    Validate labels and predicted probability matrix.
    """

    y_true = np.asarray(y_true)

    y_prob = np.asarray(y_prob, dtype = float)

    states = list(states)

    if y_true.ndim != 1:
        raise ValueError("y_true must be one-dimensional.")

    if y_prob.ndim != 2:
        raise ValueError("y_prob must be a two-dimensional array.")

    if len(y_true) != y_prob.shape[0]:
        raise ValueError("y_true and y_prob must contain "
            "the same number of samples.")

    if y_prob.shape[1] != len(states):
        raise ValueError("The number of probability columns "
            "must match the number of states.")

    if len(states) == 0:
        raise ValueError("states must not be empty.")

    if len(set(states)) != len(states):
        raise ValueError("states must not contain duplicates.")

    if np.isnan(y_prob).any():
        raise ValueError("y_prob contains NaN values.")

    if np.isinf(y_prob).any():
        raise ValueError("y_prob contains infinite values.")

    if (y_prob < 0).any():
        raise ValueError("Probabilities cannot be negative.")

    if (y_prob > 1).any():
        raise ValueError("Probabilities cannot exceed 1.")

    row_sums = y_prob.sum(axis = 1)

    if not np.allclose(row_sums, 1.0, atol = 1e-6):
        raise ValueError("Each row in y_prob must sum to 1.")

    observed_states = set(np.unique(y_true))

    unknown_states = observed_states - set(states)

    if unknown_states:
        raise ValueError("y_true contains states not included "
            f"in states: {unknown_states}")

    return y_true, y_prob, states


# ============================================================
# One-hot encoding
# ============================================================

def one_hot_encode_states(y, states = (0, 1, 2)):

    """
    Convert integer/class labels to one-hot encoding.

    Example:
        0 : [1, 0, 0]
        1 : [0, 1, 0]
        2 : [0, 0, 1]
    """

    y = np.asarray(y)

    states = list(states)

    state_to_index = {
        state: index for index, state in enumerate(states)}

    unknown_states = set(np.unique(y)) - set(states)

    if unknown_states:
        raise ValueError(f"Unknown states: {unknown_states}")

    one_hot = np.zeros((len(y), len(states)), dtype = float)

    indices = np.asarray(
        [state_to_index[state] for state in y], dtype = int)

    one_hot[np.arange(len(y)), indices] = 1.0

    return one_hot


# ============================================================
# Calibration data
# ============================================================

def compute_calibration_curves(y_true, y_prob,
    states = (0, 1, 2),
    state_labels = ("CN", "MCI", "AD"),
    n_bins = 10,
    strategy = "uniform"
    ):

    """
    Compute one-vs-rest calibration curves for each state.

    Returns:
    dict
        {
        "CN": {
            "mean_predicted_probability": ...,
            "fraction_positive": ...
            },
        ...
        }
    """

    y_true, y_prob, states = _validate_probability_inputs(y_true, y_prob, states)

    y_one_hot = one_hot_encode_states(y_true, states = states)

    if len(state_labels) != len(states):
        raise ValueError("state_labels must match states.")

    calibration_results = {}

    for k, label in enumerate(state_labels):
        fraction_positive, mean_predicted = calibration_curve(
            y_one_hot[:, k],
            y_prob[:, k],
            n_bins = n_bins,
            strategy = strategy)

        calibration_results[label] = {
            "mean_predicted_probability": mean_predicted,
            "fraction_positive": fraction_positive}

    return calibration_results


# ============================================================
# Expected Calibration Error
# ============================================================

def expected_calibration_error(y_true, y_prob,
    states = (0, 1, 2), n_bins = 10):

    """
    Compute multiclass Expected Calibration Error (ECE)
    based on the maximum predicted probability.

    Lower is better.
    """

    y_true, y_prob, states = _validate_probability_inputs(
        y_true, y_prob, states)

    predicted_positions = np.argmax(y_prob, axis = 1)

    predicted_states = np.asarray(
        [states[position] for position in predicted_positions])

    confidence = np.max(y_prob, axis = 1)

    correct = (predicted_states == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for bin_index in range(n_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        if bin_index == 0:
            in_bin = ((confidence >= lower) & (confidence <= upper))

        else:
            in_bin = ((confidence > lower) & (confidence <= upper))

        n_in_bin = np.sum(in_bin)
        if n_in_bin == 0:
            continue

        accuracy_bin = np.mean(correct[in_bin])
        confidence_bin = np.mean(confidence[in_bin])
        ece += (n_in_bin / len(y_true)) * abs(accuracy_bin - confidence_bin)

    return float(ece)



def evaluate_markov(y_true, predicted_probabilities,
    states = (0, 1, 2),
    state_labels = ("CN", "MCI", "AD")
    ):

    """
    Evaluate baseline Markov predictions.

    Inputs:
    y_true
        Observed next-state labels.

    predicted_probabilities
        Array of shape (n_samples, n_states).

    states
        Ordered state values matching probability columns.

    state_labels
        Human-readable state names.

    Returns:
    dict
        Probability and classification metrics.

    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(predicted_probabilities, dtype = float)

    states = list(states)
    state_labels = list(state_labels)

    if probabilities.ndim != 2:
        raise ValueError("predicted_probabilities must be two-dimensional.")

    if probabilities.shape[0] != len(y_true):
        raise ValueError("The numbers of labels and predictions do not match.")

    if probabilities.shape[1] != len(states):
        raise ValueError("Probability columns must match the number of states.")

    if np.isnan(probabilities).any():
        raise ValueError(
            "Predicted probabilities contain missing values. "
            "Use positive smoothing when estimating the matrix.")

    if not np.allclose(probabilities.sum(axis=1), 1.0):
        raise ValueError("Each predicted probability row must sum to 1.")

    state_to_position = {state: index for index, state in enumerate(states)}

    unknown = set(np.unique(y_true)) - set(states)

    if unknown:
        raise ValueError(f"y_true contains unknown states: {unknown}")

    predicted_positions = np.argmax(probabilities, axis = 1)

    y_pred = np.asarray(
        [states[position] for position in predicted_positions])

    # One-hot ground-truth probabilities.
    y_true_onehot = np.zeros_like(probabilities, dtype = float)

    true_positions = np.asarray(
        [state_to_position[state] for state in y_true])

    y_true_onehot[np.arange(len(y_true)), true_positions] = 1.0

    # Multiclass Brier score:
    # mean sum of squared probability errors per observation.
    brier_score = np.mean(
        np.sum((probabilities - y_true_onehot) ** 2, axis = 1) )

    metrics = {
        "n_transitions": len(y_true),

        "log_loss": log_loss(y_true, probabilities, labels = states),

        "brier_score": brier_score,

        "ece" : expected_calibration_error(
                    y_true = y_true,
                    y_prob = probabilities,
                    states = states,
                    n_bins = 10),

        "accuracy": accuracy_score(y_true, y_pred),

        "precision_macro": precision_score(y_true, y_pred,
            labels = states,
            average = "macro",
            zero_division = 0),
        "precision_weighted" : precision_score(y_true, y_pred,
            labels = states,
            average = "weighted",
            zero_division = 0),
        "recall_macro": recall_score(y_true, y_pred,
            labels = states,
            average = "macro",
            zero_division = 0),
        "recall_weighted" : recall_score(y_true, y_pred,
            labels = states,
            average = "weighted",
            zero_division = 0),
        "f1_macro": f1_score(y_true, y_pred,
            labels = states,
            average = "macro",
            zero_division = 0),
        "f1_weighted" : f1_score(y_true, y_pred,
            labels = states,
            average = "weighted",
            zero_division = 0),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels = states),
        "classification_report": classification_report(y_true, y_pred,
            labels = states,
            target_names = state_labels,
            zero_division = 0,
            digits = 4),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_probability": probabilities,
    }

    try:
        metrics["auc_macro_ovr"] = roc_auc_score(
            y_true_onehot,
            probabilities,
            average = "macro",
            multi_class = "ovr")
    except ValueError:
        # A fold may not contain all classes.
        metrics["auc_macro_ovr"] = np.nan
    
    try:
        metrics["auc_weighted_ovr"] = roc_auc_score(
            y_true_onehot,
            probabilities,
            average = "weighted",
            multi_class = "ovr")
    except ValueError:
        metrics["auc_weighted_ovr"] = np.nan

    return metrics



def run_cross_validation_markov(transition_df: pd.DataFrame,
    k: int = 5,
    subject_col: str = "subject_id",
    current_state_col: str = "current_state_id",
    next_state_col: str = "next_state_id",
    states = (0, 1, 2),
    state_labels = ("CN", "MCI", "AD"),
    smoothing: float = 1.0,
    stratify: bool = True,
    shuffle: bool = True,
    random_state: int = 42,
    verbose: bool = True
    ):

    """
    This function performs subject-level k-fold cross-validation for a baseline Markov model.

    The transition matrix is estimated separately from the training
    transitions in each fold.

    Inputs:
    transition_df:  One row per transition.

    k:  Number of folds.

    subject_col: Subject identifier.

    current_state_col, next_state_col
        Integer-encoded state columns.

    states, state_labels
        Ordered state values and readable labels.

    smoothing
        Additive smoothing for transition probabilities. A positive value
        is recommended in cross-validation so unseen transitions or states
        do not produce zero or undefined probabilities.

    stratify
        Use StratifiedGroupKFold based on next-state labels while preserving
        subject groups.

    shuffle
        Shuffle groups when supported.

    random_state
        Random seed.

    verbose
        Print fold summaries.

    Returns:
    result : dict
        Contains fold-level metrics, aggregate metrics, out-of-fold
        predictions, and transition matrices.

    """

    start_time = time.perf_counter()

    required = {subject_col, current_state_col, next_state_col}

    missing = required.difference(transition_df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if k < 2:
        raise ValueError("k must be at least 2.")

    n_subjects = transition_df[subject_col].nunique()

    if n_subjects < k:
        raise ValueError(f"k = {k} exceeds the number of subjects "
            f"({n_subjects}).")

    working_df = transition_df.reset_index(
        drop = False).rename(columns = {"index": "original_index"})

    X_dummy = np.zeros((len(working_df), 1))

    y = working_df[next_state_col].to_numpy()
    groups = working_df[subject_col].to_numpy()

    if stratify:
        splitter = StratifiedGroupKFold(n_splits = k, shuffle = shuffle,
            random_state = (random_state if shuffle else None))

        folds = splitter.split(X_dummy, y, groups)
    else:
        splitter = GroupKFold(n_splits = k)

        folds = splitter.split(X_dummy, y, groups)

    fold_results = []
    oof_rows = []

    for fold_number, (train_indices, validation_indices) in enumerate(folds, start = 1):

        train_df = working_df.iloc[train_indices].copy()

        validation_df = working_df.iloc[validation_indices].copy()

        train_subjects = set(train_df[subject_col])
        validation_subjects = set(validation_df[subject_col])

        if train_subjects & validation_subjects:
            raise RuntimeError(
                "Subject leakage detected between training "
                "and validation sets.")

        # Estimate the matrix using training transitions only.
        probability_matrix, count_matrix = \
            mb.estimate_transition_matrix(
                transition_df = train_df,
                current_state_col = current_state_col,
                next_state_col = next_state_col,
                states = states,
                state_labels = state_labels,
                smoothing = smoothing)
                

        # predict_markov() must receive the same representation as the
        # transition-matrix index. Since the matrix above is labeled
        # CN/MCI/AD, convert current IDs to labels for prediction.
        state_to_label = {
            state: label for state, label in zip(states, state_labels)}

        train_current_labels = train_df[current_state_col].map(state_to_label)

        validation_current_labels = validation_df[current_state_col].map(state_to_label)

        ##########################################################
        ### For each sample (transition), transition probabilities
        # to either state are calculated based on their current state.
        train_probability_df = mb.predict_markov(
            current_states = train_current_labels,
            transition_matrix = probability_matrix,
            return_classes = False)
        validation_probability_df = mb.predict_markov(
            current_states = validation_current_labels,
            transition_matrix = probability_matrix,
            return_classes = False)

        train_probabilities = train_probability_df.to_numpy()
        validation_probabilities = validation_probability_df.to_numpy()
    

        train_metrics = evaluate_markov(
            y_true = train_df[next_state_col],
            predicted_probabilities = train_probabilities,
            states = states,
            state_labels = state_labels)

        validation_metrics = evaluate_markov(
            y_true = validation_df[next_state_col],
            predicted_probabilities = validation_probabilities,
            states = states,
            state_labels = state_labels)

        fold_results.append(
            {
            "fold": fold_number,
            "train_subjects": sorted(train_subjects, key = str),
            "validation_subjects": sorted(validation_subjects, key = str),
            "n_train_subjects": len(train_subjects),
            "n_validation_subjects": len(validation_subjects),
            "n_train_transitions": len(train_df),
            "n_validation_transitions": len(validation_df),
            "count_matrix": count_matrix,
            "probability_matrix": probability_matrix,
            "train": train_metrics,
            "validation": validation_metrics
            })

        
        ####################################################
        ### Out-of-Fold measurements from validation results
        fold_oof = validation_df[
            ["original_index", subject_col,
             current_state_col, next_state_col]].copy()

        for probability_index, label in enumerate(state_labels):
            fold_oof[f"P_next_{label}"] = validation_probabilities[:, probability_index]

        fold_oof["predicted_next_state_id"] = validation_metrics["y_pred"]
        fold_oof["fold"] = fold_number

        oof_rows.append(fold_oof)


        if verbose:
            print(f"\n===== Fold {fold_number}/{k} =====")
            print(f"Train subjects: {len(train_subjects)}, "
                f"transitions: {len(train_df)}")
            print("Validation subjects: "
                f"{len(validation_subjects)}, "
                f"transitions: {len(validation_df)}")
            print("\nTransition matrix:")
            print(probability_matrix.round(4))

            print(
                "\nValidation: "
                f"log-loss = "
                f"{validation_metrics['log_loss']:.4f}, "
                f"Brier = "
                f"{validation_metrics['brier_score']:.4f}, "
                f"ECE = "
                f"{validation_metrics['ece']:.4f}, "
                f"accuracy = "
                f"{validation_metrics['accuracy']:.4f}, "
                f"precision = "
                f"{validation_metrics['precision_macro']:.4f}, "
                f"precision weighted = "
                f"{validation_metrics['precision_weighted']:.4f}, "
                f"recall = "
                f"{validation_metrics['recall_macro']:.4f}, "
                f"recall weighted = "
                f"{validation_metrics['recall_weighted']:.4f}, "
                f"F1 = "
                f"{validation_metrics['f1_macro']:.4f}"
                f"F1 weighted = "
                f"{validation_metrics['f1_weighted']:.4f}"
                )

    #######################################################
    # Aggregate fold-level results
    #######################################################
    scalar_metrics = [
        "log_loss",
        "brier_score",
        "ece",
        "accuracy",
        "precision_macro",
        "precision_weighted",
        "recall_macro",
        "recall_weighted",
        "f1_macro",
        "f1_weighted",
        "auc_macro_ovr",
        "auc_weighted_ovr"
        ]

    aggregate_rows = []

    for split_name in ["train", "validation"]:
        for metric_name in scalar_metrics:
            values = np.asarray(
                [fold_result[split_name][metric_name]
                    for fold_result in fold_results],
                    dtype = float)

            aggregate_rows.append(
                {
                "split": split_name,
                "metric": metric_name,
                "mean": np.nanmean(values),
                "std": np.nanstd(values, ddof = 1),
                "valid_folds": int(np.isfinite(values).sum())
                }
            )

    aggregate_metrics = pd.DataFrame(aggregate_rows)

    ### Sum fold confusion matrices rather than averaging them.
    # The fold confusion matrices are summed up to provide the
    # total number of predictions of each type across all
    # validation samples, equivalent to an overall out-of-fold confusion matrix.
    validation_confusion_matrix = np.sum(
        [fold_result["validation"]["confusion_matrix"]
            for fold_result in fold_results], axis = 0)

    oof_predictions = (pd.concat(oof_rows, ignore_index = True)
        .sort_values("original_index")
        .reset_index(drop = True))

    elapsed_time = (time.perf_counter() - start_time)

    if verbose:
        print("\n" + "=" * 50)
        print("Cross-validation summary")
        print("=" * 50)

        validation_summary = aggregate_metrics.query("split == 'validation'")

        for row in validation_summary.itertuples():
            print(f"{row.metric:20s}: "
                  f"{row.mean:.4f} +/- "
                  f"{row.std:.4f}")

        print("\nSummed validation confusion matrix:")
        print(validation_confusion_matrix)

        print(f"\nElapsed time: {elapsed_time:.2f} seconds")

    return {
        "fold_results": fold_results,
        "aggregate_metrics": aggregate_metrics,
        "oof_predictions": oof_predictions,
        "validation_confusion_matrix": validation_confusion_matrix,
        "elapsed_time_seconds": elapsed_time
        }