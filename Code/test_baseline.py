### Simple test of the baseline Markov model using the 
# trasition dataset

# Ali Zare (zareali@msu.edu, ali.zr1983@gmail.com, ali.zare@duke.edu)


import numpy as np
import pandas as pd
from model_utils import markov_baseline as mb
from model_utils import markov_baseline_cv as mbcv

#####################################################

subj_with_change = False


base_name = "transition_dataset_new"
directory = "./Processed_Data/"

file_name = directory + base_name + ".csv"

transition_df = pd.read_csv(file_name, low_memory = False)
print("===========================================")
print(f"\nTransition data imported.\n")

#####################################################

if subj_with_change:
    has_change = (transition_df["current_state"]
        .ne(transition_df["next_state"])
        .groupby(transition_df["subject_id"])
        .transform("any"))
    transition_df = (transition_df.loc[has_change].copy()
        .reset_index(drop = True))
    print("===========================================")
    print(f"\nOnly subjects with at least one change of state were kept.\n")
    print(f"As a result, only {len(transition_df['subject_id'].unique())} subjects with {len(transition_df)} transitions remain.\n")

#####################################################
### Cross-Validation
#####################################################
print("========================================")
print("Running cross-validation...")
print("========================================")
cv_results = mbcv.run_cross_validation_markov(
    transition_df = transition_df,
    k = 5,
    subject_col = "subject_id",
    current_state_col = "current_state_id",
    next_state_col = "next_state_id",
    states = (0, 1, 2),
    state_labels = ("CN", "MCI", "AD"),
    smoothing = 1.0,
    stratify = True,
    random_state = 123)


print(cv_results["aggregate_metrics"])

# print(cv_results["fold_results"][0]["probability_matrix"])

prob_mat = []
for k in range(5):
    prob_mat.append(cv_results["fold_results"][k]["probability_matrix"])
prob_mat = np.array(prob_mat)
prob_mat_avg = np.mean(prob_mat, axis = 0)
print(prob_mat_avg)

#####################################################

STATE_IDS = (0, 1, 2)
STATE_LABELS = ("CN", "MCI", "AD")

probability_matrix, count_matrix = mb.estimate_transition_matrix(
    transition_df = transition_df,
    current_state_col = "current_state_id",
    next_state_col = "next_state_id",
    states = STATE_IDS,
    state_labels = STATE_LABELS,
    smoothing = 0.0)

print("===========================================")
print("Observed counts:")
print(count_matrix)

print("\nEstimated probabilities:")
print(probability_matrix.round(4))


wilson_ci = mb.compute_transition_confidence_intervals(
    data = np.array(count_matrix),
    method = "wilson",
    confidence_level = 0.95,
    states = (0, 1, 2),
    state_labels = ("CN", "MCI", "AD"))
print("===========================================")
print("Confidence intervals (Wilson):")
print(wilson_ci[['current_state', 'next_state', 'count', 'lower_bound', 'upper_bound']])

# The bootstrap result uses percentile intervals. Some rare transitions
# may have fewer valid bootstrap samples because no transitions originate
# from that state in certain resamples.
bootstrap_ci = mb.compute_transition_confidence_intervals(
    data = transition_df,
    method = "bootstrap",
    current_state_col = "current_state_id",
    next_state_col = "next_state_id",
    subject_col = "subject_id",
    states = (0, 1, 2),
    state_labels = ("CN", "MCI", "AD"),
    n_bootstrap = 100,
    random_state = 123)
print("===========================================")
print("Confidence intervals (Bootstrap):")
print(bootstrap_ci[['current_state', 'next_state', 'count', 'lower_bound', 'upper_bound']])


# cn_to_mci_ci = confidence_intervals.query(
#     "current_state == 'CN' and next_state == 'MCI'")
# print(cn_to_mci_ci)


mb.plot_transition_matrix(count_matrix, matrix_type = "count")

mb.plot_transition_matrix(probability_matrix,
                          matrix_type = "probability",
                          decimals = 3)

mb.plot_transition_matrix_with_ci(wilson_ci,
    state_labels = ("CN", "MCI", "AD"),
    decimals = 4)
mb.plot_transition_matrix_with_ci(bootstrap_ci,
    state_labels = ("CN", "MCI", "AD"),
    decimals = 4)

### Generate baseline predictions for every row of the transition dataset
markov_probabilities, markov_predictions = mb.predict_markov(
    current_states = transition_df["current_state"],
    transition_matrix = probability_matrix,
    return_classes = True)

baseline_results = pd.concat([transition_df[
    ["transition_id", "subject_id", "current_state", "next_state"]
     ].reset_index(drop = True),
     markov_probabilities.reset_index(drop = True)], axis = 1)
baseline_results["predicted_next_state"] = markov_predictions
print(baseline_results.head())




print("\nEnd of Code!\n")