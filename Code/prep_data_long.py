# This code performs preprocessing (cleaning, filtering, variable
# selection, ...) on NACC UDS longitudinal data.  

# Ali Zare (zareali@msu.edu, ali.zr1983@gmail.com, ali.zare@duke.edu)


import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import gc
from pathlib import Path

from general_utils import helper_functions as hf

############################################################

base_name = "investigator_nacc70" # NACC UDS data
directory = "../Data/"

file_name = directory + base_name + ".csv"


# set to 1 to look for common subjects with phenotype harmonization data
pheno_flag = 0  

############################################################

df = pd.read_csv(file_name, low_memory = False)
print("===========================================")
print(f"\nCSV file imported.\n")

#
missvalcols = df.columns[df.isna().any()].tolist()
# missvalcolnum = len((df.isna().sum().values)[df.isna().sum().values != 0])
missing_counts = df.isna().sum()
missing_count_percol = missing_counts[missing_counts > 0]
missing_dict = dict(zip(missvalcols, missing_count_percol))

id_col = "NACCID"
uds_subjects = set(df[id_col].dropna().unique())
print(f"\nNumber of UDS subjects BEFORE filtering: {len(uds_subjects):}\n")


print("===========================================")
### Keep only CN, MCI, and AD visits defined based on criteria
# and create a DIAGNOSIS column.
df = hf.filter_and_label_diagnosis(df)

uds_subjects = set(df[id_col].dropna().unique())

print(f"\nNumber of UDS subjects AFTER filtering for diagnosis labels: {len(uds_subjects):}\n")
print("Visit label counts AFTER filtering:")
print(df["DIAGNOSIS"].value_counts())

############################################################
### Find common subjects among UDS and phenotype harmonization data

if pheno_flag == 1:
    # Phenotype harmonization files
    pheno_files = ['../Data/NACC_ADSP_PHC_FLAIR_2024.csv',
                '../Data/NACC_ADSP_PHC_DTI_2024.csv',
                '../Data/NACC_ADSP_PHC_Neuropath_2024.csv',
                '../Data/NACC_ADSP_PHC_Tau_Detailed_2024.csv',
                '../Data/NACC_ADSP_PHC_VascularRisk_2024.csv',
                '../Data/NACC_ADSP_PHC_Tau_Simple_2024.csv',
                '../Data/NACC_ADSP_PHC_Amyloid_Detailed_2024.csv',
                '../Data/NACC_ADSP_PHC_T1_Freesurfer_2024.csv',
                '../Data/NACC_ADSP_PHC_Amyloid_Simple_2024.csv',
                '../Data/NACC_ADSP_PHC_Biomarker_2024.csv',
                '../Data/NACC_ADSP_PHC_Cognition_2024.csv',
                '../Data/NACC_ADSP_PHC_T1_MUSE_2024.xlsx']

    # Compare UDS data with each phenotype file
    common_all = uds_subjects.copy()
    for file in pheno_files:

        if file.lower().endswith((".csv")):
            df_phen = pd.read_csv(file)
        elif file.lower().endswith((".xlsx")):
            df_phen = pd.read_excel(file)


        phenotype_subjects = set(df_phen[id_col].dropna().unique())

        common = uds_subjects & phenotype_subjects
        common_all &= phenotype_subjects

        print(f"{Path(file).name}")
        print(f"  Subjects in file:      {len(phenotype_subjects):}")
        print(f"  Common subjects:       {len(common):}")
        print()

        ### Optional: save IDs
        # pd.DataFrame(sorted(common),
        #              columns = [id_col]).to_csv(f"common_{Path(file).stem}.csv",
        #                                         index = False)

    print(f"Subjects in ALL datasets: {len(common_all)}")

############################################################
### number of visits per participant

# visit_counts = np.array([])
# for subj, sub_df in df.groupby('NACCID'):
#     visit_counts = np.append(visit_counts, int(sub_df['NACCAVST'].values[0]))

visit_counts = np.array([])
for subj, sub_df in tqdm(df.groupby('NACCID'), desc = 'Subjects'):
    visit_counts = np.append(visit_counts, int(sub_df['NACCAVST'].values[0]))

print(f'\nVisit statistics for {len(visit_counts)} subjects:')
print(f'\nMean no. visits: {visit_counts.mean():.3f}')
print(f'\nSTD: {visit_counts.std():.3f}')
print(f'\nMedian no. visits: {np.median(visit_counts):.3f}')
print(f'\nMin. no. visits: {visit_counts.min()}')
print(f'\nMax. no. visits: {visit_counts.max()}')
print("")

############################################################
### keep only subjects with a certain of number of visits

n_visit = 3
# df_sel = df.loc[(df['NACCAVST'] >= 3) & (df['NACCAVST'] <= 7),:].copy()
# df_sel = df[(df['NACCAVST'] >= 3) & (df['NACCAVST'] <= 7)].copy()
# df_sel = df[df['NACCAVST'].between(3, 7)].copy()
# df_sel = df.loc[df['NACCAVST'] >= n_visit,:].copy()
# df_sel = df[df['NACCAVST'] >= n_visit].copy()

### keep subjects with at least 3 visits
visit_cnt = df.groupby("NACCID").size()
valid_ids = visit_cnt[visit_cnt >= n_visit].index
df_sel = df[df["NACCID"].isin(valid_ids)].copy()


# Free some memory
del df
gc.collect()


# check whether the number of visits satisfies the minimum
vis_num = np.array([])
subj = []
for k, sub_df in df_sel.groupby("NACCID"):
    vis_num = np.append(vis_num, len(sub_df))
    subj.append(k)
# subj_filtered = list(np.array(subj)[vis_num < n_visit])
subj_filtered = [s for s, v in zip(subj, vis_num) if v < n_visit]


print("===========================================")
print("After filtering data for min. no. visits:")
print(f"\nFiltered dataset to keep subjects with at least {n_visit} visits:\n")
print(f"There are a total of {len(df_sel)} samples from {len(df_sel['NACCID'].unique())} subjects.\n")

print(f"\nCounts per diagnosis (NACCDUSD):\n")
print(df_sel["NACCUDSD"].value_counts())
print(f"\nCounts per diagnosis (assigned labels):\n")
print(df_sel["DIAGNOSIS"].value_counts())
############################################################
############################################################
### Choosing variables and creating new variables

FAQ_vars = ['BILLS', 'TAXES', 'SHOPPING',
            'GAMES', 'STOVE', 'MEALPREP',
            'EVENTS', 'PAYATTN', 'REMDATES', 'TRAVEL']
# Create a new column containing the sum of all FAQs
df_sel['FAQTOTAL'] = df_sel[FAQ_vars].apply(
    lambda row: pd.to_numeric(row, errors = "coerce").sum(), axis = 1
    )
df_sel.drop(FAQ_vars, axis = 1)

############################################################
### Find the distribution of conversions
converted_subj = []
for k, subdf in enumerate(df_sel.groupby("NACCID")):
    if len(subdf[1]["NACCUDSD"].unique()) > 1:
        converted_subj.append(subdf[0])
        
print(f"\nThere are {len(converted_subj)} subjects with conversion.\n")

############################################################
### Sort visits from oldest to most recent for each subject
# and create labels for visits

df_sel["visit_date"] = pd.to_datetime(
    dict(year = df_sel["VISITYR"],
         month = df_sel["VISITMO"],
         day = df_sel["VISITDAY"]))
df_sel = df_sel.sort_values(["NACCID", "visit_date"])
df_sel["visit"] = df_sel.groupby("NACCID").cumcount().add(1).astype(str).radd("visit")

############################################################
### Get information about the spread of visits
# Get first visit date per subject
first_visit = df_sel.groupby("NACCID")["visit_date"].transform("min")

# Calendar month difference
df_sel["months_from_baseline"] = (
    df_sel["visit_date"].dt.to_period("M").astype(int) -
    first_visit.dt.to_period("M").astype(int))
# Force strict monotonic increase
df_sel["months_from_baseline"] = (
    df_sel.groupby("NACCID")["months_from_baseline"]
    .cummax())

diffs = (df_sel["months_from_baseline"].diff().values).copy()
diffs[np.isnan(diffs)] = 0
diffs[diffs < 0] = 0
diffs_nz = diffs[diffs > 0].copy()

fig, ax = plt.subplots(figsize = (8, 6))
freqs, _, _ = ax.hist(diffs_nz, bins = 50, color = 'skyblue', edgecolor='black')
x_value = diffs_nz.mean()
plt.axvline(x = diffs_nz.mean(),
            color = 'red',
            linestyle = '--',
            linewidth = 1.5)
plt.annotate(f'Mean = {diffs_nz.mean():.1f}',
    xy = (diffs_nz.mean(), 100), # point to annotate
    xytext = (diffs_nz.mean() + 5, freqs.max()*0.95), # text position
    # arrowprops = dict(facecolor = 'red', shrink = 0.05),
    color = 'red',
    fontsize = 16)
plt.axvline(x = np.median(diffs_nz),
            color = 'blue',
            linestyle = '--',
            linewidth = 1.5)
plt.annotate(f'Median = {np.median(diffs_nz):.1f}',
    xy = (np.median(diffs_nz), 100), # point to annotate
    xytext = (np.median(diffs_nz) + 5, freqs.max()*0.9), # text position
    # arrowprops = dict(facecolor = 'red', shrink = 0.05),
    color = 'blue',
    fontsize = 16)
ax.tick_params(axis = 'both', labelsize = 16)
ax.set_title('Consecutive Visit Time Difference Distribution', fontsize = 16)
ax.set_xlabel('Time between Visits (months)', fontsize = 16)
ax.set_ylabel('Frequency', fontsize = 16)
ax.grid(axis = 'y', alpha = 0.75)
plt.show(block = False)

############################################################

demog_vars = ['NACCID', 'NACCAPOE', 'NACCNE4S',
              'VISITMO', 'VISITDAY', 'VISITYR',
              'visit', 'months_from_baseline',
              'NACCAGE', 'NACCAGEB', 'SEX', 'EDUC']

cols_reduced = ['CDRSUM', 'CDRGLOB',
                'NACCMMSE', 'NACCMOCA',
                'FAQTOTAL', 'REYDREC',
                'DIAGNOSIS']
df_sel = df_sel[demog_vars + cols_reduced]

###################################################################
### Remove missing data flagged with -4, 9, <0, or other
# numbers used to identify inadmissible data
print('Removing inadmissible values...')

for col in df_sel.columns:
    df_sel.loc[df_sel[col] == -4, col] = None

df_sel.loc[df_sel['NACCAPOE'] == 9, 'NACCAPOE'] = None
df_sel.loc[df_sel['NACCNE4S'] == 9, 'NACCNE4S'] = None

df_sel.loc[df_sel['FAQTOTAL'] < 0, 'FAQTOTAL'] = None

df_sel.loc[df_sel['NACCMOCA'] == 99, 'NACCMOCA'] = None
df_sel.loc[df_sel['NACCMOCA'] == 88, 'NACCMOCA'] = None

df_sel.loc[df_sel['NACCMMSE'] == 88, 'NACCMMSE'] = None
df_sel.loc[df_sel['NACCMMSE'].between(95, 98), 'NACCMMSE'] = None

print('Done!\n')

###################################################################
### Use MOCA to fill in missing values in MMSE

map_to_mmse = {2:7, 3:9, 4:10, 5:12, 6:13, 7:14, 8:15, 9:16,
               10:17, 11:18, 12:19, 13:20, 14:21, 15:22, 16:23,
               17:24, 18:25, 19:25, 20:26, 21:27, 22:27, 23:28,
               24:28, 25:29, 26:29, 27:30, 28:30, 29:30, 30:30}

mask = df_sel['NACCMMSE'].isna() & df_sel['NACCMOCA'].isin(map_to_mmse)
df_sel.loc[mask, 'NACCMMSE'] = df_sel.loc[mask, 'NACCMOCA'].map(map_to_mmse)

###################################################################
# Create transition dataset
# feature_cols = cols_reduced[:-1]
feature_cols = ['NACCAGE', 'SEX', 'NACCNE4S',
                'CDRSUM', 'CDRGLOB', 'NACCMMSE',
                'FAQTOTAL', 'NACCNE4S']

result = hf.build_transition_dataset(
    df_sel,
    feature_cols,
    "NACCID",
    "DIAGNOSIS",
    "months_from_baseline",
    allowed_states = ("CN", "MCI", "AD"),
    allow_regression = True,
    regression_action = "keep",
    drop_missing = False,
    return_summary = True)

transition_dataset = result[0]
if len(result) > 1:
    summary = result[1]

print("\nTransitoin dataset created!\n")

transition_dataset.to_csv('./Processed_Data/transition_dataset_new.csv', index = False)

###################################################################
###################################################################
### Make some plots

from visual_utils import transition_plots as tp
from visual_utils import feature_plots as fp

print("\nCreating plots...\n")

# Cohort overview
tp.plot_state_distribution(transition_dataset)
tp.plot_subject_transition_counts(transition_dataset)
tp.plot_regression_statistics(transition_dataset)

# Transition analysis
tp.plot_transition_counts(transition_dataset)
tp.plot_transition_matrix(transition_dataset)
tp.plot_transition_network(transition_dataset)
# tp.plot_sankey(transition_dataset)

# Time analysis
tp.plot_delta_t_histogram(transition_dataset)
tp.plot_delta_t_by_transition(transition_dataset)


# Feature analysis
feat = "NACCMMSE" # or any other feature

fp.plot_feature_distribution(transition_dataset, feature = feat,
                             density = True)
fp.plot_feature_by_transition(transition_dataset, feature = feat)
fp.plot_feature_by_transition(transition_dataset, feature = feat,
                              kind = "violin")
fp.plot_feature_correlations(transition_dataset,
                             ['CDRSUM', 'NACCMMSE', 'FAQTOTAL'],
                             method = "spearman")
fp.plot_missing_values(transition_dataset)
fp.plot_scatter_matrix(transition_dataset,
                       ['CDRSUM', 'NACCMMSE', 'FAQTOTAL'])
fp.plot_pca(transition_dataset,
            ['CDRSUM', 'NACCMMSE', 'FAQTOTAL'],
            color_by = "next_state")

###################################################################

print("\nEnd of Code!\n")