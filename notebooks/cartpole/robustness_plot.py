# %%
import pickle
import pandas as pd
import numpy as np

# %%

six_bins_fp = "results/run_six_bins/robustness_experiment.pkl"
# eight_bins_fp = 'results/run_eight_bins/robustness_experiment.pkl'
with open(six_bins_fp, "rb") as f:
    six_bins_data = pickle.load(f)

# with open(eight_bins_fp, 'rb') as f:
#     eight_bins_data = pickle.load(f)

# now need to get the data into a sensible format
