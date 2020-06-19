# MARK:- libs
import pandas as pd
import numpy as np

from data_path import NORMALIZED_DATASET_PATH

# MARK:- prepare data
data_df = pd.read_csv(NORMALIZED_DATASET_PATH)
features_l = data_df.columns

for f in features_l:
    print(data_df[f].value_counts())
    print("--------------------------------")
