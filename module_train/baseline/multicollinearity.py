# MARK:- libs
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.api import OLS
from sklearn.model_selection import train_test_split
from data_path import NORMALIZED_DATASET_PATH

# MARK:- prepare data
data_df = pd.read_csv(NORMALIZED_DATASET_PATH)
labels_arr = np.array(data_df['decision'])
features_df = data_df.drop('decision', axis=1)

# For each X, calculate VIF and save in dataframe
X = features_df
X['interpret'] = 1

# Compute and view VIF
vif = pd.DataFrame()
vif["variables"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# View results using print
print(vif)
