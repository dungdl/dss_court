# MARK:- libs
import numpy as np
import pandas as pd

from statsmodels.api import OLS
from sklearn.model_selection import train_test_split
from data_path import NORMALIZED_DATASET_PATH

# MARK:- prepare data
data_df = pd.read_csv(NORMALIZED_DATASET_PATH)
labels_arr = np.array(data_df['decision'])
features_df = data_df.drop('decision', axis=1)
features_arr = np.array(features_df)

feature_list = list(features_df.columns)

X_train, X_test, y_train, y_test = train_test_split(
    features_arr, labels_arr, test_size=0.2, random_state=42)

model = OLS(y_train, X_train).fit()

print(model.summary())
