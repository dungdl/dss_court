# MARK:- libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from data_path import NORMALIZED_DATASET_PATH

# MARK:- prepare data
data_df = pd.read_csv(NORMALIZED_DATASET_PATH)
labels_arr = np.array(data_df['decision'])
features_df = data_df.drop('decision', axis=1)
features_arr = np.array(features_df)

feature_list = list(features_df.columns)

X_train, X_test, y_train, y_test = train_test_split(
    features_arr, labels_arr, test_size=0.2, random_state=42)

# MARK:- create decision tree classifier

clf = DecisionTreeClassifier(criterion="entropy")

clf = clf.fit(X_train, y_train)

# MARK:- start prediction

y_pred = clf.predict(X_test)


# MARK:- evaluation
accuracy = accuracy_score(y_test, y_pred, normalize=False)

print('Accuracy:', round(accuracy, 2))
