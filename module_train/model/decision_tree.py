# MARK:- libs
import numpy as np
import pandas as pd
import pydotplus

from IPython.display import Image
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from sklearn import tree

from data_path import NORMALIZED_DATASET_PATH, IMAGE_DES_TREE_DATA_PATH

# MARK:- prepare data
data_df = pd.read_csv(NORMALIZED_DATASET_PATH)
labels_arr = np.array(data_df['decision'])
features_df = data_df.drop('decision', axis=1)
features_arr = np.array(features_df)

feature_list = list(features_df.columns)

X_train, X_test, y_train, y_test = train_test_split(
    features_arr, labels_arr, test_size=0.2, random_state=42)

# MARK:- start training
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state=42)
clf_train = clf.fit(X_train, y_train)

# MARK:- display

dot_data = tree.export_graphviz(clf_train, out_file=None, feature_names=feature_list, class_names=['HoaGiai', 'XetXu'],
                                rounded=True, filled=True)

graph = pydotplus.graph_from_dot_data(dot_data)
img = Image(graph.create_png())

with open(IMAGE_DES_TREE_DATA_PATH, "wb") as png:
    png.write(img.data)
