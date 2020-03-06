# MARK:- libs
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
import re

from data_path import *

# MARK:- Explodary data analysis

data_df = pd.read_csv(NORMALIZED_DATASET_PATH)
train_df = data_df.drop(columns=['decision'])
feature_l = train_df.columns
labels = data_df['decision'].tolist()


print(data_df['decision'].value_counts())
# TODO _stat about labels


def label_stat():
    data_labels = data_df['decision']
    plot_labels = {"1", "2"}
    label_count = data_labels.value_counts()

    plt.pie(label_count, labels=plot_labels, autopct='%1.1f%%')
    plt.axis('equal')
    plt.show()
