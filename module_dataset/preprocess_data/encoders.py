# MARK:- libs
import numpy as np
import pandas as pd
import category_encoders as ce
import re

from data_path import *

# MARK:- Preparation

data_df = pd.read_csv(NORMALIZED_DATASET_PATH)

# MARK:- WoE encoder testing


def woe_encoding():
    source_woe_df = data_df
    woe = source_woe_df.groupby('legal_rela')['decision'].mean()

    woe_df = pd.DataFrame(woe)

    woe_df = woe_df.rename(columns={'decision': 'Good'})
    woe_df['Bad'] = 1 - woe_df.Good
    # avoid divide by zero in denominator
    woe_df['Bad'] = np.where(woe_df['Bad'] == 0, 0.0000001, woe_df['Bad'])
    woe_df['Good'] = np.where(woe_df['Good'] == 0, 0.0000001, woe_df['Good'])
    # compute WoE
    woe_df['WoE'] = np.log(woe_df.Good / woe_df.Bad)
    print(woe_df)

# MARK:- Binary encoder testing


def binary_encoding():
    source_bin_df = data_df
    encoder = ce.BinaryEncoder(cols=['legal_rela'], drop_invariant=True)
    dfb = encoder.fit_transform(source_bin_df['legal_rela'])
    source_bin_df = pd.concat([source_bin_df, dfb], axis=1)

    print(source_bin_df)


# MARK:- one-hot encoder testing

def one_hot_encoding():
    one_hot_df = pd.get_dummies(
        data_df, prefix=['rela'], columns=['legal_rela'])
    print(one_hot_df)


data_df = pd.read_csv(NORMALIZED_DATASET_PATH)

map_name_dic = {'legal_rela': 'quan_he_phap_luat',
                'plaintiff_age': 'tuoi_nguyen_don',
                'defendant_age': 'tuoi_bi_don',
                'age_dist': 'do_lech_tuoi',
                'has_children': 'co_con_chung',
                'decision': 'quyet_dinh'}

datad_df = data_df.rename(columns=map_name_dic, inplace=True)

labels = data_df['quyet_dinh'].tolist()
train_df = data_df.drop(columns=['quyet_dinh'])
feature_l = train_df.columns
print(data_df.head())
