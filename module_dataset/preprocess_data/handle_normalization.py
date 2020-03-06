# MARK:- libs
import pandas as pd
import re
from pandas import ExcelWriter
from pandas import ExcelFile

from data_path import *

pd.options.mode.chained_assignment = None
# MARK:- get discrete data

source_df = pd.read_excel(RAW_DATASET_PATH)

ids = source_df['id']
legal_relas = source_df['legal_rela']
plaintiff_ages = source_df['plaintiff_age']
defendant_ages = source_df['defendant_age']
decisions = source_df['decision']

length = len(legal_relas) - 1

age_dist = []
has_childrens = []


# MARK:- data normalization

for i in range(0, length + 1):

    # TODO: convert categorical data
    relation = legal_relas[i]
    decision = decisions[i]

    corr_legal = re.search(r'\d', relation)
    corr_decis = re.search(r'\d', decision)

    legal_relas[i] = int(corr_legal.group())
    decisions[i] = int(corr_decis.group()) - 1

    # TODO: speculate childrend status from legal relation
    if legal_relas[i] == 5:
        has_childrens.append(0)
    else:
        has_childrens.append(1)

    # TODO: calculate age distances between plaintiff and defendants
    age_dist.append(abs(int(plaintiff_ages[i]) - int(defendant_ages[i])))

# MARK:- create useful dataframe

data_df = pd.DataFrame(
    [legal_relas, plaintiff_ages, defendant_ages, decisions])

data_df = data_df.transpose()

data_df['age_dist'] = pd.Series(age_dist, index=data_df.index)
data_df['has_children'] = pd.Series(has_childrens, index=data_df.index)

data_df.to_csv(NORMALIZED_DATASET_PATH, index=False)
