import os
import pandas as pd
import numpy as np
from sklearn.metrics import ndcg_score
from sklearn.metrics import top_k_accuracy_score

# Pre-process Data
# ----------------------------
# read file
annotation = "./Annotation_EN - lean.tsv"
raw_data = pd.read_table(annotation)

# get rid of NOS: 18
raw_data = raw_data[raw_data["FLAG_NOS"].isnull()]
raw_data = raw_data.drop(labels=[225], axis=0)

# get rid of issues: 5
# TODO: 227, dicussion
raw_data = raw_data.drop(labels=[3,138], axis=0)
'''
3    C0000768                  Anomaly anomaly congen  7.0    3.0   1.0       NaN                                      Weird wording (check)
126  C0009443     Acute nasopharyngitis (common cold)  8.0    1.0   1.0       NaN                            This feels like 2 terms (v)
138  C0011854                Type I diabetes mellitus  7.0    2.0   1.0       NaN                                           Repeated (-)
191  C0015934  Foetal growth retardation, unspecified  7.0    2.0   1.0       NaN                                              typo? (v)
227  C0019080                                     Hem  7.0    5.0   1.0       NaN  I feel like this is a context dependent abbrev... (check)
'''

# get rid of irrelevant columns & redo the COUNT
raw_data = raw_data[["CUI", "STR", "SCORE"]]
raw_data["CNT"] = raw_data.groupby('CUI')['STR'].transform(len)

# TODO: stats data
# print(len(raw_data))

raw_data = raw_data.dropna()
raw_data['SCORE'] = raw_data['SCORE'].astype(int)
score_data = raw_data.groupby('CUI')['SCORE'].apply(list).reset_index(name="SCR") 
gold = score_data["SCR"].tolist()

input_data = raw_data.groupby('CUI')['STR'].apply(list).reset_index(name="SYN")
cuis = input_data["CUI"].tolist()
syn_data = input_data["SYN"].tolist()




