import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
# from sklearn.metrics import ndcg_score
# from sklearn.metrics import top_k_accuracy_score

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



# Eval Metrics
# ----------------------------
def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum( (2**r - 1) / np.log2(np.arange(2, r.size + 2)))
    return 0

def dcg_normal_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum( r  / np.log2(np.arange(2, r.size + 2)))
    return 0

def ndcg_at_k(r_pred, type, k):
    if type=="strong":
        idcg_at_k = dcg_at_k(sorted(r_pred, reverse=True), k)
        return dcg_at_k(r_pred, k) / idcg_at_k
    else:
        idcg_at_k = dcg_normal_at_k(sorted(r_pred, reverse=True), k)
        return dcg_normal_at_k(r_pred, k) / idcg_at_k
    
def compute_ndcg_at_k(r_pred, type, k):
    scores = []
    for i in range(len(r_pred)):
        # import ipdb; ipdb.set_trace()
        scores.append(ndcg_at_k(r_pred[i], type, k))
    return sum(scores)/len(scores)

# def compute_ndcg(gold, prediction):
#     assert len(gold) == len(prediction)

#     # since the len of the synonyms group is different, we cannot cleanly covert them 2-darray
#     scores = []
#     for i in range(len(gold)):
#         # import ipdb; ipdb.set_trace()
#         scores.append(ndcg_score(np.expand_dims(np.asarray(gold[i]), axis=0), np.expand_dims(np.asarray(prediction[i]), axis=0)))
#     return sum(scores)/len(scores)

def compute_top1_acc(prediction):
    acc = 0
    for pred in prediction:
        pred = list(pred)
        pred_top1 = pred[-1]

        pred.sort(reverse=True)
        if pred_top1 == pred[-1]:
            acc += 1
    
    return acc/len(prediction)


def compute_top1(prediction):
    loc = 0
    for pred in prediction:
        pred_top1 = pred[-1]

        pred_filter = list(pred)
        pred_filter.sort(reverse=False)
        loc += pred_filter.index(pred_top1) + 1

    return loc/len(prediction)


def compute_RMS(gold, prediction):
    rms = 0 
    for y_true, y_pred in zip(gold, prediction):
        rms += mean_squared_error(y_true, y_pred, squared=False)
    return rms / len(gold)

# TODO: distribution of the top1