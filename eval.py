from utils.metrics import compute_ndcg_at_k, compute_top1_acc, compute_top1, compute_RMS
from utils.data import gold
from sklearn.metrics import dcg_score
import pandas as pd
from ast import literal_eval

def sort_list(list1, list2):
    list1_sort, list2_sort = zip(*sorted(zip(list1, list2), reverse=True)) 
    return list1_sort, list2_sort


def sort_all(gold, pred):
    gold_sort = []
    pred_sort = []
    for i in range(len(gold)):
        list1_sort, list2_sort = sort_list(gold[i], pred[i])
        gold_sort.append(list1_sort)
        pred_sort.append(list2_sort)
    return gold_sort, pred_sort

def check_distribution(data):

    equal_score =0 
    two_score=0
    for i in range(len(data)):
        if len(set(data[i])) == 1:
            equal_score += 1
        elif len(set(data[i])) == 2:
            two_score += 1
    
    print("size:", len(data))
    print("equal_score:", equal_score)
    print("two_score:", two_score)


# load gold and generation
check_distribution(gold)

pred_holder = gold
# pred_baseline = [[1 for x in group] for group in pred_baseline]
pred_baseline = []
for group in pred_holder:
    pred_baseline.append([*range(len(group),0,-1)])

# import ipdb; ipdb.set_trace()
#
gold_sort, baseline_sort = sort_all(gold, pred_baseline)
# baseline = compute_ndcg(gold_sort, baseline_sort)
# baseline = compute_ndcg_at_k(gold_sort, baseline_sort, k=-1)

# import ipdb; ipdb.set_trace()

# baseline = compute_ndcg_at_k(baseline_sort, type, k=-1)
# baseline_top1 = compute_top1(baseline_sort)
# baseline_top_acc = compute_top1_acc(baseline_sort)
# print(baseline_top1)
# print(baseline_top_acc)
# print("\n")


path_list = ["./output/p1_.tsv", "./output/p2_.tsv", "./output/p3_.tsv", "./output/p4_.tsv"]
for pred_path in path_list:
    # pred_path = "./output/p1_.tsv"
    pred = pd.read_table(pred_path)

    pred_un_normalized = pred["un_normalized_diff_flat"].apply(lambda x: literal_eval(str(x))).tolist()
    pred_normalized = pred["normalized_diff_flat"].apply(lambda x: literal_eval(str(x))).tolist()

    # sort the gold, sort the pred with gold
    gold_sort, pred_un_normalized_sort = sort_all(gold, pred_un_normalized)
    gold_sort, pred_normalized_sort = sort_all(gold, pred_normalized)
    
    # normalized = compute_ndcg(gold_sort, pred_normalized_sort)
    # un_normalized = compute_ndcg(gold_sort, pred_un_normalized_sort)\

    # normalized = compute_ndcg_at_k(pred_normalized_sort, type, k=-1)
    # un_normalized = compute_ndcg_at_k(pred_un_normalized_sort, type, k=-1)
    # print(un_normalized, ",", normalized)

    # un_normalized = compute_top1_acc(pred_un_normalized_sort)
    # normalized = compute_top1_acc(pred_normalized_sort)
    # print("un_normal acc: ", un_normalized)
    # print("normal acc: ", normalized)

    # un_normalized = compute_top1(pred_un_normalized_sort)
    # normalized = compute_top1(pred_normalized_sort)
    # print("un_normal loc: ", un_normalized)
    # print("normal loc: ", normalized)

    un_normalized = compute_RMS(gold_sort, pred_un_normalized_sort)
    normalized = compute_RMS(gold_sort,pred_normalized_sort)
    print("un_normal rms: ", un_normalized)
    print("normal rms: ", normalized)
    
    # print("loc: ", loc)
    print("\n")