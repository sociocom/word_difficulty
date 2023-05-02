from utils.metrics import compute_ndcg_at_k, compute_top1_acc, compute_top1, compute_RMS
from utils.data import gold
import pandas as pd
from ast import literal_eval
import numpy as np
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_path')
    parser.add_argument('--output_path')
    args = parser.parse_args()

    # read file
    pred = np.load(args.pred_path)
    assert len(pred) == len(gold)

    # sort the gold, sort the pred with gold
    gold_sort, pred_sort = sort_all(gold, pred)

    # compute the metrics
    ndcg = compute_ndcg_at_k(pred_sort, type="normal", k=-1)
    s_ndcg = compute_ndcg_at_k(pred_sort, type="strong", k=-1)
    top1_acc = compute_top1_acc(pred_sort)
    top1_loc = compute_top1(pred_sort)
    rms = compute_RMS(gold_sort, pred_sort)

    # write a file
    with open(args.output_path) as f:
        f.write("ndcg: {} \n".format(ndcg))
        f.write("s-ndcg: {} \n".format(s_ndcg))
        f.write("top1_acc: {} \n".format(top1_acc))
        f.write("top1_loc: {} \n".format(top1_loc))
        f.write("rms: {} \n".format(rms))

if __name__=="__main__":
    main()