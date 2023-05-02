import os
import pandas as pd
import math
import matplotlib.pyplot as plt


def flat(df, score):
    raw = (list(set([a for b in df[str(score)].apply(lambda x: eval(x)).tolist() for a in b])))
    raw.sort(reverse=True)
    return raw

# # Note that all the scores are ratio of (un)normalized
# TODO: plot the distribution with more bins like 100
# raw_un_norm = flat(data, "un_normalized_score")
# raw_norm = flat(data, "normalized_score")
# data_toplot = [raw_un_norm, raw_norm]

# plt.boxplot(data_toplot,patch_artist=True,labels=["un_normalized_score", "normalized_score"])
# plt.show()
# plt.savefig("./output/p1_distribution.png")



def map_diff(x, raw):
    x = eval(x)
    x_new = []
    chop_id = len(raw)/5
    for a in x:
        if a < raw[math.ceil(chop_id)]:
            x_new.append(1)
        elif raw[math.ceil(chop_id)] <= a < raw[math.ceil(chop_id*2)]:
            x_new.append(2)
        elif raw[math.ceil(chop_id*2)] <= a < raw[math.ceil(chop_id*3)]:
            x_new.append(3)
        elif raw[math.ceil(chop_id*3)] <= a < raw[math.ceil(chop_id*4)]:
            x_new.append(4)
        else:
            x_new.append(5)

    return x_new
        

def get_diff(df, score, distribution="flat"):
    raw = (list(set([a for b in df[str(score)].apply(lambda x: eval(x)).tolist() for a in b])))
    raw.sort()


    if distribution == "flat":
        column_name = str(score)[:-5] + "diff_" + str(distribution)
        df[column_name] = df[score].apply(lambda x: map_diff(x, raw)) 
    return df


input_file = "./output/p2.tsv"
data = pd.read_table(input_file)
data = get_diff(data, "un_normalized_score")
data = get_diff(data, "normalized_score")

data.to_csv("./output/p2_.tsv", sep="\t")