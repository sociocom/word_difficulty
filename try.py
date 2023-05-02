import numpy as np
import math
from sklearn.metrics import ndcg_score
from sklearn.metrics import top_k_accuracy_score
from utils.metrics import compute_ndcg, compute_ndcg_at_k, ndcg_at_k 

# y_true = np.asarray([[10, 0, 0, 1, 5]])
# y_score = np.asarray([[.1, .2, .3, 4, 70]])

# y_true = np.asarray([[10, 5, 1, 0, 0]])
# y_score = np.asarray([[.1, 70, .2, 4, .3]])

# def prob(scores):

#     list_prob = []
#     for i in range(scores.shape[1]):
#         print(scores[0][i])
#         list_prob.append(scores[0][i] / math.log(i+1+1, 2))

#     return list_prob

# dcg = sum(prob(y_score))
# idcg = sum(prob( y_true))


y_true = [5,2,2,1,1]
y_score = [5,5,1,2,1]

# DCG_list = prob(y_true)
# IDCG_list = prob(y_score)


# print(top_k_accuracy_score(y_true[0], y_score[0], k=1))

# print(sum(DCG_list)/sum(IDCG_list))
# print(ndcg_score(y_true, y_score))

# print(ndcg_score(y_true, y_score, k=1))
import ipdb; ipdb.set_trace()