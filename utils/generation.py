import torch
import math
import numpy as np
from scipy.special import softmax
from transformers import pipeline, set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer


"""
Output: A list of SCORE=AVG(LOG(PR(T_i))) w.r.t. a set of synonyms
"""
def iterate_geneator(model_name, prompt, synonyms):

  model = AutoModelForCausalLM.from_pretrained(model_name, return_dict_in_generate=True)
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids

  scores = []
  for syn in synonyms:
    # print(syn)
    syn_ids = tokenizer(syn, return_tensors="pt").input_ids
    conditional_scores = []

    generated_outputs = model.generate(input_ids, top_k=50257, max_new_tokens=1, num_beams=1, do_sample=False, num_return_sequences=1, output_scores=True)
    probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)
    # print('first-word distribution:', probs[0, 0, syn_ids[0, 0]].item())
    conditional_scores.append(probs[0, 0, syn_ids[0, 0]].item())

    if syn_ids.shape[1] > 1:
      for i in range(1, syn_ids.shape[1]+1): 
        toadd = torch.reshape(syn_ids[0,i-1], (1,-1))
        newinput_ids = torch.cat((input_ids, toadd), dim=1)
        generated_outputs = model.generate(newinput_ids, top_k=50257, max_new_tokens=1, num_beams=1, do_sample=False, num_return_sequences=1, output_scores=True)
        probs = torch.stack(generated_outputs.scores, dim=1).softmax(-1)
        # conditional_scores.append(probs[0, 0, syn_ids[0, i]].item())
        conditional_scores.append(math.log(probs[0, 0, syn_ids[0, i]].item()))

        if i+1==syn_ids.shape[1]:
          break

    # print(conditional_scores)
    # print(math.prod(conditional_scores))
    avg_score = sum(conditional_scores)/len(conditional_scores)
    scores.append(avg_score)

  return scores 


def score_normalized(scores):
  return softmax(scores)


def score_ratio(scores_lay, scores_pro):

  assert len(scores_lay) == len(scores_pro)
  return [i / j for i, j in zip(scores_lay, scores_pro)]
