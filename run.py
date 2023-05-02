import csv
import numpy as np
from utils.generation import iterate_geneator, score_normalized, score_ratio
from utils.data import syn_data
from utils.get_prompt import syn_defs

'''
generation times = 2*N(prompt)*N(synonyms)*LEN(tokenized(synonyms))
'''

output_file = "./output/p4.tsv"
fixed_prompt=False

with open(output_file, 'w', encoding='utf8', newline='') as tsv_file:
    tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
    tsv_writer.writerow(["synonyms", "scores_gpt", "scores_biogpt", "un_normalized_score", "normalized_score" ])
    
    if fixed_prompt==False:

        ind = 0 
        for syn_def, synonyms in zip(syn_defs, syn_data): 

            prompt = "What is: " + syn_def + "?" 
            scores_gpt = iterate_geneator("gpt2", prompt=prompt, synonyms=synonyms)
            scores_biogpt = iterate_geneator("microsoft/biogpt", prompt=prompt, synonyms=synonyms)

            np.set_printoptions(precision=3)

            un_normalized_score = score_ratio(scores_gpt , scores_biogpt)
            normalized_score = score_ratio(score_normalized(scores_gpt) , score_normalized(scores_biogpt))

            tsv_writer.writerow([synonyms, scores_gpt, scores_biogpt, un_normalized_score, normalized_score])

            ind +=1 
            print(ind)
        

    else:
        # prompt = "I have " p1
        # prompt = "The easiest term is " p2

        ind = 0 
        for synonyms in syn_data: 

            scores_gpt = iterate_geneator("gpt2", prompt=prompt, synonyms=synonyms)
            scores_biogpt = iterate_geneator("microsoft/biogpt", prompt=prompt, synonyms=synonyms)

            np.set_printoptions(precision=3)

            un_normalized_score = score_ratio(scores_gpt , scores_biogpt)
            normalized_score = score_ratio(score_normalized(scores_gpt) , score_normalized(scores_biogpt))

            tsv_writer.writerow([synonyms, scores_gpt, scores_biogpt, un_normalized_score, normalized_score])

            ind +=1 
            print(ind)



    