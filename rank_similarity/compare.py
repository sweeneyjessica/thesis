import pandas as pd
import pprint
import scipy.stats as stats
from collections import Counter

carto_df = pd.read_csv("../cartography/filtered/cartography_confidence_0.05/MNLI/train.tsv", sep="\t")
cleanlab_df = pd.read_csv("../cleanlab/train_preds/MNLI/mislabeled_smaller.tsv", sep="\t",  on_bad_lines="warn") 

shared_carto = 0
shared_clean = 0

ordered_shared = []

carto = list(carto_df["index"])
clean = list(cleanlab_df["index"])

shared_carto_ordering = []
shared_clean_ordering = []

for idx in clean:
    if idx in carto:
        shared_clean += 1
        shared_clean_ordering.append(idx)
    if shared_clean == 100:
        break

for idx in carto:
    if idx in shared_clean_ordering:
        shared_carto += 1
        shared_carto_ordering.append(idx)
    if shared_carto == 100:
        break

print("confirming same number both times: {} {}".format(shared_clean, shared_carto))

tau, p_value = stats.kendalltau(shared_clean_ordering, shared_carto_ordering)

print("tau: {}".format(tau))
print("p_value: {}".format(p_value))
