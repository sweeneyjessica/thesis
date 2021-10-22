import matplotlib.pyplot as plt
import pandas as pd
import pprint
from collections import Counter

carto_df = pd.read_csv("cartography/filtered/cartography_confidence_0.05/MNLI/train.tsv", sep="\t")
cleanlab_df = pd.read_csv("cleanlab/train_preds/MNLI/mislabeled_smaller.tsv", sep="\t",  on_bad_lines="warn") 
ensemble_df = pd.read_csv("all_disagreements.tsv", sep="\t", on_bad_lines="warn")


carto_df = carto_df[:19000]
cleanlab_df = cleanlab_df[:19000]
ensemble_df = ensemble_df[:19000]

carto = carto_df["index"]
clean = cleanlab_df["index"]
ensemble = ensemble_df["index"]

carto_as_set = list(carto)
clean_as_set = list(clean)
ensemble_as_set = list(ensemble)

shared_at_junction = []
shared_at_junction_clean = []
shared_at_junction_carto = []

for idx in range(0, 501):
    #if idx == 19000:
    #    idx = 18999
    
    sub_ensemble = set(ensemble_as_set[:14000])
    sub_carto = set(carto_as_set[:idx])
    sub_clean = set(clean_as_set[:idx]) 
    shared_at_junction.append(len(sub_carto.intersection(sub_clean)))
    shared_at_junction_clean.append(len(sub_ensemble.intersection(sub_clean)))
    shared_at_junction_carto.append(len(sub_ensemble.intersection(sub_carto)))

#print("total shared among:")
#print("clean x carto: {}".format(shared_at_junction[-1]))
#print("clean x ensem: {}".format(shared_at_junction_clean[-1]))
#print("carto x ensem: {}".format(shared_at_junction_carto[-1]))


plt.scatter([x*100 for x in range(len(shared_at_junction))], shared_at_junction, label="cartoxclean")
plt.scatter([x*100 for x in range(len(shared_at_junction_clean))], shared_at_junction_clean, label="cartoxensemble")
plt.scatter([x*100 for x in range(len(shared_at_junction_carto))], shared_at_junction_carto, label="cleanxensemble")
plt.legend()
plt.savefig("shared_all_500_2.png")
