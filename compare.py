import pandas as pd
import pprint
from collections import Counter

carto_df = pd.read_csv("cartography/filtered/cartography_confidence_0.05/MNLI/train.tsv", sep="\t")
cleanlab_df = pd.read_csv("cleanlab/train_preds/MNLI/mislabeled_smaller.tsv", sep="\t",  on_bad_lines="warn") 

carto_df = carto_df[:19000]
cleanlab_df = cleanlab_df[:19000]

shared = 0
total_examples = 0

ordered_shared = []

carto = list(carto_df["index"])
clean = list(cleanlab_df["index"])

print("Unique indices in cartography: {}".format(len(Counter(carto))))
print("Unique indices in cleanlab: {}".format(len(Counter(clean))))

for idx in clean:
    total_examples += 1

    if idx in carto:
        shared += 1
        
    if total_examples % 100 == 0:
        ordered_shared.append(shared)

print("total shared examples: {}".format(shared))
print("total examples: {}".format(total_examples))
print("shared throughout: {}".format(ordered_shared))
print(len(ordered_shared))

shared = 0
total_examples = 0
ordered_shared = []

for idx in carto:
    total_examples += 1
    
    if idx in clean:
        shared += 1

    if total_examples % 100 == 0:
        ordered_shared.append(shared)

print("total shared examples: {}".format(shared))
print("total examples: {}".format(total_examples))
print("shared throughout: {}".format(ordered_shared))
print(len(ordered_shared))
