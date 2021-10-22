import pandas as pd
import pprint

carto_df = pd.read_csv("cartography/filtered/cartography_confidence_0.05/MNLI/train.tsv", sep="\t")
cleanlab_df = pd.read_csv("cleanlab/train_preds/MNLI/mislabeled_smaller.tsv", sep="\t",  on_bad_lines="warn") 

carto_df = carto_df[:19000]
cleanlab_df = cleanlab_df[:19000]

shared = 0
total_examples = 0

ordered_shared = []

for idx in carto_df["index"]:
    total_examples += 1

    if idx in cleanlab_df["index"]:
        shared += 1
        
    if total_examples % 100 == 0:
        ordered_shared.append(shared)

shared = 0
total_examples = 0
topk_shared_clean = []
topk_shared_carto = []

clean = cleanlab_df["index"]
carto = carto_df["index"]


for idx in range(0, 19100, 100):
    topk_shared = 0
    if idx == 19000:
        idx = 18999
    for sub_idx in range(idx+1):
        if clean[sub_idx] in carto[:idx+1]:
            topk_shared += 1

    topk_shared_clean.append(topk_shared)

    topk_shared = 0

    for sub_idx in range(idx+1):
        if carto[sub_idx] in clean[:idx+1]:
            topk_shared += 1

    topk_shared_carto.append(topk_shared)

print("carto looking in clean")
print(topk_shared_carto)

print("clean looking in carto")
print(topk_shared_clean)            

print("examples in both methods: {}".format(shared))         


