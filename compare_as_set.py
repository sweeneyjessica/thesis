import pandas as pd
import pprint
from collections import Counter

carto_df = pd.read_csv("cartography/filtered/cartography_confidence_0.05/MNLI/train.tsv", sep="\t")
cleanlab_df = pd.read_csv("cleanlab/train_preds/MNLI/mislabeled_smaller.tsv", sep="\t",  on_bad_lines="warn") 

carto_df = carto_df[:19000]
cleanlab_df = cleanlab_df[:19000]

carto = carto_df["index"]
clean = cleanlab_df["index"]

carto_as_set = set(carto)
clean_as_set = set(clean)

intersection = carto_as_set.intersection(clean)

print("Carto x clean: {} ".format(len(intersection)))

intersection = clean_as_set.intersection(carto)

print("Clean x carto: {} ".format(len(intersection)))
