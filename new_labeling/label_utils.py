import pandas as pd
import sys

carto_data = "/home2/js20/thesis/cartography/filtered/cartography_confidence_0.05/MNLI/train.tsv"
cleanlab_data = "/home2/js20/thesis/cleanlab/train_preds/MNLI/mislabeled.tsv"
conll_data = "/home2/js20/thesis/conll_method/mislabeled.tsv"

full_data = "~/thesis/data/mnli/MNLI/train.tsv"


def subsample(data_file):
    data = pd.read_csv(data_file, sep="\t", on_bad_lines="warn", engine="python")
    data = data[:500]
    entail_sample = data.loc[data['gold_label'] == 'entailment'].sample(n=34)
    contra_sample = data.loc[data['gold_label'] == 'contradiction'].sample(n=34)
    neutra_sample = data.loc[data['gold_label'] == 'neutral'].sample(n=34)

    random_sample = pd.concat([entail_sample, contra_sample, neutra_sample])
    random_sample = random_sample.sample(frac=1)

    return random_sample

def interactive_labeling(random_sample, identifier, first=False):
    data = random_sample 
    labels = []
    key = {"1":"Entailment", "2":"Neutral", "3":"Contradiction"}

    for idx, example in data.iterrows():
        sentence1 = example["sentence1"]
        sentence2 = example["sentence2"]
        
        print("\n")
        print(sentence1)
        print(sentence2) 
        print("Entailment: 1, Neutral: 2, Contradiction: 3")
        label = input()
        if label == "exit":
            with open("{}_labels.txt".format(identifier), "a") as output:
                for lab in labels:
                    output.write("{}\n".format(lab))
            if first:
                with open("{}_leftover_for_labeling.csv".format(identifier), "wb") as output:
                    random_sample.to_csv(output, mode="wb")
            break   
        labels.append(key[label])

    if label == "exit":
       return 
    
    with open("{}_labels.txt".format(identifier), "a") as output:
        for lab in labels:
            output.write("{}\n".format(lab))

def read_data(resume_idx, identifier):
    data = pd.read_csv("{}_leftover_for_labeling.csv".format(identifier)) 
    return data[resume_idx:]


def compare_labels_general(identifier):
    assigned = []
    with open(f"{identifier}_labels.txt") as label_input:
        for lab in label_input:
            assigned.append(lab.strip().lower())
        
    
    identifier_data = pd.read_csv(f"{identifier}_leftover_for_labeling.csv")

    identifier_mislabeled = {"entailment":{"entailment":0, "neutral":0, "contradiction":0}, "neutral":{"entailment":0, "neutral":0, "contradiction":0}, "contradiction":{"entailment":0, "neutral":0, "contradiction":0}}


    #dict[given_label][my_label] entail, neutral, contra
    with open("clean_yn.txt", "w") as output:
        for idx, example in identifier_data.iterrows():
            identifier_mislabeled[example["gold_label"]][assigned[idx]] += 1
            if example["gold_label"] == assigned[idx]:
                output.write("Wrong\n")
            else:
                output.write("Right\n")

    print(f"\n\n{identifier}")
    print_confusion_matrix(identifier_mislabeled)

def compare_labels():
    carto = []
    with open("carto_labels.txt") as label_input:
        for lab in label_input:
            carto.append(lab.strip().lower())
        
    cleanlab = []
    with open("cleanlab_labels.txt") as label_input:
        for lab in label_input:
            cleanlab.append(lab.strip().lower())
    
    clean_data = pd.read_csv("cleanlab_leftover_for_labeling.csv")
    carto_data = pd.read_csv("carto_leftover_for_labeling.csv")

    clean_mislabeled = {"entailment":{"entailment":0, "neutral":0, "contradiction":0}, "neutral":{"entailment":0, "neutral":0, "contradiction":0}, "contradiction":{"entailment":0, "neutral":0, "contradiction":0}}

    carto_mislabeled = {"entailment":{"entailment":0, "neutral":0, "contradiction":0}, "neutral":{"entailment":0, "neutral":0, "contradiction":0}, "contradiction":{"entailment":0, "neutral":0, "contradiction":0}}

    #dict[given_label][my_label] entail, neutral, contra
    with open("clean_yn.txt", "w") as output:
        for idx, example in clean_data.iterrows():
            clean_mislabeled[example["gold_label"]][cleanlab[idx]] += 1
            if example["gold_label"] == cleanlab[idx]:
                output.write("Wrong\n")
            else:
                output.write("Right\n")

    with open("carto_yn.txt", "w") as output:
        for idx, example in carto_data.iterrows():
            carto_mislabeled[example["gold_label"]][carto[idx]] += 1
            if example["gold_label"] == carto[idx]:
                output.write("Wrong\n")
            else:
                output.write("Right\n")

    print("Cleanlab")
    print_confusion_matrix(clean_mislabeled)

    print("\n\nCarto")
    print_confusion_matrix(carto_mislabeled)


def print_confusion_matrix(confusion_test):
    # Confusion_train and _test are build from running classification after
    # building the classifier. They're dicts where the keys are the gold
    # labels, and their values are dicts whose keys are the silver labels,
    # and those values are the number of examples seen with that combination
    # of gold and silver labels.

    test_gold_labels = list(confusion_test.keys())
    test_train_labels = list(confusion_test[test_gold_labels[0]].keys())
    test_acc = 0
    test_total = 0

    print("Confusion matrix:")
    print("row is the gold from SNLI, column is my label\n")
    column_names = "             "
    for lab in sorted(test_train_labels):
        column_names += lab + " "

    print(column_names)

    for lab in sorted(test_gold_labels):
        row = lab
        for guess in sorted(confusion_test[lab].keys()):
            row += " " + str(confusion_test[lab][guess])
            if guess == lab:
                test_acc += confusion_test[lab][guess]
            test_total += confusion_test[lab][guess]

        print(row)

    test_acc = test_acc / test_total

    print("\n Test accuracy={}".format(test_acc))
    print("For our purposes, acc should be low")

if __name__ == "__main__":
    """if len(sys.argv) == 2:
        resume = int(sys.argv[1]) 

        random_sample = pd.read_csv("clean_random_leftover_for_labeling.csv")[resume:] 
        interactive_labeling(random_sample, "clean_random")
    else:
        random = subsample(carto_data, "carto")
        interactive_labeling(random, "carto", first=True)
    """
    #clean_sample = subsample(cleanlab_data)
    #carto_sample = subsample(carto_data)
    #conll_sample = subsample(conll_data)

    resume = int(sys.argv[1])

    sample = pd.read_csv("carto_leftover_for_labeling.csv")[resume:]
    interactive_labeling(sample, "carto", first=False)
     
