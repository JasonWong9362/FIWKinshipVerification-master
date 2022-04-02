import pandas as pd
import numpy as np
from glob import glob
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


all_loss = ["binary_crossentropy", "FL_alpha05_gamma05", "FL_alpha05_gamma1"]
col_name = ["loss_function", "threshold", "relationship", "acc", ]
record = pd.DataFrame(columns=col_name)

for loss in all_loss:
    which_loss = loss

    root_path = "../submission/" + which_loss
    # list all relationships
    all_lists = glob("../input/test-private-lists/" + "*.csv")
    all_labels = glob("../input/test-private-labels/" + "*.csv")
    all_relationships = [l.split('\\')[1].split('.')[0] for l in all_lists]

    for threshold in range(2,9):  # define threshold

        threshold = threshold * 0.1
        threshold = round(threshold, 1)
        avg_acc = []

        for i, relationship in enumerate(all_relationships):

            # read result
            read_path = root_path + "/" + relationship + ".csv"
            result_pd = pd.read_csv(read_path, header=0)
            result_np = result_pd.values
            preds = np.reshape(result_np, result_np.shape[0])

            # convert to binary
            preds = preds > threshold  # ** reverse **
            preds = preds.astype(int)

            # read label
            labels = pd.read_csv(all_labels[i])
            labels = labels.values
            labels = np.reshape(labels, labels.shape[0])

            # append in df
            raw_acc = accuracy_score(labels, preds)
            acc = round(accuracy_score(labels, preds), 3)
            avg_acc.append(raw_acc)
            type_name = loss + " " + str(threshold) + " " + relationship
            new_row = {"loss_function": loss, "threshold": str(threshold), "relationship": relationship,
                       'acc': acc}
            record = record.append(new_row, ignore_index=True)

        avg_acc = np.array(avg_acc)
        avg_acc = round(np.mean(avg_acc), 3)
        new_row = {"loss_function": 'AVG', "threshold": " ", "relationship": " ",
                   'acc': avg_acc}
        record = record.append(new_row, ignore_index=True)

    record.to_csv("../submission/result.csv", index=False)
