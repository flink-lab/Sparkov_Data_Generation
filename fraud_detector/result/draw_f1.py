import pandas as pd
from pandas import DataFrame
import numpy as np


class F1DataBuilder:

    def __init__(self):
        self.col_index = {"TP": 1, "TN": 2, "FP": 3, "FN": 4, "total": 5}

    def build_data(self, f1: DataFrame):
        res = dict()
        for index, row in f1.iterrows():
            time_stamp = row['timestamp']
            insert_index = self.col_index[row['type']]
            if time_stamp not in res.keys():
                record = [time_stamp, 0, 0, 0, 0, 0]
                res[time_stamp] = record
            else:
                record = res.get(time_stamp)
            record[insert_index] = row['count']
        res_list = sorted(list(res.values()), key=lambda x: x[0])
        total_index = self.col_index['total']
        first_timestamp = res_list[0][0]
        for record in res_list:
            record[total_index] = sum(record[1:total_index])
            record[0] -= first_timestamp
        return np.asarray(res_list)


def draw(confusion_matrix_update, confusion_matrix_raw):
    from matplotlib import pyplot as plt

    color = ['k', 'r', 'b', 'c', 'm', 'g']
    labels = ["Precision=TP/(TP+FP)", "Recall=TP/(TP+FN)", "F1 score for fraud",
              "Precision2=TN/(TN+FN)", "Recall2=TN/(TN+FP)", "F1 score for normal"]

    def get_f1(confusion_matrix):
        precision = confusion_matrix[:, 1] / (confusion_matrix[:, 1] + confusion_matrix[:, 3])
        recall = confusion_matrix[:, 1] / (confusion_matrix[:, 1] + confusion_matrix[:, 4])
        precision2 = confusion_matrix[:, 2] / (confusion_matrix[:, 2] + confusion_matrix[:, 4])
        recall2 = confusion_matrix[:, 2] / (confusion_matrix[:, 2] + confusion_matrix[:, 3])
        f1score_fraud = 2 * precision * recall / (precision + recall)
        f1score_normal = 2 * precision2 * recall2 / (precision2 + recall2)
        return f1score_fraud

    f1score_fraud_update = get_f1(confusion_matrix_update)
    f1score_fraud_raw = get_f1(confusion_matrix_raw)
    # plt.plot(confusion_matrix[:, 0], precision, c=color[0], label=labels[0])
    # plt.plot(confusion_matrix[:, 0], recall, c=color[1], label=labels[1])
    #
    # plt.plot(confusion_matrix[:, 0], precision2, c=color[3], label=labels[3])
    # plt.plot(confusion_matrix[:, 0], recall2, c=color[4], label=labels[4])
    plt.plot(confusion_matrix[:, 0], f1score_fraud_update, c=color[2], label=labels[2])
    # plt.plot(confusion_matrix[:, 0], f1score_normal, c=color[5], label=labels[5])
    for i in [120, 225, 345]:
        if len(confusion_matrix_update) <= i:
            break
        plt.plot(confusion_matrix_update[i, 0], f1score_fraud_update[i], 'ro')

    plt.plot(confusion_matrix_raw[:, 0], f1score_fraud_raw, c='m', label="F1 score for fraud without model update")
    plt.xlim(150, 600)
    plt.ylim(0, 1)

    plt.xlabel('time /s')
    plt.ylabel('score')
    plt.title('Evaluation Result by applying Model Update (At red point)')

    plt.legend(loc=0)
    plt.show()


if __name__ == '__main__':
    f1 = pd.read_csv("confusion_matrix_update7.csv")
    f2 = pd.read_csv("confusion_matrix_raw7.csv")
    builder = F1DataBuilder()
    confusion_matrix = builder.build_data(f1)
    confusion_matrix_raw = builder.build_data(f2)
    draw(confusion_matrix, confusion_matrix_raw)
    # code = r""
