
from sklearn.metrics import f1_score,recall_score,precision_score, accuracy_score
import numpy as np
import torch
import torch.nn as nn




class LabelScorer:
    def __init__(self):
        self.clear()


    def clear(self):
        self.prediciton_list = []
        self.reference_list = []


    def update(self, prediction, labels):
        # print(prediction)
        # print(labels)
        self.prediciton_list.extend(prediction)
        self.reference_list.extend(labels)


    def get_avg_scores(self):
        accu = accuracy_score(self.reference_list, self.prediciton_list)
        micro_precision = precision_score(self.reference_list, self.prediciton_list)
        micro_recall = recall_score(self.reference_list, self.prediciton_list)
        micro_f1 = f1_score(self.reference_list, self.prediciton_list)
        return accu*100, micro_precision*100, micro_recall*100,  micro_f1*100


        #
        # macro_precision = precision_score(self.reference_list, self.prediciton_list,  average="macro")
        # macro_recall = recall_score(self.reference_list, self.prediciton_list,  average="macro")
        # macro_f1 = f1_score(self.reference_list, self.prediciton_list,  average="macro")
        #
        # weighted_precision = precision_score(self.reference_list, self.prediciton_list, average="weighted")
        # weighted_recall = recall_score(self.reference_list, self.prediciton_list, average="weighted")
        # weighted_f1 = f1_score(self.reference_list, self.prediciton_list,  average="weighted")

        # return micro_precision,  micro_recall, micro_f1

    def print_avg_scores(self):
        # avg_precisions,  avg_recalls, avg_F1 = self.get_avg_scores()
        # print(f"Average micro precisions: {avg_precisions}")
        # print(f"Average micro recalls: {avg_recalls}")
        # print(f"Average micro f1: {avg_F1}")
        accu = self.get_avg_scores()
        print(f"Average micro accuracy: {accu}")



