import torch
import torch.nn as nn
from transformers import BertModel,BertForSequenceClassification




class BertClassifier(nn.Module):
    def __init__(self, config, transformer_width, num_labels):
        super(BertClassifier, self).__init__()
        self.bert_layer = BertModel.from_pretrained(config["pretrain_model"])
        # self.tm = BertForSequenceClassification.from_pretrained(config["pretrain_model"])
        self.classifier_layer = nn.Linear(transformer_width, num_labels)
        # self.softmax = nn.Softmax(dim=-1)


    def forward(self, x, mask):
        _, y = self.bert_layer(x, mask)
        output = self.classifier_layer(y)
        # logits = self.softmax(output)
        return output


    '''
    TypeError: softmax() received an invalid combination of arguments - got (Tensor), but expected one of:
     * (Tensor input, name dim, *, torch.dtype dtype)
     * (Tensor input, int dim, torch.dtype dtype)
     '''