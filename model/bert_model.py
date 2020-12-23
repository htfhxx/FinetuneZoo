import torch
import torch.nn as nn
from transformers import BertModel,BertForSequenceClassification, BertTokenizer, BertForPreTraining




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





def test():
    bert_model_path = '../checkpoints/bert-base-chinese/' # pytorch_model.bin
    bert_config_path = '../checkpoints/bert-base-chinese/' # bert_config.json
    vocab_path = '../checkpoints/bert-base-chinese/vocab.txt'

    tokenizer = BertTokenizer.from_pretrained(vocab_path)
    # model = BertModel.from_pretrained(bert_model_path, config=bert_config_path)
    model = BertForPreTraining.from_pretrained(bert_model_path, config=bert_config_path)

    text_batch = ["哈哈哈", "嘿嘿嘿", "嘿嘿嘿", "嘿嘿嘿"]
    encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
    input_ids = encoding['input_ids']
    print(input_ids)
    print(input_ids.shape)
    output1,output2 = model(input_ids)
    print(output1)
    print(output2)
    print(output1.shape)
    print(output2.shape)


if __name__ == '__main__':
    test()


