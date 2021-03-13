import torch
import torch.nn as nn
from transformers import BertModel,BertForSequenceClassification, BertTokenizer, BertForPreTraining
from torch.nn import functional as F



class BertClassifier(nn.Module):
    def __init__(self, config, transformer_width = 768, num_labels=2):
        super(BertClassifier, self).__init__()
        # self.bert_layer = BertModel.from_pretrained(config["pretrain_model"])
        # self.classifier_layer = nn.Linear(transformer_width, num_labels)
        self.bert_layer = BertForSequenceClassification.from_pretrained(config["pretrain_model"], return_dict=True)

    def forward(self, batch, use_gpu):
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        mask_ids = batch["mask_ids"]
        labels = batch["labels"]

        if use_gpu:
            input_ids = input_ids.cuda()
            token_type_ids = token_type_ids.cuda()
            mask_ids = mask_ids.cuda()
            labels = labels.cuda()

        # _, y = self.bert_layer(input_ids=input_ids, attention_mask=mask_ids, token_type_ids=token_type_ids)
        # logits = self.classifier_layer(y)

        outputs = self.bert_layer(input_ids=input_ids, attention_mask=mask_ids, token_type_ids=token_type_ids)

        labels = torch.squeeze(labels, dim=-1)

        loss = F.cross_entropy(outputs.logits, labels)
        prediction = outputs.logits.argmax(-1)

        return loss, prediction








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


