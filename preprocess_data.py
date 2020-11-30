import argparse
import json
import numpy as np
import pickle

from transformers import BertTokenizer

def process_set(file, tokenizer):
    content = {
        "text": list(),
        "label": list()
    }
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line_list = line.strip().split('\t')
            if(len(line_list)!=3):
                print('data process wrong..')
                exit()
            # index_a = tokenizer.encode(line_list[0])
            # index_b = tokenizer.encode(line_list[1])
            index_ab = tokenizer.encode(line_list[0]+' [SEP] '+line_list[1])
            # print(line_list[0]+'[SEP]'+line_list[1])
            # print(index_a)
            # print(index_b)
            # print(index_ab)
            # print('--')
            content['text'].append(index_ab)
            content["label"].append(int(line_list[2]))
    return content


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--config", default="utils/config/preprocess_data_LCQMC.config")
    parser.add_argument("--train_dir", default="data/LCQMC/train.txt")
    parser.add_argument("--dev_dir", default="data/LCQMC/dev.txt")
    parser.add_argument("--test_dir", default="data/LCQMC/test.txt")
    parser.add_argument("--vocab_path", default="checkpoints/roberta_wwm_ext_pytorch/vocab.txt")
    parser.add_argument("--indexed_data_dir", default="data/LCQMC/")
    args = parser.parse_args()
    # with open(args.config, "r") as config_file:
    #     config = json.loads(config_file.read())

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)


    print('tokenize train data...')
    indexed_train = process_set(args.train_dir, tokenizer)
    print('tokenize dev data...')
    indexed_dev = process_set(args.dev_dir, tokenizer)
    print('tokenize test data...')
    indexed_test = process_set(args.test_dir, tokenizer)
    print('writing to json')
    with open(args.indexed_data_dir+'/'+'indexed_train_bert.json', "w") as f:
        f.write(json.dumps(indexed_train))
    with open(args.indexed_data_dir+'/'+'/indexed_dev_bert.json', "w") as f:
        f.write(json.dumps(indexed_dev))
    with open(args.indexed_data_dir+'/'+'/indexed_test_bert.json', "w") as f:
        f.write(json.dumps(indexed_test))
    print('finished!')


if __name__ == "__main__":
    main()
