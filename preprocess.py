import argparse
import json
import os
import numpy as np
import pickle

from transformers import BertTokenizer


def truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()
    return tokens_a,tokens_b

def process_set(file, tokenizer, max_seq_length):
    content = {
        "tokens": list(),
        "input_ids": list(),
        "token_type_ids": list(),
        "mask_ids": list(),
        "labels": list()
    }
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            line_list = line.strip().split('\t')
            if(len(line_list)!=3):
                print('data process wrong..')
                exit()

            tokens = []
            token_type_ids = []

            texta = line_list[0]
            textb = line_list[1]
            tokens_a = tokenizer.tokenize(texta)
            tokens_b = tokenizer.tokenize(textb)
            tokens_a, tokens_b = truncate_seq_pair(tokens_a, tokens_b, max_seq_length-3)

            # 搞定tokens和segment_ids
            tokens.append("[CLS]")
            token_type_ids.append(0)
            for token in tokens_a:
                tokens.append(token)
                token_type_ids.append(0)
            tokens.append("[SEP]")
            token_type_ids.append(0)
            if tokens_b:
                for token in tokens_b:
                    tokens.append(token)
                    token_type_ids.append(1)
            tokens.append("[SEP]")
            token_type_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)


            content['tokens'].append(tokens)
            content['input_ids'].append(input_ids)
            content['token_type_ids'].append(token_type_ids)
            content['mask_ids'].append(input_mask)
            content['labels'].append(int(line_list[2]))
        assert len(content['tokens']) == len(content['input_ids']) == len(content['token_type_ids']) == len(content['mask_ids']) == len(content['labels'])
    return content


def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--config", default="utils/config/preprocess_data_LCQMC.config")
    parser.add_argument("--data_dir", default="data/original/LCQMC/")
    parser.add_argument("--save_dir", default="data/indexed/LCQMC/")
    parser.add_argument("--vocab_path", default="checkpoints/bert-base-chinese/vocab.txt")
    parser.add_argument("--max_seq_length", type=int, default=128)
    args = parser.parse_args()

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    tokenizer = BertTokenizer.from_pretrained(args.vocab_path)

    print('processing train data...')
    indexed_train = process_set(args.data_dir+'/train.tsv', tokenizer, args.max_seq_length)
    with open(args.save_dir+'/'+'indexed_train.json', "w",encoding='utf-8') as f:
        f.write(json.dumps(indexed_train,ensure_ascii=False))

    print('processing dev data...')
    indexed_dev = process_set(args.data_dir+'/dev.tsv', tokenizer, args.max_seq_length)
    with open(args.save_dir+'/'+'/indexed_dev.json', "w",encoding='utf-8') as f:
        f.write(json.dumps(indexed_dev,ensure_ascii=False))

    if os.path.exists(args.data_dir+'/test.tsv'):
        print('processing test data...')
        indexed_test = process_set(args.data_dir+'/test.tsv', tokenizer, args.max_seq_length)
        with open(args.save_dir+'/'+'/indexed_test.json', "w",encoding='utf-8') as f:
            f.write(json.dumps(indexed_test,ensure_ascii=False))

    print('finished!')


if __name__ == "__main__":
    main()
