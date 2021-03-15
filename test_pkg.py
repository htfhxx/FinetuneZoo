import argparse
import json
import os
import time
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.Criterion import LabelScorer
from utils.Data import *
from model.bert_model import BertClassifier
from transformers import AdamW
from utils.util import *


parser = argparse.ArgumentParser()
parser.add_argument("--config", default="utils/train_bert.config")
parser.add_argument("--save_dir_name", default="balalala")
parser.add_argument("--pretrain_model", default="checkpoints/bert-base-chinese/")
parser.add_argument("--load_model", default="no")  # yes or no
parser.add_argument("--load_checkpoint", default="checkpoints/best_checkpoints.model")
parser.add_argument("--mode", default="test")  #   test or reference
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=16) # 16
parser.add_argument("--epoches", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=0.00002)
parser.add_argument("--use_gpu", default='yes')  # no
parser.add_argument("--gpus", default='0')
args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = json.loads(config_file.read())

config["config"] = args.config
config["save_dir_name"] = args.save_dir_name
config["pretrain_model"] = args.pretrain_model
config["load_model"] = args.load_model
config["load_checkpoint"] = args.load_checkpoint
config["mode"] = args.mode
config["max_len"] = args.max_len
config["batch_size"] = int(args.batch_size)
config["epoches"] = args.epoches
config["seed"] = args.seed
config["eval_steps"] = args.eval_steps
config["learning_rate"] = args.learning_rate

logging,log_path, checkpoint_dir = set_up_logging(config)

set_seed(config["seed"])

if args.use_gpu == 'yes' and torch.cuda.is_available():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    config['use_gpu'] = True
    config['gpus'] = args.gpus
    logging("=" * 20+"Running on device: {}  ".format(args.gpus)+"=" * 20)
elif args.use_gpu == 'no':
    config['use_gpu'] = False
    logging("=" * 20+"Running on device: cpu" + "=" * 20)
else:
    logging("Wrong GPU configuration..")
    exit()






def dev_epoch(epoch, config, model, dev_loader):
    model.eval()

    dev_loss = 0.0
    #dev_loader_tqdm = tqdm(dev_loader, ncols=80)
    scorer = LabelScorer()
    epoch_start_time =time.time()
    for idx, batch in enumerate(dev_loader):
        loss, prediction = model(batch, config['use_gpu'])

        prediction = prediction.cpu().clone().numpy()
        labels = list(batch['labels'].numpy())
        scorer.update(prediction, labels)

        dev_loss += loss.item()
        # logging("Avg. batch test loss: {}".format(dev_loss / (idx + 1)))
        # logging("Testing ----> Epoch: {},  Batch: {}/{},  This epoch takes {}/{} s,  Avg.batch test loss: {}".format(epoch, idx+1,len(dev_loader), '{:.2f}'.format(time.time()-epoch_start_time), '{:.2f}'.format((time.time()-epoch_start_time)/(idx+1)*len(dev_loader)), '{:.5f}'.format(dev_loss / (idx + 1))))

        #dev_loader_tqdm.set_description(description)
    dev_loss /= len(dev_loader)
    avg_accu, precision, recall, f1 =  scorer.get_avg_scores()
    avg_accu = '{:.2f}'.format(avg_accu)
    precision = '{:.2f}'.format(precision)
    recall = '{:.2f}'.format(recall)
    f1 = '{:.2f}'.format(f1)
    dev_loss = '{:.4f}'.format(dev_loss)
    return dev_loss, avg_accu,precision, recall, f1




def test(file, model, test_loader, checkpoint_dir, config):

    test_loss, avg_accu, precision, recall, f1 = dev_epoch(
        epoch=0,
        config=config,
        model=model,
        dev_loader=test_loader
    )
    logging("* Result on test set: {}".format(file))
    logging("-> loss={}, Accuracy={}, Precision={}, Recall={}, F1={}".format(test_loss, avg_accu, precision, recall, f1))




def main():
    model = BertClassifier(config)
    if config["load_model"] == 'yes':
        logging('using our pretrained model')
        checkpoint = torch.load(config["load_checkpoint"])
        # print(list(model.bert_layer.named_parameters())[4])
        model.load_state_dict(checkpoint)
        # print(list(model.bert_layer.named_parameters())[4])

    if config['use_gpu']:
        if len(config['gpus'].split(',')) >= 2:
            model = nn.DataParallel(model)
        model = model.cuda()


    files = ['data/indexed/LCQMC/indexed_dev.json', 'data/indexed/LCQMC/indexed_test.json', 'data/indexed/BQ/indexed_dev.json', 'data/indexed/BQ/indexed_test.json', 'data/indexed/AFQMC/indexed_dev.json']

    for file in files:
        logging("=" * 20+"Preparing test data..."+"=" * 20)
        with open(file, "r", encoding='utf-8') as f:  # test_dir
            indexed_test_data = json.loads(f.read())
        test_data = SentencePairDataset(indexed_test_data, config['max_len'], padding_idx=0)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=config['batch_size'])

        test(file, model, test_loader, config["checkpoint_dir"], config)







if __name__ == "__main__":
    main()
