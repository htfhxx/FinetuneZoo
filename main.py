import argparse
import json
import os
import time
import torch
from torch.nn import functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.Criterion import LabelScorer
from utils.Data import LcqmcDataset
from model.bert_model import BertClassifier
from transformers import AdamW
from utils import util

def set_up_logging(config):
    if not os.path.exists(config['log_path']):
        os.mkdir(config['log_path'])
    tmp_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime()).replace(':','_')
    checkpoint_dir = config["checkpoint_dir"] + tmp_time + '/'
    log_path = config["log_path"] + tmp_time + '/'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    logging = util.logging(log_path + 'log.txt')  # 往这个文件里写记录
    logging("__________logging the args_________")
    for k, v in config.items():
        logging("%s:\t%s" % (str(k), str(v)))
    logging("\n")
    return logging,log_path,checkpoint_dir

# prepration for config
parser = argparse.ArgumentParser()
parser.add_argument("--config", default="utils/config/train_bert.config")
parser.add_argument("--pretrain_model", default="checkpoints/bert-base-chinese/")
parser.add_argument("--use_our_pretrain", default="no")  # yes
parser.add_argument("--our_pretrain_model", default="checkpoints/best_BERT_epoch.tar")
parser.add_argument("--mode", default="test")  # train test
parser.add_argument("--max_len", default=512)
parser.add_argument("--batch_size", default=16)
parser.add_argument("--epoches", default=20)
parser.add_argument("--learning_rate", default=0.00002)
parser.add_argument("--max_patience_epoches", default=5)
parser.add_argument("--log_path", default='log/')
parser.add_argument("--checkpoint_dir", default='checkpoints/')
parser.add_argument("--use_gpu", default='yes')  # no
parser.add_argument("--gpus", default='0,1')
args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = json.loads(config_file.read())

config["config"] = args.config
config["pretrain_model"] = args.pretrain_model
config["use_our_pretrain"] = args.use_our_pretrain
config["our_pretrain_model"] = args.our_pretrain_model
config["mode"] = args.mode
config["max_len"] = args.max_len
config["batch_size"] = int(args.batch_size)
config["epoches"] = args.epoches
config["learning_rate"] = args.learning_rate
config["max_patience_epoches"] = args.max_patience_epoches
config["log_path"] = args.log_path
config["checkpoint_dir"] = args.checkpoint_dir

logging,log_path, checkpoint_dir = set_up_logging(config)

# config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"


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




def train_epoch(epoch, config, model, optimizer, train_loader):
    model.train()

    train_loss = 0.0

    scorer = LabelScorer()
    epoch_start_time = time.time()
    for idx, batch in enumerate(train_loader):
        # if idx>10:
        #     break

        text = batch["text"]
        mask = batch["mask"]
        labels = batch["label"]
        if config['use_gpu']:
            text = text.cuda()
            mask = mask.cuda()
            labels = labels.cuda()

        labels = torch.squeeze(labels, dim=-1)

        optimizer.zero_grad()

        logits = model(text, mask)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()


        prediction = logits.argmax(-1)
        prediction = prediction.cpu().clone().numpy()
        labels = labels.cpu().clone().numpy()
        scorer.update(prediction, labels)

        train_loss += loss.item()
        logging("Training ----> Epoch: {},  Batch: {}/{},  This epoch takes {}/{}s,  Avg.batch train loss: {}".format(epoch, idx+1,len(train_loader), '{:.2f}'.format(time.time()-epoch_start_time),'{:.2f}'.format((time.time()-epoch_start_time)/(idx+1)*len(train_loader)), '{:.5f}'.format(train_loss / (idx + 1))))
        # train_loader_tqdm.set_description(description)
    train_loss /= len(train_loader)
    avg_accu = scorer.get_avg_scores()
    avg_accu = '{:.4f}'.format(avg_accu)
    train_loss = '{:.4f}'.format(train_loss)
    return train_loss, avg_accu


def dev_epoch(epoch, config, model, dev_loader):
    model.eval()

    dev_loss = 0.0
    #dev_loader_tqdm = tqdm(dev_loader, ncols=80)
    scorer = LabelScorer()
    epoch_start_time =time.time()
    for idx, batch in enumerate(dev_loader):
        # if idx>10:
        #     break

        text = batch["text"]
        mask = batch["mask"]
        labels = batch["label"]
        if config['use_gpu']:
            text = text.cuda()
            mask = mask.cuda()
            labels = labels.cuda()

        labels = torch.squeeze(labels, dim=-1)

        logits = model(text, mask)

        loss = F.cross_entropy(logits, labels)



        prediction = logits.argmax(-1)
        prediction = prediction.cpu().clone().numpy()
        labels = labels.cpu().clone().numpy()
        scorer.update(prediction, labels)

        dev_loss += loss.item()
        # logging("Avg. batch test loss: {}".format(dev_loss / (idx + 1)))
        logging("Testing ----> Epoch: {},  Batch: {}/{},  This epoch takes {}/{} s,  Avg.batch test loss: {}".format(epoch, idx+1,len(dev_loader), '{:.2f}'.format(time.time()-epoch_start_time), '{:.2f}'.format((time.time()-epoch_start_time)/(idx+1)*len(dev_loader)), '{:.5f}'.format(dev_loss / (idx + 1))))

        #dev_loader_tqdm.set_description(description)
    dev_loss /= len(dev_loader)
    avg_accu = scorer.get_avg_scores()
    avg_accu = '{:.4f}'.format(avg_accu)
    dev_loss = '{:.4f}'.format(dev_loss)
    return dev_loss, avg_accu


def train(model, train_loader, dev_loader, config):


    epoches = config["epoches"]
    learning_rate = config["learning_rate"]
    max_patience_epoches = config["max_patience_epoches"]
    # criterion = Criterion()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = AdamW(model.parameters(), lr=learning_rate)


    patience_count, best_accu = 0, 0
    start_epoch = 0
    for epoch in range(start_epoch, epoches):
        train_loss,  avg_accu = train_epoch(
            epoch = epoch,
            config=config,
            model=model,
            optimizer=optimizer,
            train_loader=train_loader
        )
        logging("* Train epoch {}".format(epoch + 1))
        logging("-> loss={}, Accuracy={}".format(train_loss, avg_accu))

        with open(log_path + 'train_result.csv', 'a') as file:
            file.write("{},{},{}\n".format(epoch, train_loss,avg_accu))

        dev_loss, dev_accu  = dev_epoch(
            epoch = epoch,
            config=config,
            model=model,
            dev_loader=dev_loader
        )
        logging("* Dev epoch {}".format(epoch + 1))
        logging("-> loss={}, Accuracy={}".format(dev_loss,dev_accu))
        with open(log_path + 'valid_result.csv', 'a') as file:
            file.write("{},{},{}\n".format(epoch, dev_loss,dev_accu))

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, os.path.join(checkpoint_dir, 'bert_model_epoch'+str(epoch)+'.tar'))




        if float(dev_accu) >  float(best_accu):
            patience_count = 0
            best_accu = dev_accu

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, os.path.join(checkpoint_dir, 'best_bert_model.tar'))
            logging('new epoch saved as the best model {}'.format(epoch))

            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, os.path.join(config["checkpoint_dir"], 'bert_model_epoch' + '.tar'))

        else:
            patience_count += 1
            if patience_count >= max_patience_epoches:
                logging("Early Stopping at epoch {}".format(epoch + 1))
                break




def test(model, test_loader, checkpoint_dir, config):
    checkpoint = torch.load(os.path.join(checkpoint_dir, "best_bert_model.tar"))
    model.load_state_dict(checkpoint["model"])

    test_loss,  avg_accu = dev_epoch(
        epoch=0,
        config=config,
        model=model,
        dev_loader=test_loader
    )
    logging("* Result on test set")
    logging("-> loss={}, Accuracy={}".format(test_loss, avg_accu))




def main():


    model = BertClassifier(config, transformer_width=768 , num_labels=2)
    if config["use_our_pretrain"] == 'yes':
        logging('using our pretrained model')
        checkpoint = torch.load(config["our_pretrain_model"])
        # print(list(model.bert_layer.named_parameters())[4])
        model.bert_layer.load_state_dict(checkpoint['model'])
        # print(list(model.bert_layer.named_parameters())[4])

    if config['use_gpu']:
        if len(config['gpus'].split(',')) >= 2:
            model = nn.DataParallel(model)
        model = model.cuda()

    # prepration for data
    if config['mode']=='train':
        logging("=" * 20+"Preparing training data..."+"=" * 20)
        with open(config["train_dir"], "r") as f:
            indexed_train_data = json.loads(f.read())

        # indexed_train_data['text'] = indexed_train_data['text'][:100]
        # indexed_train_data['label'] = indexed_train_data['label'][:100]

        train_data = LcqmcDataset(indexed_train_data, config['max_len'], padding_idx=0)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=config['batch_size'])

        logging("=" * 20+"Preparing dev data..."+"=" * 20)
        with open(config["dev_dir"], "r") as f:
            indexed_dev_data = json.loads(f.read())
        dev_data = LcqmcDataset(indexed_dev_data, config['max_len'], padding_idx=0)
        dev_loader = DataLoader(dev_data, shuffle=True, batch_size=config['batch_size'])

        train(model,  train_loader, dev_loader, config)
    elif config['mode']=='test':
        logging("=" * 20+"Preparing test data..."+"=" * 20)
        with open(config["dev_dir"], "r") as f:  # test_dir
            indexed_test_data = json.loads(f.read())
        test_data = LcqmcDataset(indexed_test_data, config['max_len'], padding_idx=0)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=config['batch_size'])

        test(model, test_loader, config["checkpoint_dir"], config)
    else:
        logging('wrong mode!')
        exit()




if __name__ == "__main__":
    main()
