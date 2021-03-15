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
parser.add_argument("--data_train_path", default="data/indexed/LCQMC/indexed_train.json")
parser.add_argument("--data_dev_path", default="data/indexed/LCQMC/indexed_dev.json")
parser.add_argument("--data_test_path", default="data/indexed/LCQMC/indexed_test.json")
parser.add_argument("--pretrain_model", default="checkpoints/bert-base-chinese/")
parser.add_argument("--load_model", default="no")  # yes or no
parser.add_argument("--load_checkpoint", default="checkpoints/best_checkpoints.model")
parser.add_argument("--mode", default="train")  # train or test or reference
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=16) # 16
parser.add_argument("--epoches", type=int, default=3)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval_steps", type=int, default=1000)
parser.add_argument("--learning_rate", type=float, default=0.00002)
parser.add_argument("--use_gpu", default='yes')  # no
parser.add_argument("--gpus", default='0')
parser.add_argument("--optimizer", default='AdamW')  # RecAdam
args = parser.parse_args()

with open(args.config, "r") as config_file:
    config = json.loads(config_file.read())

config["config"] = args.config
config["save_dir_name"] = args.save_dir_name
config["data_train_path"] = args.data_train_path
config["data_dev_path"] = args.data_dev_path
config["data_test_path"] = args.data_test_path
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
config["optimizer"] = args.optimizer

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
    if config['mode'] == 'reference':
        total_sentences = []
        total_labels = []
        total_prediction = []
    for idx, batch in enumerate(dev_loader):
        # if idx > 2:
        #     break

        loss, prediction = model(batch, config['use_gpu'])

        prediction = prediction.cpu().clone().numpy()
        labels = list(batch['labels'].clone().numpy())
        scorer.update(prediction, labels)

        dev_loss += loss.item()
        logging("Testing ----> Epoch: {},  Batch: {}/{},  This epoch takes {}/{} s,  Avg.batch test loss: {}".format(epoch, idx+1,len(dev_loader), '{:.2f}'.format(time.time()-epoch_start_time), '{:.2f}'.format((time.time()-epoch_start_time)/(idx+1)*len(dev_loader)), '{:.5f}'.format(dev_loss / (idx + 1))))

        if config['mode'] == 'reference':
            total_sentences.extend(batch['sentences'])
            total_labels.extend(labels)
            total_prediction.extend(prediction)


    dev_loss /= len(dev_loader)
    avg_accu, precision, recall, f1 =  scorer.get_avg_scores()
    avg_accu = '{:.2f}'.format(avg_accu)
    precision = '{:.2f}'.format(precision)
    recall = '{:.2f}'.format(recall)
    f1 = '{:.2f}'.format(f1)
    dev_loss = '{:.4f}'.format(dev_loss)

    if config['mode'] == 'reference':
        assert len(total_sentences)==len(total_prediction)==len(total_labels)
        with open(log_path + '/reference_result.csv','w',encoding='utf-8') as f:
            f.write('sentence_pair \t labels \t prediction \n')
            for idx in range(len(total_sentences)):
                f.write(total_sentences[idx])
                f.write('\t')
                f.write(str(total_labels[idx]))
                f.write('\t')
                f.write(str(total_prediction[idx]))
                f.write('\n')

    return dev_loss, avg_accu,precision, recall, f1


def train(model, train_loader, dev_loader, config):

    epoches = config["epoches"]
    learning_rate = config["learning_rate"]
    if config['optimizer'] == 'Adam':
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    else:
        logging('wrong optimizer.')
        exit()


    best_accu = 0
    step_cnt = 0
    start_time = time.time()

    for epoch in range(epoches):
        train_loss = 0.0
        train_scorer = LabelScorer()
        model.train()
        for idx, batch in enumerate(train_loader):
            # if idx > 50:
            #     break
            optimizer.zero_grad()

            loss, prediction = model(batch, config['use_gpu'])
            loss.backward()
            optimizer.step()

            prediction = prediction.cpu().clone().numpy()
            labels = batch['labels'].numpy()
            train_scorer.update(prediction, labels)

            train_loss += loss.item()
            logging(
                "Training ----> Epoch: {}/{},  Batch: {}/{}*3,  training takes {}/{}s,  Avg.batch train loss: {}".format(
                    epoch, epoches, idx + 1, len(train_loader), '{:.2f}'.format(time.time() - start_time),
                    '{:.2f}'.format((time.time() - start_time) / (step_cnt+1) * len(train_loader) *3),
                    '{:.5f}'.format(train_loss / (idx + 1))))

            step_cnt +=1

            if step_cnt % config['eval_steps'] == 0:
                model.eval()
                dev_loss, dev_accu, precision, recall, f1 = dev_epoch(
                    epoch=epoch,
                    config=config,
                    model=model,
                    dev_loader=dev_loader
                )
                logging("* Dev epoch {}".format(epoch + 1))
                logging("-> loss={}, Accuracy={}, Precision={}, Recall={}, F1={}".format(dev_loss, dev_accu, precision, recall, f1))
                with open(log_path + 'valid_result.csv', 'a') as file:
                    file.write("{},{},{},{},{},{}\n".format(epoch, dev_loss, dev_accu, precision, recall, f1))

                torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'checkpint_steps_' + str(step_cnt) + '.model'))

                if float(dev_accu) > float(best_accu):
                    best_accu = dev_accu
                    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_checkpoints.model'))
                    logging('new epoch saved as the best model {}'.format(epoch))

                model.train()

        train_loss /= len(train_loader)
        avg_accu, precision, recall, f1 = train_scorer.get_avg_scores()
        avg_accu = '{:.2f}'.format(avg_accu)
        precision = '{:.2f}'.format(precision)
        recall = '{:.2f}'.format(recall)
        f1 = '{:.2f}'.format(f1)
        train_loss = '{:.4f}'.format(train_loss)


        logging("* Train steps {} finished.".format(step_cnt))
        logging("-> loss={}, Accuracy={}, Precision={}, Recall={}, F1={}".format(train_loss, avg_accu, precision, recall, f1))

        with open(log_path + 'train_result.csv', 'a') as file:
            file.write("{},{},{},{},{},{}\n".format(epoch, train_loss, avg_accu, precision, recall,f1))

    model.eval()
    dev_loss, dev_accu, precision, recall, f1 = dev_epoch(
        epoch=epoch,
        config=config,
        model=model,
        dev_loader=dev_loader
    )
    logging("* Dev epoch {}".format(epoch + 1))
    logging("-> loss={}, Accuracy={}, Precision={}, Recall={}, F1={}".format(dev_loss, dev_accu, precision, recall, f1))
    with open(log_path + 'valid_result.csv', 'a') as file:
        file.write("{},{},{},{},{},{}\n".format(epoch, dev_loss, dev_accu, precision, recall,f1))

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'checkpint_steps_' + str(step_cnt) + '.model'))

    if float(dev_accu) > float(best_accu):
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_checkpoints.model'))
        logging('new epoch saved as the best model {}'.format(epoch))



def test(model, test_loader, config):

    test_loss, avg_accu, precision, recall, f1 = dev_epoch(
        epoch=0,
        config=config,
        model=model,
        dev_loader=test_loader
    )
    logging("* Result on test set")
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

    # prepration for data
    if config['mode']=='train':
        logging("=" * 20+"Preparing training data..."+"=" * 20)
        with open(config["data_train_path"], "r", encoding='utf-8') as f:
            indexed_train_data = json.loads(f.read())
        train_data = SentencePairDataset(indexed_train_data, config['max_len'], padding_idx=0)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=config['batch_size'])

        logging("=" * 20+"Preparing dev data..."+"=" * 20)
        with open(config["data_dev_path"], "r", encoding='utf-8') as f:
            indexed_dev_data = json.loads(f.read())
        dev_data = SentencePairDataset(indexed_dev_data, config['max_len'], padding_idx=0)
        dev_loader = DataLoader(dev_data, shuffle=True, batch_size=config['batch_size'])

        train(model,  train_loader, dev_loader, config)


    elif config['mode']=='test' or config['mode']=='reference':
        logging("=" * 20+"Preparing test data..."+"=" * 20)
        with open(config["data_test_path"], "r", encoding='utf-8') as f:  # test_dir
            indexed_test_data = json.loads(f.read())
        test_data = SentencePairDataset(indexed_test_data, config['max_len'], padding_idx=0)
        test_loader = DataLoader(test_data, shuffle=False, batch_size=config['batch_size'])

        test(model, test_loader,  config)


    else:
        logging('wrong mode!')
        exit()




if __name__ == "__main__":
    main()
