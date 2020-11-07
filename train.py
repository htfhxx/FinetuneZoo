import argparse
import json
import os
import torch
from torch.nn import functional as F
# from tqdm import tqdm
from torch.utils.data import DataLoader
from utils.Criterion import LabelScorer
from utils.Data import LcqmcDataset
from model.bert_model import BertClassifier
import time
from transformers import AdamW







def train_epoch(epoch, config, device, model, optimizer, train_loader):
    model.train()

    train_loss = 0.0
    # train_loader_tqdm = tqdm(train_loader, ncols=80)
    scorer = LabelScorer()
    epoch_start_time = time.time()
    for idx, batch in enumerate(train_loader):
        # if idx>10:
        #     break

        text = batch["text"].to(device)
        labels = torch.squeeze(batch["label"].to(device), dim=-1)

        optimizer.zero_grad()

        logits = model(text)

        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()


        prediction = logits.argmax(-1)
        prediction = prediction.cpu().clone().numpy()
        labels = labels.cpu().clone().numpy()
        scorer.update(prediction, labels)

        train_loss += loss.item()
        print("Training ----> Epoch: {},  Batch: {}/{},  This epoch takes {}/{}s,  Avg.batch train loss: {}".format(epoch, idx+1,len(train_loader), '{:.2f}'.format(time.time()-epoch_start_time),'{:.2f}'.format((time.time()-epoch_start_time)/(idx+1)*len(train_loader)), '{:.5f}'.format(train_loss / (idx + 1))))
        # train_loader_tqdm.set_description(description)
    train_loss /= len(train_loader)
    avg_accu = scorer.get_avg_scores()
    avg_accu = '{:.4f}'.format(avg_accu)
    train_loss = '{:.4f}'.format(train_loss)
    return train_loss, avg_accu


def dev_epoch(epoch, config, device, model, dev_loader):
    model.eval()

    dev_loss = 0.0
    #dev_loader_tqdm = tqdm(dev_loader, ncols=80)
    scorer = LabelScorer()
    epoch_start_time =time.time()
    for idx, batch in enumerate(dev_loader):
        # if idx>10:
        #     break

        text = batch["text"].to(device)
        labels = torch.squeeze(batch["label"].to(device), dim=-1)

        logits = model(text)

        loss = F.cross_entropy(logits, labels)



        prediction = logits.argmax(-1)
        prediction = prediction.cpu().clone().numpy()
        labels = labels.cpu().clone().numpy()
        scorer.update(prediction, labels)

        dev_loss += loss.item()
        # print("Avg. batch test loss: {}".format(dev_loss / (idx + 1)))
        print("Testing ----> Epoch: {},  Batch: {}/{},  This epoch takes {}/{} s,  Avg.batch test loss: {}".format(epoch, idx+1,len(dev_loader), '{:.2f}'.format(time.time()-epoch_start_time), '{:.2f}'.format((time.time()-epoch_start_time)/(idx+1)*len(dev_loader)), '{:.5f}'.format(dev_loss / (idx + 1))))

        #dev_loader_tqdm.set_description(description)
    dev_loss /= len(dev_loader)
    avg_accu = scorer.get_avg_scores()
    avg_accu = '{:.4f}'.format(avg_accu)
    dev_loss = '{:.4f}'.format(dev_loss)
    return dev_loss, avg_accu


def train(model, train_loader, dev_loader, config):

    checkpoint_dir = config["checkpoint_dir"]
    epoches = config["epoches"]
    learning_rate = config["learning_rate"]
    max_patience_epoches = config["max_patience_epoches"]
    # criterion = Criterion()
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = AdamW(model.parameters(), lr=learning_rate)


    patience_count, best_accu = 0, 0
    start_epoch = 0
    for epoch in range(start_epoch, epoches):
        model.train()
        train_loss,  avg_accu = train_epoch(
            epoch = epoch,
            config=config,
            device=config["device"],
            model=model,
            optimizer=optimizer,
            train_loader=train_loader
        )
        print("* Train epoch {}".format(epoch + 1))
        print("-> loss={}, Accuracy={}".format(train_loss, avg_accu))

        with open('log/train_result.csv', 'a') as file:
            file.write("{},{},{}\n".format(epoch, train_loss,avg_accu))

        model.eval()
        dev_loss, dev_accu  = dev_epoch(
            epoch = epoch,
            config=config,
            device=config["device"],
            model=model,
            dev_loader=dev_loader
        )
        print("* Dev epoch {}".format(epoch + 1))
        print("-> loss={}, Accuracy={}".format(dev_loss,dev_accu ))
        with open('log/valid_result.csv', 'a') as file:
            file.write("{},{},{}\n".format(epoch, dev_loss,avg_accu))

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
            print('new epoch saved as the best model', epoch)

        else:
            patience_count += 1
            if patience_count >= max_patience_epoches:
                print("Early Stopping at epoch {}".format(epoch + 1))
                break




def test(model, test_loader, checkpoint_dir, config):
    checkpoint = torch.load(os.path.join(checkpoint_dir, "best_bert_model.tar"))
    model.load_state_dict(checkpoint["model"])

    test_loss,  avg_accu = dev_epoch(
        epoch=0,
        config=config,
        device=config["device"],
        model=model,
        dev_loader=test_loader
    )
    print("* Result on test set")
    print("-> loss={}, Accuracy={}".format(test_loss, avg_accu))

def main():

    '''prepration for config'''
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="utils/config/train_bert.config")
    parser.add_argument("--pretrain_model", default="checkpoints/roberta_wwm_ext_pytorch/")
    parser.add_argument("--mode", default="train")
    parser.add_argument("--max_len", default=512)
    parser.add_argument("--batch_size", default=3)
    parser.add_argument("--epoches", default=30)
    parser.add_argument("--learning_rate", default=0.00002)
    parser.add_argument("--max_patience_epoches", default=10)
    args = parser.parse_args()

    with open(args.config, "r") as config_file:
        config = json.loads(config_file.read())

    config["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("=" * 20, "Running on device: {}".format(config["device"]), "=" * 20)

    config["max_len"] = args.max_len
    config["batch_size"]  = args.batch_size
    config["epoches"] = args.epoches
    config["learning_rate"] = args.learning_rate
    config["max_patience_epoches"] = args.max_patience_epoches
    config["pretrain_model"] = args.pretrain_model



    model = BertClassifier(config, transformer_width=768 , num_labels=2).to(config["device"])

    # prepration for data
    if args.mode is 'train':
        print("=" * 20, "Preparing training data...", "=" * 20)
        with open(config["train_dir"], "r") as f:
            indexed_train_data = json.loads(f.read())
        train_data = LcqmcDataset(indexed_train_data, config['max_len'], padding_idx=0)
        train_loader = DataLoader(train_data, shuffle=True, batch_size=config['batch_size'])

        print("=" * 20, "Preparing dev data...", "=" * 20)
        with open(config["dev_dir"], "r") as f:
            indexed_dev_data = json.loads(f.read())
        dev_data = LcqmcDataset(indexed_dev_data, config['max_len'], padding_idx=0)
        dev_loader = DataLoader(dev_data, shuffle=True, batch_size=config['batch_size'])

        train(model,  train_loader, dev_loader, config)
    elif args.mode is 'test':
        print("=" * 20, "Preparing test data...", "=" * 20)
        with open(config["test_dir"], "r") as f:
            indexed_test_data = json.loads(f.read())
        test_data = LcqmcDataset(indexed_test_data, config['max_len'], padding_idx=0)
        test_loader = DataLoader(test_data, shuffle=True, batch_size=config['batch_size'])

        test(model, test_loader, config["checkpoint_dir"], config)





if __name__ == "__main__":
    main()
