
# Introduction

BERT fine-tune LCQMC分类任务

##  requirements
```angular2
python3.7
torch == 1.5.1
tqdm == 4.46.0
scikit-learn == 0.23.1
numpy == 1.19.4
transformers == 3.4.0
nltk==3.5
```

## 文件结构

```
$ tree
.
|-- checkpoints                           训练过程中保存的模型 or PLM-baseline 
|   |-- 2020-11-19-15_59_27                     训练过程中保存的模型 
|   |-- 2020-11-19-16_55_27
|   |   |-- bert_model_epoch0.tar
|   |   |-- bert_model_epoch1.tar
|   |   `-- best_bert_model.tar
|   `-- roberta_wwm_ext_pytorch                  PLM-baseline 
|       |-- config.json
|       |-- pytorch_model.bin
|       `-- vocab.txt
|-- data
|   `-- LCQMC
|       |-- dev.txt
|       |-- indexed_dev_bert.json
|       |-- indexed_test_bert.json
|       |-- indexed_train_bert.json
|       |-- test.txt
|       `-- train.txt
|-- log                                     日志 打印一份 保存一份
|   |-- 2020-11-19-15_59_27
|   |   `-- log.txt
|   `-- 2020-11-19-16_55_27
|       |-- log.txt
|       |-- train_result.csv
|       `-- valid_result.csv
|-- model                     
|   `-- bert_model.py
|-- preprocess_data.py                  数据预处理
|-- readme.md      
|-- train.py                           主训练&测试代码
`-- utils
    |-- Criterion.py
    |-- Data.py
    |-- config
    |   `-- train_bert.config
    `-- util.py

```


##  Fine-tune

* 预处理数据
```
python3 preprocess_data.py
```
* 训练
```
python3 train_bert.py --mode train
```
```
nohup python3 -u train_bert.py  > myout.file 2>&1 &
tail -f myout.file

cat myout.file | grep 'loss='
```
* 测试
```
python3 train_bert.py --mode test
```





# 下游任务效果

## LCQMC

| Model | Valid  | Test | Path |
| :----:| :----: | :----: | ------ |
| BERT |    |   |   |
| Roberta_wwm_ext | 88.91 | 86.54 |  |
| Roberta_wwm_ext（batch_size++） | 89.31 | 86.68 |  |





## 简洁的日志Demo
```
train_dir:	data/LCQMC/indexed_train_bert.json
dev_dir:	data/LCQMC/indexed_dev_bert.json
test_dir:	data/LCQMC/indexed_test_bert.json
config:	utils/config/train_bert.config
pretrain_model:	checkpoints/roberta_wwm_ext_pytorch/
mode:	train
max_len:	512
batch_size:	64
epoches:	5
learning_rate:	2e-05
max_patience_epoches:	2
log_path:	log/
checkpoint_dir:	checkpoints/


====================Running on device: cuda:0====================
====================Preparing training data...====================
real max_len: 164
====================Preparing dev data...====================
real max_len: 76
Training ----> Epoch: 0,  Batch: 1/3731,  This epoch takes 1.10/4113.65s,  Avg.batch train loss: 0.68036
Training ----> Epoch: 0,  Batch: 2/3731,  This epoch takes 2.16/4026.98s,  Avg.batch train loss: 0.73325
Training ----> Epoch: 0,  Batch: 3/3731,  This epoch takes 3.21/3995.38s,  Avg.batch train loss: 0.73004
Training ----> Epoch: 0,  Batch: 4/3731,  This epoch takes 4.27/3980.77s,  Avg.batch train loss: 0.74274
Training ----> Epoch: 0,  Batch: 5/3731,  This epoch takes 5.32/3972.51s,  Avg.batch train loss: 0.73818
Training ----> Epoch: 0,  Batch: 6/3731,  This epoch takes 6.40/3982.84s,  Avg.batch train loss: 0.73410
Training ----> Epoch: 0,  Batch: 7/3731,  This epoch takes 7.48/3988.75s,  Avg.batch train loss: 0.72727
Training ----> Epoch: 0,  Batch: 8/3731,  This epoch takes 8.56/3993.80s,  Avg.batch train loss: 0.72373
Training ----> Epoch: 0,  Batch: 9/3731,  This epoch takes 9.64/3994.87s,  Avg.batch train loss: 0.71619
Training ----> Epoch: 0,  Batch: 10/3731,  This epoch takes 10.71/3994.37s,  Avg.batch train loss: 0.71253
Training ----> Epoch: 0,  Batch: 11/3731,  This epoch takes 11.77/3993.47s,  Avg.batch train loss: 0.70371
Training ----> Epoch: 0,  Batch: 12/3731,  This epoch takes 12.83/3988.44s,  Avg.batch train loss: 0.70574
Training ----> Epoch: 0,  Batch: 13/3731,  This epoch takes 13.89/3985.10s,  Avg.batch train loss: 0.70070
Training ----> Epoch: 0,  Batch: 14/3731,  This epoch takes 14.94/3981.79s,  Avg.batch train loss: 0.70130
Training ----> Epoch: 0,  Batch: 15/3731,  This epoch takes 16.00/3978.52s,  Avg.batch train loss: 0.69708
Training ----> Epoch: 0,  Batch: 16/3731,  This epoch takes 17.06/3978.72s,  Avg.batch train loss: 0.69614
Training ----> Epoch: 0,  Batch: 17/3731,  This epoch takes 18.12/3976.23s,  Avg.batch train loss: 0.69470
...
```














