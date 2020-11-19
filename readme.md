
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
|-- log
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

| Model | Valid  | Test |
| :----:| :----: | :----: |
| BERT |    |   |
| Roberta_wwm_ext | 88.91 | 86.54 |















