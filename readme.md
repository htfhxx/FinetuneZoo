
# Introduction

BERT Fine-Tune 语义文本相似度任务（句对分类 - LCQMC）

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
python preprocess.py
python preprocess.py --data_dir data/original/LCQMC/  --save_dir data/indexed/LCQMC/  --vocab_path checkpoints/bert-base-chinese/vocab.txt --max_seq_length 128
python preprocess.py --data_dir data/original/BQ/  --save_dir data/indexed/BQ/  --vocab_path checkpoints/bert-base-chinese/vocab.txt --max_seq_length 128
python preprocess.py --data_dir data/original/AFQMC/  --save_dir data/indexed/AFQMC/  --vocab_path checkpoints/bert-base-chinese/vocab.txt --max_seq_length 128

```

* 训练
```
python main.py --mode train --batch_size 64 --save_dir_name train_lcqmc_seed1 --seed 1 

```
* 测试
```
python main.py --mode test --load_model yes --load_checkpoint checkpoints/train_lcqmc_seed1/best_checkpoints.model --data_test_path data/indexed/LCQMC/indexed_test.json --save_dir_name test_lcqmc_seed1

```
* 测试大礼包
```
python test_pkg.py --mode test --load_model yes --load_checkpoint checkpoints/train_lcqmc_seed1/best_checkpoints.model --save_dir_name test_lcqmc_seed1 

```
* 临时
```

```



# 下游任务效果

## 使用LCQMC训练


### 汇总结果：多次随机种子最大值

| Model  | 训练集 | LCQMC-Test | BQ-Valid | BQ-Test | AFQMC-valid  |
| :----: | :----: | :----: | :----: | :----: | :----: |
|||||||
| BERT-base-Chinese | LCQMC  | 87.51(88.28)| 62.35(41.93)| 60.96(38.73)| 69.79(39.18)|
| BERT-wwm-ext      | LCQMC  | | | | |
|||||||
| BERT-RecAdam      | LCQMC  | | | | |
| SMART-BERT        | LCQMC  | | | | |
|||||||



### 所有实验


使用LCQMC作为训练集

| Model | seed | LCQMC-Valid  | LCQMC-Test | BQ-Valid | BQ-Test | AFQMC-valid  |
| :----:| :----: | :----: | :----: | :----: | :----: | :----: |
|||||||
| BERT-base-Chinese | seed=1  | 89.47(89.70) | 86.71(87.72)| 59.67(34.37)| 58.42(31.39)| 69.79(39.18)|
| BERT-base-Chinese | seed=2  | 88.96(89.34) | 85.69(86.98)| 60.19(36.15)| 59.39(34.15)| 69.25(41.36)|
| BERT-base-Chinese | seed=3  | 89.16(89.40) | 86.47(87.54)| 62.35(41.93)| 60.96(38.73)| 68.88(39.64)|
| BERT-base-Chinese | seed=42 | 89.43(89.50) | 87.51(88.28)| 58.63(31.47)| 58.34(30.73)| 69.65(39.58)|
|||||||
| BERT-base-Chinese | seed=   | () | ()| ()| ()| ()|







