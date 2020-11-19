
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















