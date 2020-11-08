
### Introduction
BERT fine-tune LCQMC分类任务

###  requirements
python3.7
torch == 1.5.1
tqdm == 4.46.0
scikit-learn == 0.23.1
numpy == 1.19.4
transformers == 3.4.0

###  Train & Test

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





FROM docker.yard.oa.com:14917/yard/env:cuda10.0-py36-env-2.1

RUN pip install torch==1.5.1
RUN pip install tqdm == 4.46.0
RUN pip install numpy == 1.19.4
RUN pip install transformers == 3.4.0
RUN pip install tensorboardX
RUN pip install scikit-learn == 0.23.1
RUN pip install scipy
RUN pip install rouge
RUN pip install nltk==3.5


RUN echo Finished installing pip

RUN python -m nltk.downloader 'punkt'
RUN python -m nltk.downloader 'wordnet'

RUN echo Finished install nltk punkt, wordnet

CMD ["/bin/bash"]
