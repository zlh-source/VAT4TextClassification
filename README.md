# VAT4TextClassification-基于VAT的文本分类
## 虚拟对抗的论文
[Virtual Adversarial Training: A Regularization Method for Supervised and Semi-Supervised Learning](https://arxiv.org/abs/1704.03976)
## 模型结构
BERT+Linear
## 数据集
情感分类数据集[sentiment.zip](/sentiment.zip)
## 消融实验
![result](/result.png)
## 注意事项
1.BERT的结构是 Trans(word_embed + position_embed + token_type_embed) , Trans()代表N层transformer结构，添加VAT产生的噪音noise_embed之后，BERT的结构成为了 Trans(word_embed +   position_embed + token_type_embed + noise_embed)。因此当使用[Transformers库](https://github.com/huggingface/transformers)调用BERT时，需要修改BERT的源码[modeling_bert.py](modeling_bert.py)来实现对底层编码添加噪音。

2.原版是[苏剑林的keras代码](https://github.com/bojone/bert4keras/blob/master/examples/task_sentiment_virtual_adversarial_training.py)，我用torch进行了复现
