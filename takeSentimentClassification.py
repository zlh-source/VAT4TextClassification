#! -*- coding:utf-8 -*-
import json
import numpy as np
from bert4keras.backend import keras
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.utils import to_categorical
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.functional as F
from transformers.modeling_bert import BertModel,BertPreTrainedModel,BertConfig
from transformers import WEIGHTS_NAME,AdamW, get_linear_schedule_with_warmup
from vat import virtual_adversarial_training,KLD,set_seed
from copy import deepcopy
import torch.optim as optim

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "5" #注意修改
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device=torch.device("cpu")

# 配置信息
seed=1
steps_per_epoch=30
alpha=1.0
warmup = 0.0
weight_decay = 0
lr=2e-5
VAT_lr=1e-5
batch_size=32
num_epoch=100
num_classes = 2
maxlen = 128
train_frac = 0.01  # 标注数据的比例
use_vat = True  # 可以比较True/False的效果
output_path="./dataset/sentiment/output/07/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# BERT base
BertPath={
'bert_vocab_path': './base-chinese/bert-base-chinese-vocab.txt',
'bert_config_path': './base-chinese/config.json',
'bert_model_path': './base-chinese/pytorch_model.bin'
}

set_seed()


def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data('./dataset/sentiment/sentiment.train.data')
valid_data = load_data('./dataset/sentiment/sentiment.valid.data')
test_data = load_data('./dataset/sentiment/sentiment.test.data')

# 模拟标注和非标注数据
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 0) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]

# 建立分词器

tokenizer = Tokenizer(BertPath["bert_vocab_path"], do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_mask_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, _ ,mask_ids = tokenizer.encode(text, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_mask_ids.append(mask_ids)
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_mask_ids = sequence_padding(batch_mask_ids)
                batch_labels = to_categorical(batch_labels, num_classes)
                yield batch_token_ids, batch_mask_ids, batch_labels
                batch_token_ids, batch_mask_ids, batch_labels = [], [], []


# 转换数据集
train_generator = data_generator(train_data, batch_size).forfit()
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)
vat_generator=data_generator(unlabeled_data, batch_size).forfit()

class TextModel(BertPreTrainedModel):
    def __init__(self, config):
        super(TextModel, self).__init__(config)
        self.bert = BertModel(config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear=nn.Linear(config.hidden_size,num_classes)
        #self.softmax = nn.Softmax(dim=-1)
        self.init_weights()

    def forward(self, token_ids, mask_token_ids,noise=None):
        '''
        :param token_ids:[batch,seq]
        :param mask_token_ids: [batch,seq]
        :return: [batch]
        '''

        bert_out = self.get_embed(token_ids, mask_token_ids,noise)
        embed = self.dropout(bert_out)
        logits=self.linear(embed)
        return logits
    def get_embed(self,token_ids, mask_token_ids,noise=None):
        bert_out = self.bert(input_ids=token_ids.long(), attention_mask=mask_token_ids.long(),noise=noise)
        return bert_out[1]

config = BertConfig.from_pretrained(BertPath["bert_config_path"])

model= TextModel.from_pretrained(BertPath["bert_model_path"],config=config)
model.to(device)


def evaluate(data,state="eval"):
    total, right = 0., 0.
    softmax=nn.Softmax(dim=-1)
    model.eval()
    for batch in tqdm(data,desc=state,ncols=80):
        batch = [torch.tensor(d).to(device) for d in batch]
        batch_token_ids, batch_mask_ids, batch_labels = batch
        with torch.no_grad():
            logits = model(batch_token_ids, batch_mask_ids)
            pred = softmax(logits).detach().cpu().numpy()
        y_pred = pred.argmax(axis=1)
        y_true = batch_labels.cpu().numpy().argmax(axis=1)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total



if __name__ == '__main__':
    t_total = steps_per_epoch * num_epoch

    """ 优化器准备 """
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    optimizer = optim.Adam(model.parameters(), lr=lr)

    vat_optimizer = optim.Adam(model.parameters(), lr=VAT_lr)


    best_val_acc,best_test_acc=0.0,0.0
    step = 0
    for epoch in range(num_epoch):
        model.train()
        epoch_loss = 0
        epoch_vat_loss = 0
        with tqdm(total=steps_per_epoch, desc="train", ncols=80) as t:
            for i, batch in zip(range(steps_per_epoch),train_generator):
                batch = [torch.tensor(d).to(device) for d in batch]
                batch_token_ids, batch_mask_ids, batch_labels = batch
                logits = model(token_ids=batch_token_ids, mask_token_ids=batch_mask_ids)
                loss=KLD(target=batch_labels,input=logits,target_from_logits=False)
                # loss.backward()
                # optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                # model.zero_grad()
                #vat
                if use_vat:
                    batch=next(vat_generator)
                    batch = [torch.tensor(d).to(device) for d in batch]
                    batch_token_ids, batch_mask_ids, _ = batch
                    r_adv,logits=virtual_adversarial_training(model,batch_token_ids, batch_mask_ids)
                    y_hat= model(token_ids=batch_token_ids, mask_token_ids=batch_mask_ids,noise=r_adv)
                    vat_loss = KLD(input=y_hat, target=logits)
                    epoch_vat_loss += (vat_loss.item() / steps_per_epoch)
                    # vat_loss.backward()
                    # vat_optimizer.step()
                    # vat_scheduler.step()  # Update learning rate schedule
                    # model.zero_grad()
                    loss=loss+alpha*vat_loss
                loss.backward()
                optimizer.step()
                model.zero_grad()

                step += 1
                epoch_loss += (loss.item()/steps_per_epoch)

                t.set_postfix(loss="[%.5f,%.5f]"%(epoch_loss,epoch_vat_loss))
                t.update(1)

        val_acc = evaluate(valid_generator,state="eval")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(output_path, WEIGHTS_NAME))  # 保存最优模型权重
            torch.save(optimizer.state_dict(), os.path.join(output_path, "optimizer.pt"))
            #torch.save(scheduler.state_dict(), os.path.join(output_path, "scheduler.pt"))

        test_acc = evaluate(test_generator,state="test")
        if best_test_acc<test_acc:
            best_test_acc=test_acc
        with open(output_path + "log.txt", "a") as f:
            print(
                u'epoch:%d\tval_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f, best_test_acc: %.5f\n' %
                (epoch, val_acc, best_val_acc, test_acc,best_test_acc), file=f
            )
        print(
            u'epoch:%d\tval_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f, best_test_acc: %.5f\n' %
            (epoch, val_acc, best_val_acc, test_acc,best_test_acc)
        )
