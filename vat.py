import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from torch.autograd import Variable
import torch.optim as optim

softmax=nn.Softmax(dim=1)
log_softmax=nn.LogSoftmax(dim=1)
kld_loss=nn.KLDivLoss()

def set_seed():
    # 每次运行代码时设置相同的seed，则每次生成的随机数也相同，如果不设置seed，则每次生成的随机数都会不一样
    np.random.seed(1)
    torch.manual_seed(1)
set_seed()

def _l2_normalize(d):
    if type(d) is torch.Tensor:
        d = d.cpu().detach().numpy()
    d=d / (np.sqrt((d ** 2).sum()) + 1e-8)
    return torch.from_numpy(d)


def KLD(input , target ,input_from_logits=True ,target_from_logits=True ):
    if input_from_logits:
        input=log_softmax(input)
    if target_from_logits:
        target=softmax(target)
    return kld_loss(input=input,target=target)

def virtual_adversarial_training(
    model,batch_token_ids, batch_mask_ids, epsilon=1, xi=10, iters=1
):
    new_model=deepcopy(model)
    device=model.device
    with torch.no_grad():
        logits=new_model(token_ids=batch_token_ids, mask_token_ids=batch_mask_ids)
        logits=logits.detach() #无噪音embed的模型输出
    logits=Variable(logits.to(device), requires_grad=False)
    d=np.random.randn(batch_token_ids.shape[0],batch_token_ids.shape[1],768) #噪音初始化
    for _ in range(iters):  # 迭代求扰动
        d = xi * _l2_normalize(d).float()
        d = Variable(d.to(device), requires_grad=True)
        y_hat = new_model(token_ids=batch_token_ids, mask_token_ids=batch_mask_ids,noise=d)
        delta_kl = KLD(input=y_hat , target=logits)
        delta_kl.backward()
        d = d.grad.detach() #得到噪音d的梯度
        new_model.zero_grad()

    d = epsilon*_l2_normalize(d).to(device)
    d = Variable(d.to(device),requires_grad=True)
    r_adv = epsilon *d #噪音

    return r_adv,logits