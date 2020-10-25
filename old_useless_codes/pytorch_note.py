#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from __future__ import print_function
import torch

#malloc但是不初始化数值
torch.empty(5,3)

#uniform distribution on the interval :math:[0, 1)
torch.rand(5,3)

torch.zeros(5,3,dtype=torch.long)
torch.empty(5,3)

#Construct a tensor directly from data
torch.tensor([5.5,3])

torch.randn_like(x,dtype=torch.float)    #override dtype!

print(x.size()) #torch.Size([5,3])

torch.randn(4, 4)
x.view(16)
x.view(-1,8)

b=a.numpy()
torch.from_numpy(a)

with open(savefilename,'wb') as f2:
    pickle.dump(train_data,f2)
with open("./Greed_batch/Greed_batch1.4train",'rb') as f3:
    train_data=pickle.load(f3)