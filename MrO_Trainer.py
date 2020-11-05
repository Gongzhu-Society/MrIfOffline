#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
from Util import ORDER_DICT
import torch,copy,math
import torch.nn as nn
import torch.optim as optim

class MrO_Father(nn.Module):
    def num_paras(self):
        return sum([p.numel() for p in self.parameters()])

    def num_layers(self):
        ax=0
        for name,child in self.named_children():
            ax+=1
        return ax

    def __str__(self):
        stru=[]
        for name,child in self.named_children():
            stru.append(tuple(child.state_dict()['weight'].t().size()))
        return "%s %s %s"%(self.__class__.__name__,stru,self.num_paras())

Class MrO(MrO_Father):
    """
        输入当前局面, 返回评估分数
    """
    def __init__(self):
        super(MrO,self).__init__()
        self.fc1=nn.Linear(52*5+16*4+4*4,256)#我手里的牌(52), 桌上的牌(52x3), 另外三个人手牌的并集(52), 四个人手里的分, 四个人的断门
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,64)
        self.fc4=nn.Linear(64,64)
        self.fc5=nn.Linear(64,1)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=self.fc5(x)
        return x

def test_net_accu(net,testloder):
    with torch.no_grad():
        test_iter=testloder.__iter__()
        i=test_iter.__next__()
        netout=net(i[0])
        test_loss=loss_func_mul(netout,i[2],i[1])
        test_corr=correct_num(netout,i[2],i[1])/len(i[2])
    try:
        test_iter.__next__()
        log("testloader can still be loaded!")
        raise KeyBoardInterrupt
    except StopIteration:
        log("testloader indeed stoped")
    return test_loss,test_corr

def train(NetClass,order):
    train_files=[2,3,4,5,6,7]
    traindata=[]
    for num in train_files:
        parse_data("./Greed_batch/Greed_batch%d.txt"%(num),traindata,order)
    batch_size=200
    trainloader=torch.utils.data.DataLoader(traindata,batch_size=batch_size,drop_last=True)
    train_iter_num=int(len(traindata)/batch_size)
    testdata=[]
    parse_data("./Greed_batch/Greed_batch1.txt",testdata,order,max_num=128)
    testloder=torch.utils.data.DataLoader(testdata,batch_size=len(testdata))

    net=NetClass()
    log(net)
    test_loss,test_corr=test_net_accu(net,testloder)
    log("random init: %f %f"%(test_loss,test_corr))
    log("epoch num: train_loss test_loss test_correct_ratio")
    optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
    for epoch in range(1000):
        running_loss=0
        for i in trainloader:
            optimizer.zero_grad()
            netout=net(i[0])
            loss=loss_func_mul(netout,i[2],i[1])
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        if epoch%50==0:
            test_loss,test_corr=test_net_accu(net,testloder)
            log("%3d: %f %f %f"%(epoch,running_loss/train_iter_num,test_loss,test_corr))
    #save_name='%s_%s_%s.ckpt'%(net.__class__.__name__,net.num_layers(),net.num_paras())
    #torch.save(net.state_dict(),save_name)
    #log("saved net to %s"%(save_name))