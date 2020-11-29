#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import ORDER_DICT,ORDER_DICT4,ORDER_DICT5,ORDER_DICT2
#from MrRandom import MrRandom
import torch,copy,math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import re
pattern_shuffle=re.compile("shuffle: (.+)")
pattern_play=re.compile("greed([0-3]) played ([SHDC][0-9JQKA]{1,2}), (.+)")
pattern_gamend=re.compile("game end: (\\[.+?\\])")
pattern_trick=re.compile("trick end. winner is ([0-3]), (.+)")

class NN_Last(nn.Module):
    def __init__(self):
        super(NN_Last,self).__init__()
        self.fc1=nn.Linear(52*5+16*4+4*4,256)#我手里剩的牌, 另外仨人手里剩的牌, 前三个人打的牌, 四个人手里的分, 四个人的断门
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,64)
        self.fc4=nn.Linear(64,64)
        self.fc5=nn.Linear(64,52)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=self.fc5(x)
        return x

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

class NN_Third(nn.Module):
    def __init__(self):
        super(NN_Third,self).__init__()
        self.fc1=nn.Linear(52*4+16*4+4*4,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,64)
        self.fc4=nn.Linear(64,64)
        self.fc5=nn.Linear(64,64)
        self.fc6=nn.Linear(64,64)
        self.fc7=nn.Linear(64,52)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=F.relu(self.fc6(x))
        x=self.fc7(x)
        return x

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

class NN_Second(nn.Module):
    def __init__(self):
        super(NN_Second,self).__init__()
        self.fc1=nn.Linear(52*3+16*4+4*4,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,64)
        self.fc4=nn.Linear(64,64)
        self.fc5=nn.Linear(64,64)
        self.fc6=nn.Linear(64,64)
        self.fc7=nn.Linear(64,64)
        self.fc8=nn.Linear(64,64)
        self.fc9=nn.Linear(64,52)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=F.relu(self.fc6(x))
        x=F.relu(self.fc7(x))
        x=F.relu(self.fc8(x))
        x=self.fc9(x)
        return x

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

class NN_First(nn.Module):
    def __init__(self):
        super(NN_First,self).__init__()
        self.fc1=nn.Linear(52*2+16*4+4*4,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,64)
        self.fc4=nn.Linear(64,64)
        self.fc5=nn.Linear(64,64)
        self.fc6=nn.Linear(64,64)
        self.fc7=nn.Linear(64,64)
        self.fc8=nn.Linear(64,64)
        self.fc9=nn.Linear(64,64)
        self.fca=nn.Linear(64,64)
        self.fcb=nn.Linear(64,52)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=F.relu(self.fc6(x))
        x=F.relu(self.fc7(x))
        x=F.relu(self.fc8(x))
        x=F.relu(self.fc9(x))
        x=F.relu(self.fca(x))
        x=self.fcb(x)
        return x

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

def gen_legal_onehot(cards_on_table,cards_l):
    if len(cards_on_table)>2:
        c_num=ORDER_DICT4[cards_on_table[1][0]]
        mask=[0]*(c_num*13)+[1]*(13)+[0]*((3-c_num)*13)
        mask=torch.tensor(mask)
        l_onehot=mask*cards_l
        if torch.sum(l_onehot)>0:
            return l_onehot
    return cards_l.clone()

def parse_data(file,list_pt,order,max_num=1e5):
    """order是把第几个出牌的挑出来,比如说order=1的意思就是第二个出牌的"""
    log("parsing data from %s..."%(file))
    pre_len=len(list_pt)
    cards_ll=None #list of list of cards, cards remain
    score_ll=None
    void_ll=None
    ax=0
    f=open(file)
    for line in f:
        s0=pattern_play.search(line)
        if s0:
            who=int(s0.group(1))
            which=ORDER_DICT[s0.group(2)]
            cards_on_table=eval(s0.group(3))
            if len(cards_on_table)-2==order:
                cards_legal=gen_legal_onehot(cards_on_table,cards_ll[who])
                if torch.sum(cards_legal)>1:
                    cards_in_others=cards_ll[0]+cards_ll[1]+cards_ll[2]+cards_ll[3]-cards_ll[who]
                    to_input=[cards_ll[who],cards_in_others]
                    for i in cards_on_table[1:-1]:
                        to_input.append(torch.zeros(52))
                        to_input[-1][ORDER_DICT[i]]=1
                    for i in range(4):
                        temp=(i+cards_on_table[0])%4
                        to_input.append(score_ll[temp])
                        to_input.append(void_ll[temp])
                    to_input=torch.cat(to_input)
                    list_pt.append([to_input,cards_legal,which])
            cards_ll[who][which]=0
            if cards_on_table[-1][0]!=cards_on_table[1][0]:
                void_ll[who][ORDER_DICT4[cards_on_table[1][0]]]=1
            continue
        s1=pattern_shuffle.search(line)
        if s1:
            cards_ll=[torch.zeros(52),torch.zeros(52),torch.zeros(52),torch.zeros(52)]
            cards_shuffle=eval(s1.group(1))
            for i in range(4):
                assert len(cards_shuffle[i])==13
                for c in cards_shuffle[i]:
                    cards_ll[i][ORDER_DICT[c]]=1
            score_ll=[torch.zeros(16),torch.zeros(16),torch.zeros(16),torch.zeros(16)]
            void_ll=[torch.zeros(4),torch.zeros(4),torch.zeros(4),torch.zeros(4)]
            continue
        s2=pattern_gamend.search(line)
        if s2:
            cards_ll=None
            score_ll=None
            void_ll=None
            ax+=1
            if ax>=max_num:
                break
            continue
        s3=pattern_trick.search(line)
        if s3:
            scores=eval(s3.group(2))
            for i in range(4):
                for j in scores[i]:
                    score_ll[i][ORDER_DICT5[j]]=1
    f.close()
    log("parse finish. got %d datas"%(len(list_pt)-pre_len,))

def loss_func_mul(netout,target_index,legal_mask):
    o1=netout*legal_mask
    loss_1=F.cross_entropy(o1,target_index)
    return loss_1

def correct_num(netout,target_index,legal_mask):
    _,max_i=torch.max(netout*legal_mask,1)
    return (max_i==target_index).sum().item()

def train(NetClass,order):
    files=["./Greed_batch/Greed_batch2.txt","./Greed_batch/Greed_batch3.txt",
           "./Greed_batch/Greed_batch4.txt","./Greed_batch/Greed_batch5.txt",
           "./Greed_batch/Greed_batch6.txt","./Greed_batch/Greed_batch7.txt",]
    traindata=[]
    for f_name in files:
        parse_data(f_name,traindata,order)
    batch_size=200
    trainloader=torch.utils.data.DataLoader(traindata,batch_size=batch_size,drop_last=True)
    train_iter_num=int(len(traindata)/batch_size)
    testdata=[]
    parse_data("./Greed_batch/Greed_batch1.txt",testdata,order,max_num=128)
    testloder=torch.utils.data.DataLoader(testdata,batch_size=len(testdata))

    net=NetClass()
    log(net)
    for i in testloder:
        netout=net(i[0])
        test_loss=loss_func_mul(netout,i[2],i[1])
        test_corr=correct_num(netout,i[2],i[1])/len(i[2])
    log("random init: %f %f"%(test_loss,test_corr))
    log("epoch num: train_loss test_loss test_correct_ratio")
    optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.9)
    optimizer.zero_grad()
    scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[200,400],gamma=0.1)
    for epoch in range(1000):
        running_loss=0
        for i in trainloader:
            netout=net(i[0])
            loss=loss_func_mul(netout,i[2],i[1])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss+=loss.item()
        scheduler.step()
        if epoch%50==0:
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
            log("%3d: %f %f %f"%(epoch,running_loss/train_iter_num,test_loss,test_corr))
    #save_name='%s_%s_%s.ckpt'%(net.__class__.__name__,net.num_layers(),net.num_paras())
    #torch.save(net.state_dict(),save_name)
    #log("saved net to %s"%(save_name))

def test_accu(NetClass,FileName,order):
    net=NetClass()
    log(net)
    net.load_state_dict(torch.load(FileName))
    testdata=[]
    parse_data("./Greed_batch/Greed_batch1.txt",testdata,order,max_num=128)
    testloder=torch.utils.data.DataLoader(testdata,batch_size=len(testdata))
    with torch.no_grad():
        test_iter=testloder.__iter__()
        i=test_iter.__next__()
        log("data size: %s"%(i[0].size(),))
        netout=net(i[0])
        test_loss=loss_func_mul(netout,i[2],i[1])
        test_corr=correct_num(netout,i[2],i[1])/len(i[2])
        try:
            test_iter.__next__()
            log("testloader can still be loaded!")
            raise KeyBoardInterrupt
        except StopIteration:
            log("testloader indeed stoped")
    log("%f %f"%(test_loss,test_corr))

def test_DataLoader():
    datalist=[[i,2*i,torch.tensor([1,2,3])] for i in range(2)]
    #datalist=[]
    #parse_data("./Greed_batch/Greed_batch2.txt",datalist,3,max_num=1)
    #parse_data("./Greed_batch/Greed_batch3.txt",datalist,3,max_num=2)
    dataloader=torch.utils.data.DataLoader(datalist,batch_size=1,drop_last=True)
    it=dataloader.__iter__()
    i=it.__next__()
    for i in dataloader:
        print(i)


if __name__=="__main__":
    #test_DataLoader()
    #train(NN_Last,3)
    #train(NN_Third,2)
    #train(NN_Second,1)
    #train(NN_First,0)
    #test_accu(NN_First,'./NetPara20200715/NN_First_11_121012.ckpt',0)
    #test_accu(NN_Second,'./NetPara20200715/NN_Second_9_126004.ckpt',1)
    #test_accu(NN_Third,'./NetPara20200715/NN_Third_7_130996.ckpt',2)
    test_accu(NN_Last,'./NetPara20200715/NN_Last_5_135988.ckpt',3)

"""
# Save and load the entire model.
torch.save(resnet, 'model.ckpt')
model = torch.load('model.ckpt')

# Save and load only the model parameters (recommended).
torch.save(resnet.state_dict(), 'params.ckpt')
resnet.load_state_dict(torch.load('params.ckpt'))
"""