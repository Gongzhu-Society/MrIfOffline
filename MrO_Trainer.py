#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
from Util import ORDER_DICT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle

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
            if 'weight' in child.state_dict():
                #stru.append(tuple(child.state_dict()['weight'].t().size()))
                stru.append(child.state_dict()['weight'].shape)
        return "%s %s %s"%(self.__class__.__name__,stru,self.num_paras())

class MrO_Last(MrO_Father):
    """
        输入当前局面, 返回评估分数
    """
    def __init__(self):
        super(MrO_Last,self).__init__()
        self.fc0=nn.Linear(52*8+16*4,512)#四个人手里的牌(52x4), 桌上的牌(52x4), 四个人手里的分(16x4)
        self.fc1=nn.Linear(512,512)
        self.fc2=nn.Linear(512,512)
        self.fc3=nn.Linear(512,256)
        self.fc4=nn.Linear(256,128)
        self.fc5=nn.Linear(128,128)
        self.fc6=nn.Linear(128,128)
        self.fc7=nn.Linear(128,128)
        self.fc8=nn.Linear(128,128)
        self.fc9=nn.Linear(128,128)
        self.fca=nn.Linear(128,128)
        self.fcb=nn.Linear(128,128)
        self.fcc=nn.Linear(128,128)
        self.fcd=nn.Linear(128,128)
        self.fce=nn.Linear(128,64)
        self.fcf=nn.Linear(64,1)

        self.avgp=torch.nn.AvgPool1d(2)

    def forward(self, x):
        x=F.relu(self.fc0(x))
        x=F.relu(self.fc1(x))+x
        x=F.relu(self.fc2(x))+x
        x=F.relu(self.fc3(x))+self.avgp(x.view(-1,1,512)).view(-1,256)
        x=F.relu(self.fc4(x))+self.avgp(x.view(-1,1,256)).view(-1,128)
        x=F.relu(self.fc5(x))+x
        x=F.relu(self.fc6(x))+x
        x=F.relu(self.fc7(x))+x
        x=F.relu(self.fc8(x))+x
        x=F.relu(self.fc9(x))+x
        x=F.relu(self.fca(x))+x
        x=F.relu(self.fcb(x))+x
        x=F.relu(self.fcc(x))+x
        x=F.relu(self.fcd(x))+x
        x=F.relu(self.fce(x))+self.avgp(x.view(-1,1,128)).view(-1,64)
        x=self.fcf(x)
        return x

class MrO_First_Failed(MrO_Father):
    def __init__(self):
        super(MrO_First,self).__init__()
        self.fc0=nn.Linear(52*5+16*4,128)#四个人手里的牌(52x4), 桌上的牌(52x4), 四个人手里的分(16x4)
        self.fc1=nn.Linear(128,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,128)
        self.fc4=nn.Linear(128,64)
        self.fc5=nn.Linear(64,64)
        self.fc6=nn.Linear(64,64)
        self.fc7=nn.Linear(64,64)
        self.fc8=nn.Linear(64,64)
        self.fc9=nn.Linear(64,64)
        self.fca=nn.Linear(64,64)
        self.fcb=nn.Linear(64,64)
        self.fcc=nn.Linear(64,64)
        self.fcd=nn.Linear(64,32)
        self.fce=nn.Linear(32,16)
        self.fcf=nn.Linear(16,1)

        self.avgp=nn.AvgPool1d(2)
        #self.adap1=nn.AdaptiveAvgPool1d(32)
        #self.adap2=nn.AdaptiveAvgPool1d(16)
        self.norm1=nn.BatchNorm1d(128,affine=False)
        self.norm2=nn.BatchNorm1d(64,affine=False)

    def forward(self, x):
        x=F.relu(self.fc0(x))
        x=self.norm1(x)
        x=F.relu(self.fc1(x))+x
        x=self.norm1(x)
        x=F.relu(self.fc2(x))+x
        x=self.norm1(x)
        x=F.relu(self.fc3(x))+x
        x=self.norm1(x)
        x=F.relu(self.fc4(x))+self.avgp(x.view(-1,1,128)).view(-1,64)
        x=self.norm2(x)
        x=F.relu(self.fc5(x))+x
        x=self.norm2(x)
        x=F.relu(self.fc6(x))+x
        x=self.norm2(x)
        x=F.relu(self.fc7(x))+x
        x=self.norm2(x)
        x=F.relu(self.fc8(x))+x
        x=self.norm2(x)
        x=F.relu(self.fc9(x))+x
        x=self.norm2(x)
        x=F.relu(self.fca(x))+x
        x=self.norm2(x)
        x=F.relu(self.fcb(x))+x
        x=self.norm2(x)
        x=F.relu(self.fcc(x))+x
        x=self.norm2(x)
        x=F.relu(self.fcd(x))+self.avgp(x.view(-1,1,64)).view(-1,32)
        x=F.relu(self.fce(x))+self.avgp(x.view(-1,1,32)).view(-1,16)
        x=self.fcf(x)
        return x

class MrO_First(MrO_Father):
    def __init__(self):
        super(MrO_First,self).__init__()
        self.conv1=nn.Conv2d(1,8,(4,5))
        self.fc0=nn.Linear(48*8+52+16*4,128)#四个人手里的牌(52x4), 桌上的牌(52x4), 四个人手里的分(16x4)
        self.fc1=nn.Linear(128,128)
        self.fc2=nn.Linear(128,128)
        self.fc3=nn.Linear(128,64)
        self.fc4=nn.Linear(64,64)
        self.fc5=nn.Linear(64,64)
        self.fc6=nn.Linear(64,64)
        self.fc7=nn.Linear(64,32)
        self.fc8=nn.Linear(32,16)
        self.fc9=nn.Linear(16,1)

        self.avgp=nn.AvgPool1d(2)
        #self.adap1=nn.AdaptiveAvgPool1d(32)
        #self.adap2=nn.AdaptiveAvgPool1d(16)
        #self.norm1=nn.BatchNorm1d(128,affine=False)
        #self.norm2=nn.BatchNorm1d(64,affine=False)

    def forward(self,x):
        f1=self.conv1(x[:,0:52*4].view(-1,1,4,52)).view(-1,48*8)
        x=torch.cat((f1,x[:,52*4:52*5+16*4]),1)
        x=F.relu(self.fc0(x))
        x=F.relu(self.fc1(x))+x
        x=F.relu(self.fc2(x))+x
        x=F.relu(self.fc3(x))+self.avgp(x.view(-1,1,128)).view(-1,64)
        x=F.relu(self.fc4(x))+x
        x=F.relu(self.fc5(x))+x
        x=F.relu(self.fc6(x))+x
        x=F.relu(self.fc7(x))+self.avgp(x.view(-1,1,64)).view(-1,32)
        x=F.relu(self.fc8(x))+self.avgp(x.view(-1,1,32)).view(-1,16)
        x=self.fc9(x)
        return x

def cards_in_hand_oh(cards_list_list,my_cards):
    """
        get a 52x4 one hot
        the order is [me,me-1,me-2,me-3]
    """
    oh=torch.zeros(52*4,dtype=torch.uint8)
    for c in my_cards:
        oh[ORDER_DICT[c]]=1
    for c in cards_list_list[2]:
        oh[ORDER_DICT[c]+52]=1
    for c in cards_list_list[1]:
        oh[ORDER_DICT[c]+52*2]=1
    for c in cards_list_list[0]:
        oh[ORDER_DICT[c]+52*3]=1
    return oh

def four_cards_oh(cards_on_table,my_choice):
    """
        get oh of four_cards like ["","",...], the length depends on len(cards_on_table)
        the order is [me,me-1,me-2,me-3]
    """
    if len(cards_on_table)==1:
        oh=torch.zeros(52,dtype=torch.uint8)
        oh[ORDER_DICT[my_choice]]=1
    elif len(cards_on_table)==2:
        oh=torch.zeros(52*2,dtype=torch.uint8)
        oh[ORDER_DICT[my_choice]]=1
        oh[ORDER_DICT[cards_on_table[1]]+52]=1
    elif len(cards_on_table)==3:
        oh=torch.zeros(52*3,dtype=torch.uint8)
        oh[ORDER_DICT[my_choice]]=1
        oh[ORDER_DICT[cards_on_table[1]]+52*2]=1
        oh[ORDER_DICT[cards_on_table[2]]+52]=1
    elif len(cards_on_table)==4:
        oh=torch.zeros(52*4,dtype=torch.uint8)
        oh[ORDER_DICT[my_choice]]=1
        oh[ORDER_DICT[cards_on_table[1]]+52*3]=1
        oh[ORDER_DICT[cards_on_table[2]]+52*2]=1
        oh[ORDER_DICT[cards_on_table[3]]+52]=1
    return oh

def score_oh(score,place):
    """
        get a 16x4 one hot
        the order is [me,me-1,me-2,me-3]
    """
    oh=torch.zeros(16*4,dtype=torch.uint8)
    for i in range(4):
        for c in score[(place-i)%4]:
            oh[ORDER_DICT5[c]+52*i]=1
    return oh

def test_net_accu(net,testloder):
    with torch.no_grad():
        test_iter=testloder.__iter__()
        i=test_iter.__next__()
        netout=net(i[0])
        test_loss=F.l1_loss(netout.view(-1),i[1])
    try:
        test_iter.__next__()
        log("testloader can still be loaded!")
        raise KeyBoardInterrupt
    except StopIteration:
        pass
    return test_loss

def train(Net_Class,order):
    from MrGreed import gen_data_for_o
    net=Net_Class()
    device=torch.device("cuda:0")
    #device=torch.device("cpu")
    net.to(device)
    log(net)
    #prepare data
    with open("./Greed_Data_for_O/Greed_32.data",'rb') as f:
        testdata=pickle.load(f)[order]
    testdata=[[i[0].float().to(device),i[1].float().to(device)] for i in testdata]
    testloder=torch.utils.data.DataLoader(testdata,batch_size=len(testdata))
    log("testdata len: %d"%(len(testdata)))
    with open("./Greed_Data_for_O/Greed_128.data",'rb') as f:
        traindata=pickle.load(f)[order]
    traindata=[[i[0].float().to(device),i[1].float().to(device)] for i in traindata]
    batch_size=1000
    trainloader=torch.utils.data.DataLoader(traindata,batch_size=batch_size,drop_last=True)
    log("traindata len: %d"%(len(traindata)))
    #train
    test_loss=test_net_accu(net,testloder)
    log("random init: %f"%(test_loss,))
    log("epoch num: train_loss test_loss")
    optimizer=optim.SGD(net.parameters(),lr=0.005,momentum=0.9)
    for epoch in range(500):
        running_loss=0
        for i in trainloader:
            optimizer.zero_grad()
            netout=net(i[0])
            loss=F.l1_loss(netout.view(-1),i[1])
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
        if epoch%20==0:
            test_loss=test_net_accu(net,testloder)
            log("%3d: %f %f"%(epoch,running_loss/(len(traindata)//batch_size),test_loss))
            if epoch==200:
                optimizer=optim.SGD(net.parameters(),lr=0.001,momentum=0.9)
                log("decrease lr")
    save_name='%s_%s_%s.pkl'%(net.__class__.__name__,net.num_layers(),net.num_paras())
    torch.save(net,save_name)
    log("saved net to %s"%(save_name))

if __name__=="__main__":
    #train(MrO_Last,3)
    train(MrO_First,0)