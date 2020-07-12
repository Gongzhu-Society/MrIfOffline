#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import ORDER_DICT
from MrRandom import MrRandom
import torch,copy,pickle
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print_level=0

class MrNN(MrRandom):
    pass

class NN_First(nn.Module):
    def __init__(self):
        super(NN_First,self).__init__()
        self.fc1=nn.Linear(52*2,52*2)  #我手里剩的牌和另外仨人手里剩的牌
        self.fc2=nn.Linear(52*2,52*2)
        self.fc3=nn.Linear(52*2,52*2)
        self.fc4=nn.Linear(52*2,52)
        self.fc5=nn.Linear(52,52)
        self.fc6=nn.Linear(52,52)
        self.fc7=nn.Linear(52,52)
        self.fc8=nn.Linear(52,52)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=F.relu(self.fc6(x))
        x=F.relu(self.fc7(x))
        x=torch.exp(self.fc8(x))
        #x=F.relu(self.fc5(x))
        return x

    def num_flat_features(self,x):
        size=x.size()[1:]  #all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

ORDER_DICT4={'S':0,'H':1,'D':2,'C':3}

def gen_legal_onehot(cards_on_table,cards_in_hand):
    if len(cards_on_table)>2:
        c_num=ORDER_DICT4[cards_on_table[1][0]]
        mask=[0]*(c_num*13)+[1]*(13)+[0]*((3-c_num)*13)
        mask=torch.tensor(mask)
        l_onehot=mask*cards_in_hand
        if torch.sum(l_onehot)>0:
            return l_onehot
    return cards_in_hand.clone()

def parse_train_data(file,trick_num):
    import re
    pattern_shuffle=re.compile("shuffle: (.+)")
    pattern_play=re.compile("greed([0-3]) played ([SHDC][0-9JQKA]{1,2}), (.+)")
    pattern_gamend=re.compile("game end: (\\[.+?\\])")
    cards_in_hand=None
    train_data={0:[],1:[],2:[],3:[]}
    log("parsing data from %s..."%(file))
    f=open(file)
    ax=0
    for line in f.readlines():
        s0=pattern_play.search(line)
        if s0:
            who=int(s0.group(1))
            which=ORDER_DICT[s0.group(2)]
            cards_on_table=eval(s0.group(3))
            order=len(cards_on_table)-2
            #桌上的牌的onehot
            table_oh=torch.zeros(52)
            for c in cards_on_table[1:len(cards_on_table)-1]:
                table_oh[ORDER_DICT[c]]=1
            #别人手里的牌
            cards_in_others=cards_in_hand[0]+cards_in_hand[1]+cards_in_hand[2]+cards_in_hand[3]-cards_in_hand[who]
            #合法的选择
            cards_legal=gen_legal_onehot(cards_on_table,cards_in_hand[who])
            #最后选择的卡的onehot
            cards_choiced=torch.zeros(52)
            cards_choiced[which]=1
            #手里的牌的onehot
            my_cards=cards_in_hand[who].clone()
            #log(my_cards)
            l_temp=[order,table_oh,cards_in_others,my_cards,cards_legal,cards_choiced]
            train_data[order].append(l_temp)
            assert cards_in_hand[who][which]==1
            cards_in_hand[who][which]=0
            continue
        s1=pattern_shuffle.search(line)
        if s1:
            cards_in_hand=[torch.zeros(52),torch.zeros(52),torch.zeros(52),torch.zeros(52)]
            cards_shuffle=eval(s1.group(1))
            for i in range(4):
                assert len(cards_shuffle[i])==13
                for c in cards_shuffle[i]:
                    cards_in_hand[i][ORDER_DICT[c]]=1
            #log("get shuffle: %s"%(cards_in_hand))
            continue
        s2=pattern_gamend.search(line)
        if s2:
            ax+=1
            if ax==trick_num:
                break
            #print("get game end: %s"%(s2.group(1)))
            #break
            continue
    f.close()
    log("parse finish")
    savefilename=file.split('.')
    savefilename[-1]='4train'
    with open('.'.join(savefilename),'wb') as f2:
        pickle.dump(train_data,f2)
    return train_data

def loss_func_single(netout,target,legal_mask):
    #print("%s\n%s\n%s"%(netout,target,legal_mask))
    o1=netout*legal_mask
    #log(o1)
    o2=o1/(torch.sum(o1)+1e-7)
    #log(o2)
    #print(o2.sum())
    loss_1=torch.sum(torch.pow(o2-target,2))
    if print_level>=2:
        log(o1)
        log(o2)
        log(loss_1)
        input()
    #loss_2=torch.sum(torch.pow(netout-o1,2))
    #print(loss_1.item(),loss_2.item())
    #log(loss_1)
    return loss_1

def train_first():
    global print_level
    #with open("./Greed_batch/Greed_batch1.4train",'rb') as f3:
    #    train_data=pickle.load(f3)
    net=NN_First()
    train_data=parse_train_data("./Greed_batch/Greed_batch1.txt",trick_num=512)
    check_data=parse_train_data("./Greed_batch/Greed_batch1.txt",trick_num=16)
    optimizer=optim.SGD(net.parameters(),lr=0.005,momentum=0.9)
    
    for epoch in range(1000):
        log("%dth epoch"%(epoch))
        optimizer.zero_grad()
        running_loss=0
        for i,lis in enumerate(train_data[0]):
            inputs=torch.cat((lis[2],lis[3]))
            netout=net(inputs)
            loss=loss_func_single(netout,lis[5],lis[4])
            if torch.isnan(loss):
                print_level=2
                loss_func_single(netout,lis[5],lis[4])
            loss.backward()
            running_loss+=loss.item()
            if i%52==51:
                #print("step: %f"%(running_loss))
                #running_loss=0
                #input()
                optimizer.step()
                optimizer.zero_grad()
        log("finis loss: %f"%(running_loss/(i+1)))
        check_loss=0
        for i,lis in enumerate(check_data[0]):
            inputs=torch.cat((lis[2],lis[3]))
            netout=net(inputs)
            loss=loss_func_single(netout,lis[5],lis[4])
            check_loss+=loss.item()
        log("check loss: %f"%(check_loss/(i+1),))
    torch.save(net.state_dict(),'nn_first.data')

if __name__=="__main__":
    train_first()
    #parse_train_data("./Greed_batch/Greed_batch1.txt")
    #log(gen_legal_onehot([0,'C5'],torch.zeros(52)))
    