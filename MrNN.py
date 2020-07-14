#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import ORDER_DICT,ORDER_DICT4
from MrRandom import MrRandom
import torch,copy,pickle,re
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

print_level=0

class MrNN(MrRandom):
    pass

class NN_First(nn.Module):
    def __init__(self):
        super(NN_First,self).__init__()
        self.fc1=nn.Linear(52*2,52*8)  #我手里剩的牌和另外仨人手里剩的牌
        self.fc2=nn.Linear(52*8,52*8)
        #self.fc3=nn.Linear(52*8,52*8)
        self.fc4=nn.Linear(52*8,52*4)
        self.fc5=nn.Linear(52*4,52*2)
        self.fc6=nn.Linear(52*2,52)
        #self.fc7=nn.Linear(52,52)
        self.fc8=nn.Linear(52,52)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        #x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=F.relu(self.fc6(x))
        #x=F.relu(self.fc7(x))
        x=self.fc8(x)
        #x=torch.exp(self.fc8(x))
        #x=F.relu(self.fc5(x))
        return x

    def num_flat_features(self,x):
        size=x.size()[1:]  #all dimensions except the batch dimension
        num_features=1
        for s in size:
            num_features*=s
        return num_features

def gen_legal_onehot(cards_on_table,cards_in_hand):
    if len(cards_on_table)>2:
        c_num=ORDER_DICT4[cards_on_table[1][0]]
        mask=[0]*(c_num*13)+[1]*(13)+[0]*((3-c_num)*13)
        mask=torch.tensor(mask)
        l_onehot=mask*cards_in_hand
        if torch.sum(l_onehot)>0:
            return l_onehot
    return cards_in_hand.clone()

pattern_shuffle=re.compile("shuffle: (.+)")
pattern_play=re.compile("greed([0-3]) played ([SHDC][0-9JQKA]{1,2}), (.+)")
pattern_gamend=re.compile("game end: (\\[.+?\\])")

def parse_train_data(file,trick_num=1e5):
    log("parsing data from %s..."%(file))
    cards_in_hand=None
    train_data={0:[],1:[],2:[],3:[]}
    ax=0
    f=open(file)
    for line in f.readlines():
        s0=pattern_play.search(line)
        if s0:
            who=int(s0.group(1))
            which=ORDER_DICT[s0.group(2)]
            cards_on_table=eval(s0.group(3))
            order=len(cards_on_table)-2
            #合法的选择
            cards_legal=gen_legal_onehot(cards_on_table,cards_in_hand[who])
            if torch.sum(cards_legal)>1:
                #别人手里的牌
                cards_in_others=cards_in_hand[0]+cards_in_hand[1]\
                    +cards_in_hand[2]+cards_in_hand[3]-cards_in_hand[who]
                #手里的牌的onehot
                my_cards=cards_in_hand[who].clone()
                #log(my_cards)
                l_temp=[copy.copy(cards_on_table[1:-1]),cards_in_others,my_cards,cards_legal,which]
                train_data[order].append(l_temp)
            #assert cards_in_hand[who][which]==1
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
            continue
        s2=pattern_gamend.search(line)
        if s2:
            ax+=1
            if ax==trick_num:
                break
            continue
    f.close()
    log("parse finish: %d, %d"%(len(train_data[0]),len(train_data[1])))
    return train_data

def loss_func_mul(netout,target_index,legal_mask):
    """netout and legal_mask is stacked by their monomer
       target_index should already be tensor"""
    o1=netout*legal_mask
    loss_1=F.cross_entropy(o1,target_index)
    return loss_1

def correct_num(netout,target_index,legal_mask):
    _,max_i=torch.max(netout*legal_mask,1)
    return torch.sum(max_i==target_index).item()

def train_first_adv():
    files=["./Greed_batch/Greed_batch2.txt","./Greed_batch/Greed_batch3.txt"]
    train_datas=[parse_train_data(f) for f in files]
    inputs=[];legal_masks=[];corr_ans=[]
    for i in train_datas:
        inputs.append(torch.stack([torch.cat((j[1],j[2])) for j in i[0]]))
        legal_masks.append(torch.stack([j[3] for j in i[0]]))
        corr_ans.append(torch.tensor([j[4] for j in i[0]]))
    check_data=parse_train_data("./Greed_batch/Greed_batch1.txt",trick_num=64)
    check_input=torch.stack([torch.cat((j[1],j[2])) for j in check_data[0]])
    check_mask=torch.stack([j[3] for j in check_data[0]])
    check_target=torch.tensor([j[4] for j in check_data[0]])
    
    #开始训练网络
    net=NN_First()
    optimizer=optim.SGD(net.parameters(),lr=0.01,momentum=0.7)
    optimizer.zero_grad()
    for epoch in range(5000):
        running_loss=0
        for i in range(len(files)):
            netout=net(inputs[i])
            loss=loss_func_mul(netout,corr_ans[i],legal_masks[i])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            running_loss+=loss.item()
    
        if epoch%10==0:
            netout=net(check_input)
            check_loss=loss_func_mul(netout,check_target,check_mask)
            check_corr=correct_num(netout,check_target,check_mask)/len(check_target)
            log("%3d finish. %f %f %f"%(epoch,running_loss/(i+1),check_loss,check_corr))

if __name__=="__main__":
    train_first_adv()
    #train_first()
    #parse_train_data("./Greed_batch/Greed_batch1.txt")
    #log(gen_legal_onehot([0,'C5'],torch.zeros(52)))
    
"""
savefilename=file.split('.')
savefilename[-1]='4train'
with open('.'.join(savefilename),'wb') as f2:
    pickle.dump(train_data,f2)
with open("./Greed_batch/Greed_batch1.4train",'rb') as f3:
    train_data=pickle.load(f3)
"""