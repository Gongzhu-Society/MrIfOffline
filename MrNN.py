#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import ORDER_DICT,ORDER_DICT5,ORDER_DICT4,INIT_CARDS
from MrRandom import MrRandom
from MrNN_Trainer import NN_First,NN_Second,NN_Third,NN_Last
import torch
import torch.nn.functional as F

print_level=0

class MrNN(MrRandom):
    def to(self,name):
        device=torch.device(name)
        for n in self.nets:
            n.to(device)

    def prepare_net(self,infos):
        """
            infos: a list of [NetClass,File]
        """
        self.nets=[]
        for NetClass,FileName in infos:
            net_temp=NetClass()
            net_temp.load_state_dict(torch.load(FileName))
            self.nets.append(net_temp)
            #log("load %s from %s"%(NetClass.__name__,FileName))

    def gen_void_info(self):
        """
            获得端门信息, i.e. void_ll, 其中 ll 是 list of list 的意思.
            比如 (1,0,1,0) 意思是黑桃方片断了, 此顺序由 Util.py 中的 ORDER_DICT4 定义
        """
        void_ll=[torch.zeros(4),torch.zeros(4),torch.zeros(4),torch.zeros(4)]
        for h in self.history:
            for i,c in enumerate(h[2:]):
                seat=(h[0]+i+1)%4
                if c[0]!=h[1][0]:
                    void_ll[seat][ORDER_DICT4[h[1][0]]]=1
        for i,c in enumerate(self.cards_on_table[2:]):
            seat=(self.cards_on_table[0]+i+1)%4
            if c[0]!=self.cards_on_table[1][0]:
                void_ll[seat][ORDER_DICT4[self.cards_on_table[1][0]]]=1
        return void_ll

    def gen_mycards_oh(self):
        """
            获得自己手牌的 one_hot 编码
        """
        mycards_oh=torch.zeros(52,dtype=torch.uint8)
        for c in self.cards_list:
            mycards_oh[ORDER_DICT[c]]=1
        return mycards_oh

    def gen_legal_mask(self,mycards_oh):
        if len(self.cards_on_table)>1:
            c_num=ORDER_DICT4[self.cards_on_table[1][0]]
            mask=torch.tensor([0]*(c_num*13)+[1]*(13)+[0]*((3-c_num)*13),dtype=torch.uint8)
            mask=mask*mycards_oh
            if mask.any():
                return mask
            else:
                return mycards_oh
        else:
            return mycards_oh

    def pick_a_card(self):
        seat=len(self.cards_on_table)-1
        #log("my(%s) turn: %s %s"%(self.name,self.cards_on_table,self.cards_list))
        mycards_oh=self.gen_mycards_oh()
        legal_mask=self.gen_legal_mask(mycards_oh)
        #生成别人的手牌
        othercards_oh=torch.ones(52,dtype=torch.uint8)
        for h in self.history:
            for c in h[1:]:
                othercards_oh[ORDER_DICT[c]]=0
        for c in self.cards_on_table[1:]:
            othercards_oh[ORDER_DICT[c]]=0
        othercards_oh-=mycards_oh
        assert (othercards_oh<=1).all()
        #log("get othercards: %s"%(othercards_oh))

        to_input=[mycards_oh,othercards_oh]
        for i in self.cards_on_table[1:]:
            to_input.append(torch.zeros(52,dtype=torch.uint8))
            to_input[-1][ORDER_DICT[i]]=1

        void_ll=self.gen_void_info()
        #log("get void_ll: %s"%(void_ll))
        for i in range(4):
            temp=(i+self.cards_on_table[0])%4
            to_input.append(torch.zeros(16))
            for c in self.scores[temp]:
                to_input[-1][ORDER_DICT5[c]]=1
            to_input.append(void_ll[temp])

        to_input=torch.cat(to_input)
        with torch.no_grad():
            netout=self.nets[seat](to_input)
            netout+=(legal_mask*(2*netout.abs().max())+2)
            netout=F.softmax(netout,dim=0)
            netout*=legal_mask
            netout/=netout.sum()
        _,max_i=torch.max(netout,dim=0)
        #log("%d: %f"%(max_i,netout[max_i]))
        return INIT_CARDS[max_i]

    @staticmethod
    def family_name():
        return 'MrNN'

def test_legal_mask():
    n0=MrNN(room=0,place=0,name="n0")
    n0.cards_list=['S4', 'S6', 'S9', 'SA', 'H7', 'HK', 'DJ', 'C7', 'C8', 'CQ']
    n0.cards_on_table=[3,'H8']
    mycards_oh=n0.gen_mycards_oh()
    log("get mycards_oh: %s"%(mycards_oh))
    legal_mask=n0.gen_legal_mask(mycards_oh)
    log("get legal_mask: %s"%(legal_mask))

def test_with_c10():
    para_dir="./NetPara20200715/"
    n0=MrNN(room=0,place=0,name="n0")
    n0.prepare_net([(NN_First,para_dir+'NN_First_11_121012.ckpt'),
                    (NN_Second,para_dir+'NN_Second_9_126004.ckpt'),
                    (NN_Third,para_dir+'NN_Third_7_130996.ckpt'),
                    (NN_Last,para_dir+'NN_Last_5_135988.ckpt')])
    n0.cards_list=['C3','CA','H2','H3','H4','H5','H6','H7','H8','H9','H10']
    n0.history=[(2,'D3','D4','DJ','D6'),(1,'S2','HJ','SQ','S5')]
    n0.cards_on_table=[2, 'C2', 'C10']
    n0.scores=[['DJ'],[],['HJ'],['SQ','HA','HQ','HK']]
    log(n0.pick_a_card())

if __name__=="__main__":
    test_with_c10()
    #test_legal_mask()