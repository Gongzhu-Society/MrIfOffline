#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
from MrRandom import MrRandom
from MrGreed import MrGreed
import torch
from MrO_Trainer import cards_in_hand_oh,score_oh,four_cards_oh,MrO_Net


class MrO(MrRandom):
    """
        O is the next letter of N in MrNN.
    """
    N_SAMPLE=10

    def __init__(self,netpara,room=0,place=0,name="default_o"):
        super(MrO,self).__init__(room,place,name)
        self.net=torch.load(netpara)
        self.device=torch.device("cuda:1")
        self.net.to(self.device)

    def pick_a_card(self):
        #确认桌上牌的数量和自己坐的位置相符
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place
        #general data
        suit=self.decide_suit()
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        d_legal={c:0 for c in MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)} #dict of legal choice
        #data for sampling
        void_info=MrGreed.gen_void_info(self.place,self.history,self.cards_on_table)
        cards_remain=MrGreed.calc_cards_remain(self.history,self.cards_on_table,self.cards_list)
        void_info=MrGreed.gen_void_info(self.place,self.history,self.cards_on_table)
        if len(self.cards_on_table)==4:
            lens=[len(self.cards_list)-1,len(self.cards_list)-1,len(self.cards_list)-1]
        elif len(self.cards_on_table)==3:
            lens=[len(self.cards_list),len(self.cards_list)-1,len(self.cards_list)-1]
        elif len(self.cards_on_table)==2:
            lens=[len(self.cards_list),len(self.cards_list),len(self.cards_list)-1]
        elif len(self.cards_on_table)==1:
            lens=[len(self.cards_list),len(self.cards_list),len(self.cards_list)]
        expire_date=0 #for sampling
        for _ in range(MrO.N_SAMPLE):
            #sampling
            if expire_date==0:
                cards_list_list,exchange_info,bx=MrGreed.gen_scenario(void_info,cards_remain,lens)
                expire_date=max(bx-5,0)
            else:
                exhausted_flag=MrGreed.alter_scenario(cards_list_list,exchange_info,void_info)
                if exhausted_flag==1:
                    break
                expire_date-=1
            #decide
            for c in d_legal:
                netin=[]
                netin.append(cards_in_hand_oh(cards_list_list,self.cards_list))
                netin.append(four_cards_oh(self.cards_on_table,c))
                netin.append(score_oh(self.score,self.place))
                with torch.no_grad():
                    score=self.net(torch.cat(netin).float().to(self.device))
                d_legal[c]+=score.item()
        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice

    @staticmethod
    def family_name():
        return 'MrO'

def try_mro():
    from MrIf import MrIf
    from OfflineInterface import OfflineInterface
    import itertools,numpy
    #netpara="MrO_Net_9_164609.pkl"
    netpara="MrO_Net_15_395265.pkl"
    o0=MrO(netpara,room=0,place=0,name='o0')
    o2=MrO(netpara,room=0,place=2,name='o2')
    f1=MrIf(room=0,place=1,name="if1")
    f3=MrIf(room=0,place=3,name="if3")
    offlineinterface=OfflineInterface([o0,f1,o2,f3],print_flag=False)
    N1=64;N2=2
    stats=[]
    for k,l in itertools.product(range(N1),range(N2)):
        if l==0:
            cards=offlineinterface.shuffle()
        else:
            cards=cards[39:52]+cards[0:39]
            offlineinterface.shuffle(cards=cards)
        for i,j in itertools.product(range(13),range(4)):
            offlineinterface.step()
        stats.append(offlineinterface.clear())
        if l==N2-1:
            print("%4d"%(sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])),end=" ",flush=True)
        offlineinterface.prepare_new()
    s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    print("%.2f %.2f"%(numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1)),))

if __name__=="__main__":
    try_mro()
