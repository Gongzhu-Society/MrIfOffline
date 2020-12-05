#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
from Util import ORDER_DICT2,SCORE_DICT
from MrRandom import MrRandom
from MrGreed import MrGreed
from ScenarioGen import ScenarioGen
from MCTS.mcts import mcts
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy,itertools,numpy,time

print_level=0

class NN_PV(nn.Module):
    """
        输入当前局面, 返回评估分数
    """
    def __init__(self):
        super(MrO_Last,self).__init__()
        self.fc0=nn.Linear(52*7+16*4*2,1024)
        self.fc1=nn.Linear(1024,1024)
        self.fc2=nn.Linear(1024,512)


        self.avgp=torch.nn.AvgPool1d(2)

    def forward(self, x):
        x=F.relu(self.fc0(x))
        x=F.relu(self.fc1(x))+x
        x=F.relu(self.fce(x))+self.avgp(x.view(-1,1,128)).view(-1,64)
        x=self.fcf(x)
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
            if 'weight' in child.state_dict():
                #stru.append(tuple(child.state_dict()['weight'].t().size()))
                stru.append(child.state_dict()['weight'].shape)
        return "%s %s %s"%(self.__class__.__name__,stru,self.num_paras())

class GameState():
    def __init__(self,cards_lists,score_lists,cards_on_table,play_for):
        self.cards_lists=cards_lists
        self.cards_on_table=cards_on_table
        self.score_lists=score_lists
        self.play_for=play_for

        #decide cards_dicts, suit and pnext
        self.cards_dicts=[MrGreed.gen_cards_dict(i) for i in self.cards_lists]
        if len(self.cards_on_table)==1:
            self.suit="A"
        else:
            self.suit=self.cards_on_table[1][0]
        self.pnext=(self.cards_on_table[0]+len(self.cards_on_table)-1)%4
        self.remain_card_num=sum([len(i) for i in self.cards_lists])

    def getCurrentPlayer(self):
        if (self.pnext-self.play_for)%2==0:
            return 1
        else:
            return -1

    def getPossibleActions(self):
        return MrGreed.gen_legal_choice(self.suit,self.cards_dicts[self.pnext],self.cards_lists[self.pnext])

    def takeAction(self,action):
        #log(action)
        neo_state=copy.deepcopy(self)
        neo_state.cards_lists[neo_state.pnext].remove(action)
        neo_state.cards_dicts[neo_state.pnext][action[0]].remove(action)
        neo_state.remain_card_num-=1
        neo_state.cards_on_table.append(action)
        #log(neo_state.cards_on_table)
        #input()
        assert len(neo_state.cards_on_table)<=5
        if len(neo_state.cards_on_table)<5:
            neo_state.pnext=(neo_state.pnext+1)%4
            if len(neo_state.cards_on_table)==2:
                neo_state.suit=neo_state.cards_on_table[1][0]
        else:
            #decide pnext
            score_temp=-1024
            for i in range(4):
                if neo_state.cards_on_table[i+1][0]==neo_state.cards_on_table[1][0] and ORDER_DICT2[neo_state.cards_on_table[i+1][1]]>score_temp:
                    winner=i #in relative order
                    score_temp=ORDER_DICT2[neo_state.cards_on_table[i+1][1]]
            neo_state.pnext=(neo_state.cards_on_table[0]+winner)%4
            #clear scores
            neo_state.score_lists[neo_state.pnext]+=[c for c in neo_state.cards_on_table[1:] if c in SCORE_DICT]
            #clean table
            neo_state.cards_on_table=[neo_state.pnext,]
            neo_state.suit='A'
        return neo_state

    def isTerminal(self):
        if self.remain_card_num==0:
            return True
        else:
            return False

    def getReward(self):
        scores=[MrRandTree.clear_fmt_score(self.fmt_scores[(self.play_for+i)%4]) for i in range(4)]
        return scores[0]+scores[2]-scores[1]-scores[3]

class MrZeroTree():
    def cards_lists_oh(cards_lists,place):
        """
            return a 208-length one hot, in raletive order
            the order is [me,me+1,me+2,me+3]
        """
        oh=torch.zeros(52*4,dtype=torch.uint8)
        for i in range(4):
            for c in cards_lists[(place+i)%4]:
                oh[52*i+ORDER_DICT[c]]=1
        return oh

    def four_cards_oh(cards_on_table,place):
        """
            return a 156-legth oh, in anti-relative order
            the order is [me-1,me-2,me-3]
        """
        assert (cards_on_table[0]+len(cards_on_table)-1)%4==place
        oh=torch.zeros(52*3,dtype=torch.uint8)
        for c,i in enumerate(cards_on_table[:0:-1]):
            oh[52*i+ORDER_DICT[c]]=1
        return oh

    def score_lists_oh(score_lists,place):
        """
            return a 64-length one hot, in relative order
            the order is [me,me+1,me+2,me+3]
        """
        oh=torch.zeros(16*4,dtype=torch.uint8)
        for i in range(4):
            for c in score[(place+i)%4]:
                oh[52*i+ORDER_DICT5[c]]=1
        return oh

    def prepare_ohs(cards_lists,cards_on_table,score_lists,place):
        """
            double the time of four_cards for it to focus
        """
        oh_card=MrZeroTree.cards_lists_oh(cards_lists,place)
        oh_score=MrZeroTree.score_lists_oh(score_lists,place)
        oh_table=MrZeroTree.four_cards_oh(cards_on_table,place)
        return torch.cat([oh_card,oh_score,oh_table,oh_table])

    def pick_a_card(self,train_mode=False):
        #确认桌上牌的数量和自己坐的位置相符
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place
        #utility datas
        suit=self.decide_suit() #inherited from MrRandom
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        #如果别无选择
        if cards_dict.get(suit)!=None and len(cards_dict[suit])==1:
            choice=cards_dict[suit][0]
            return choice

        legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)
        d_legal={c:0 for c in legal_choice} #dict of legal choice
        sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=MrRandTree.N_SAMPLE,METHOD1_PREFERENCE=100)
        for cards_list_list in sce_gen:
            cards_lists=[None,None,None,None]
            cards_lists[self.place]=copy.copy(self.cards_list)
            for i in range(3):
                cards_lists[(self.place+i+1)%4]=cards_list_list[i]
            score_lists=copy.deepcopy(self.score)
            #initialize gamestate
            gamestate=GameState(cards_lists,score_lists,self.cards_on_table,self.place)
            searcher=mcts(iterationLimit=200,explorationConstant=200)
            searcher.search(initialState=gamestate,needNodeValue=False)
            for action,node in searcher.root.children.items():
                d_legal[action]+=node.totalReward/node.numVisits
                d_legal_temp[action]=node.totalReward/node.numVisits
            if train_mode:
                best_score=float("-inf")
                for k,v in d_legal_temp.items():
                    if v>best_score:
                        best_score=v
                        best_choices=[k]
                    elif v==best_score:
                        best_choices.append(k)
                target=torch.zeros(52+1)
                for c in best_choices:
                    target[ORDER_DICT[c]]=1/len(best_choices)
                target[52]=best_score
                netin=MrZeroTree.prepare_ohs(cards_lists,self.cards_on_table,score_lists,self.place)
                self.train_datas.append([netin,target])
        if print_level>=1:
            log("d_legal: %s"%(d_legal))
            input("press any key to continue...")
        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice

