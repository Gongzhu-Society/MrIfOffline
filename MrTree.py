#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from MrIf import LOGFILE,log,MrRandom,MrIf,cards_order
from OfflineInterface import calc_score
import copy,random,numpy

INIT_CARDS=[ 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'SJ', 'SQ', 'SK', 'SA', 
            'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'HJ', 'HQ', 'HK', 'HA', 
            'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'DJ', 'DQ', 'DK', 'DA', 
            'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'CJ', 'CQ', 'CK', 'CA']

class MrTree(MrRandom):
    def pick_a_card(self):
        if len(self.cards_on_table)==1:
            suit="A"
        else:
            suit=self.cards_on_table[1][0]
        #计算别人有什么牌
        cards_remain=copy.copy(INIT_CARDS)
        for h in self.history:
            for c in h[1:5]:
                cards_remain.remove(c)
        for c in self.cards_on_table[1:]:
            cards_remain.remove(c)
        for c in self.cards_list:
            cards_remain.remove(c)
        deep=len(cards_remain)+len(self.cards_list)
        #模拟可能的情况
        for i in range(100):
            random.shuffle(cards_remain)
            s=Scenario(deep,True)
            s.cards_remain=[copy.copy(self.cards_list),cards[13:26],cards[26:39],cards[39:52]]
            s.score_cards=copy.deepcopy(self.score_cards)
            s.cards_on_table=copy.copy(self.cards_on_table[1:])
            s.pnext=0
            s.suit=suit

class Scenario():
    SCORE_DICT={'SQ':-100,'DJ':100,'C10':0,
            'H2':0,'H3':0,'H4':0,'H5':-10,'H6':-10,'H7':-10,'H8':-10,'H9':-10,'H10':-10,
            'HJ':-20,'HQ':-30,'HK':-40,'HA':-50,'JP':-60,'JG':-70}
    ORDER_DICT1={'S':-300,'H':-200,'D':-100,'C':0,'J':-200}
    ORDER_DICT2={'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'1':10,'J':11,'Q':12,'K':13,'A':14,'P':15,'G':16}
    
    def __init__(self,deep,max_score_flag):
        #self.cards_remain=cards_remain
        #self.score_cards=score_cards
        #self.cards_on_table=cards_on_table
        #self.pnext=pnext
        #self.suit=suit
        self.deep=deep 
        self.max_score_flag=max_score_flag       
        self.descendants=[]
        self.mcts_w=0
        self.mcts_n=0

    def random_play(self):
        cards_remain_tmp=copy.deepcopy(self.cards_remain)
        score_cards_tmp=copy.deepcopy(self.score_cards)
        cards_on_table_tmp=copy.copy(self.cards_on_table)
        pnext_tmp=self.pnext
        suit_tmp=self.suit
        for i in range(self.deep):
            #log("%d's round, %s, %s\n%s"%(pnext_tmp,suit_tmp,cards_on_table_tmp,cards_remain_tmp))
            #input()
            if suit_tmp!="A":
                legal_choice=[]
                for i,c in enumerate(cards_remain_tmp[pnext_tmp]):
                    if c[0]==suit_tmp:
                        legal_choice.append(i)
                if len(legal_choice)==0:
                    cho_num=random.randint(0,len(cards_remain_tmp[pnext_tmp])-1)
                else:
                    cho_num=random.choice(legal_choice)
            else:
                cho_num=random.randint(0,len(cards_remain_tmp[pnext_tmp])-1)
            choice=cards_remain_tmp[pnext_tmp].pop(cho_num)
            if len(cards_on_table_tmp)==3:
                winner=1
                score_temp=Scenario.ORDER_DICT2[cards_on_table_tmp[0][1]]
                if cards_on_table_tmp[1][0]==suit_tmp\
                and Scenario.ORDER_DICT2[cards_on_table_tmp[1][1]]>score_temp:
                    winner=2
                    score_temp=Scenario.ORDER_DICT2[cards_on_table_tmp[1][1]]
                if cards_on_table_tmp[2][0]==suit_tmp\
                and Scenario.ORDER_DICT2[cards_on_table_tmp[2][1]]>score_temp:
                    winner=3
                    score_temp=Scenario.ORDER_DICT2[cards_on_table_tmp[2][1]]
                if choice[0]==suit_tmp\
                and Scenario.ORDER_DICT2[choice[1]]>score_temp:
                    winner=4
                pnext_tmp=(winner+pnext_tmp)%4
                for i in cards_on_table_tmp:
                    if i in Scenario.SCORE_DICT:
                        score_cards_tmp[pnext_tmp].append(i)
                if choice in Scenario.SCORE_DICT:
                    score_cards_tmp[pnext_tmp].append(choice)
                cards_on_table_tmp=[]
                suit_tmp="A"
                #log("clear winner is %d, %s"%(pnext_tmp,score_cards_tmp))
            elif len(cards_on_table_tmp)==0:
                cards_on_table_tmp.append(choice)
                pnext_tmp=(pnext_tmp+1)%4
                suit_tmp=cards_on_table_tmp[0][0]
            else:
                cards_on_table_tmp.append(choice)
                pnext_tmp=(pnext_tmp+1)%4
        scores_num=calc_score(score_cards_tmp[0])+calc_score(score_cards_tmp[2])\
                  -calc_score(score_cards_tmp[1])-calc_score(score_cards_tmp[3])
        self.mcts_w+=scores_num
        self.mcts_n+=1
        return scores_num

    def give_birth(self,choice_num):
        #global total_num_ct
        #total_num_ct+=1
        #if total_num_ct%100000==0:
        #    log("total_num_ct reached %d"%(total_num_ct,))
        d_temp=Scenario(self.deep-1,not self.max_score_flag)
        self.descendants.append(d_temp)
        d_temp.cards_remain=copy.deepcopy(self.cards_remain)
        choice=d_temp.cards_remain[self.pnext].pop(choice_num)
        d_temp.score_cards=copy.deepcopy(self.score_cards)
        if len(self.cards_on_table)==3:
            winner=1
            score_temp=Scenario.ORDER_DICT2[self.cards_on_table[0][1]]
            if self.cards_on_table[1][0]==self.suit\
            and Scenario.ORDER_DICT2[self.cards_on_table[1][1]]>score_temp:
                winner=2
                score_temp=Scenario.ORDER_DICT2[self.cards_on_table[1][1]]
            if self.cards_on_table[2][0]==self.suit\
            and Scenario.ORDER_DICT2[self.cards_on_table[2][1]]>score_temp:
                winner=3
                score_temp=Scenario.ORDER_DICT2[self.cards_on_table[2][1]]
            if choice[0]==self.suit\
            and Scenario.ORDER_DICT2[choice[1]]>score_temp:
                winner=4
            d_temp.pnext=(winner+self.pnext)%4
            for i in self.cards_on_table:
                if i in Scenario.SCORE_DICT:
                    d_temp.score_cards[d_temp.pnext].append(i)
            if choice in Scenario.SCORE_DICT:
                d_temp.score_cards[d_temp.pnext].append(choice)
            d_temp.suit="A"
            d_temp.cards_on_table=[]
        else:
            d_temp.pnext=(self.pnext+1)%4
            d_temp.cards_on_table=copy.copy(self.cards_on_table)
            d_temp.cards_on_table.append(choice)
            if len(self.cards_on_table)==0:
                d_temp.suit=d_temp.cards_on_table[0][0]
            else:
                d_temp.suit=self.suit
        #log("me, at deep=%d, gave birth to: %s"%(self.deep,d_temp))
        #input()

    def breed(self):
        nochoice_flag=True
        if self.suit!="A":
            for i,c in enumerate(self.cards_remain[self.pnext]):
                if c[0]==self.suit:
                    nochoice_flag=False
                    self.give_birth(i)
            if nochoice_flag:
                for i in range(len(self.cards_remain[self.pnext])):
                    self.give_birth(i)
        else:
            for i in range(len(self.cards_remain[self.pnext])):
                self.give_birth(i)

    def uct(self,lnt):
        if self.max_score_flag:
            return self.mcts_w/self.mcts_n+100*numpy.sqrt(lnt/self.mcts_n)
        else:
            return -1*self.mcts_w/self.mcts_n+100*numpy.sqrt(lnt/self.mcts_n)

    def get_best_child(self):
        lnt=numpy.log(self.mcts_n)
        d_best=self.descendants[0]
        d_best_val=self.descendants[0].uct(lnt)
        for d in self.descendants[1:]:
            temp_val=d.uct(lnt)
            if temp_val>d_best_val:
                d_best=d
                d_best_val=temp_val
        return d_best

    def mcts(self):
        if len(self.descendants)==0:
            self.breed()
            for d in self.descendants:
                self.mcts_w+=d.random_play()
                self.mcts_n+=1
            #if self.deep==48:
            #    log("finish first trick")
            #log("get to bottom deep=%d mcts_w=%d mcts_n=%d"%(self.deep,self.mcts_w,self.mcts_n))
            return self.mcts_w,self.mcts_n
        dw,dn=self.get_best_child().mcts()
        self.mcts_w+=dw
        self.mcts_n+=dn
        #log("mcts back prop dw=%d dn=%d"%(dw,dn))
        return dw,dn

    def __str__(self):
        return str(self.__dict__)

def test_breed():
    cards=[ 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'SJ', 'SQ', 'SK', 'SA', 
            'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'HJ', 'HQ', 'HK', 'HA', 
            'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'DJ', 'DQ', 'DK', 'DA', 
            'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'CJ', 'CQ', 'CK', 'CA']
    random.shuffle(cards)
    s=Scenario(52)
    s.cards_remain=[cards[0:13],cards[13:26],cards[26:39],cards[39:52]]
    for i in s.cards_remain:
        i.sort(key=cards_order)
    s.score_cards=[[],[],[],[]]
    s.cards_on_table=[]
    s.pnext=0
    s.suit="A"
    log("create a scenario: %s"%(s))
    s.breed()
    log("origin scenario: %s"%(s))
    log("total_num_ct: %d"%(total_num_ct))

def test_random_play():
    cards=[ 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'SJ', 'SQ', 'SK', 'SA', 
            'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'HJ', 'HQ', 'HK', 'HA', 
            'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'DJ', 'DQ', 'DK', 'DA', 
            'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'CJ', 'CQ', 'CK', 'CA']
    random.shuffle(cards)
    s=Scenario(52)
    s.cards_remain=[cards[0:13],cards[13:26],cards[26:39],cards[39:52]]
    for i in s.cards_remain:
        i.sort(key=cards_order)
    s.score_cards=[[],[],[],[]]
    s.cards_on_table=[]
    s.pnext=0
    s.suit="A"
    log("begin")
    for i in range(5):
        s.random_play()
        log(s.__dict__)
    log("end")

def test_mcts():
    cards=[ 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'SJ', 'SQ', 'SK', 'SA', 
            'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'HJ', 'HQ', 'HK', 'HA', 
            'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'DJ', 'DQ', 'DK', 'DA', 
            'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'CJ', 'CQ', 'CK', 'CA']
    random.shuffle(cards)
    s=Scenario(52,True)
    s.cards_remain=[cards[0:13],cards[13:26],cards[26:39],cards[39:52]]
    for i in s.cards_remain:
        i.sort(key=cards_order)
    s.score_cards=[[],[],[],[]]
    s.cards_on_table=[]
    s.pnext=0
    s.suit="A"
    log(s)
    for i in range(100):
        s.mcts()
    log(s)
    log(s.get_best_child())

def stat_random_play():
    N=10000
    l=[]
    for i in range(N):
        l.append(test_random_play())
    log("%s, %s"%(numpy.mean(l),numpy.sqrt(numpy.var(l)/(N-1))))

if __name__=="__main__":
    test_mcts()
    #test_breed()
    #test_random_play()
    #stat_random_play()

"""
全展开情况笔记，从52开始，随机发牌
展开层数 1   2   3   4   5    8      9       12
情况数   13  53  159 466 4042 170689 1564813
"""