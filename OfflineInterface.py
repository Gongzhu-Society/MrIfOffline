#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from Util import log,calc_score,cards_order
from Util import ORDER_DICT2,SCORE_DICT

import random,itertools,numpy,copy,time

class OfflineInterface():
    """ONLY for 4 players"""
    def __init__(self,players,print_flag=True):
        """
            players: list of robots or humans, each of them is a instace of MrRandom, Human, MrIf, etc.
        """
        self.players=players
        self.print_flag=print_flag

        self.pstart=0                     #index of start player
        self.pnext=self.pstart            #index of next player
        self.cards_on_table=[self.pnext,] #see MrRandom.cards_on_table
        self.history=[]                   #see MrRandom.history
        self.scores=[[],[],[],[]]
        #self.cards_remain should be initialized by self.shuffle
        #self.cards_remain=[[],[],[],[]]

    def shuffle(self,cards=None):
        if cards==None:
            cards=['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'SJ', 'SQ', 'SK', 'SA',
                   'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'HJ', 'HQ', 'HK', 'HA',
                   'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'DJ', 'DQ', 'DK', 'DA',
                   'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'CJ', 'CQ', 'CK', 'CA']
            random.shuffle(cards)
        else:
            assert len(cards)==52, "len of 'cards' should equal 52 if you want to assign rather than shuffle"
        self.cards_remain=[]
        for i in range(4):
            list_temp=cards[i*13:i*13+13]
            #list_temp.sort(key=cards_order) #will sort influence performance? 2021 Feb 7th
            self.cards_remain.append(list_temp)
            #seperate player and judge by copy.copy
            self.players[i].cards_list=copy.copy(list_temp)
        if self.print_flag:
            log("shuffle: %s"%(self.cards_remain))
        return cards

    def judge_legal(self,choice):
        if choice not in self.cards_remain[self.pnext]:
            return False
        if len(self.cards_on_table)==1 or choice[0]==self.cards_on_table[1][0]:
            return True
        for i in self.cards_remain[self.pnext]:
            if i[0]==self.cards_on_table[1][0]:
                return False
        else:
            return True

    def step(self):
        #初始化玩家
        self.players[self.pnext].cards_on_table=copy.copy(self.cards_on_table)
        self.players[self.pnext].history=copy.deepcopy(self.history)
        self.players[self.pnext].scores=copy.deepcopy(self.scores)
        #get this player's choice
        choice=self.players[self.pnext].pick_a_card()
        if not self.judge_legal(choice):
            log("choice %s, %s illegal"%(choice,self.cards_on_table),l=2)
            input()
            return 1
        self.cards_remain[self.pnext].remove(choice)
        self.players[self.pnext].cards_list.remove(choice)
        self.cards_on_table.append(choice)
        if self.print_flag:
            log("%s played %s, %s"%(self.players[self.pnext].name,choice,self.cards_on_table,))
        #如果一墩结束
        if len(self.cards_on_table)==5:
            #判断赢家
            winner=1
            score_temp=ORDER_DICT2[self.cards_on_table[1][1]]
            if self.cards_on_table[2][0]==self.cards_on_table[1][0]\
            and ORDER_DICT2[self.cards_on_table[2][1]]>score_temp:
                winner=2
                score_temp=ORDER_DICT2[self.cards_on_table[2][1]]
            if self.cards_on_table[3][0]==self.cards_on_table[1][0]\
            and ORDER_DICT2[self.cards_on_table[3][1]]>score_temp:
                winner=3
                score_temp=ORDER_DICT2[self.cards_on_table[3][1]]
            if self.cards_on_table[4][0]==self.cards_on_table[1][0]\
            and ORDER_DICT2[self.cards_on_table[4][1]]>score_temp:
                winner=4
            self.pnext=(winner+self.pnext)%4
            #结算有分的牌
            for i in self.cards_on_table[1:]:
                if i in SCORE_DICT:
                    self.scores[self.pnext].append(i)
            #更新数据结构
            self.history.append(copy.copy(self.cards_on_table))
            self.cards_on_table=[self.pnext,]
            if self.print_flag:
                log("trick end. winner is %s, %s"%(self.pnext,self.scores))
        else:
            self.pnext=(self.pnext+1)%4
        return 0

    def step_complete_info(self):
        #初始化玩家
        self.players[self.pnext].cards_on_table=copy.copy(self.cards_on_table)
        self.players[self.pnext].history=copy.deepcopy(self.history)
        self.players[self.pnext].scores=copy.deepcopy(self.scores)
        self.players[self.pnext].cards_remain=copy.deepcopy(self.cards_remain)
        #get this player's choice
        choice=self.players[self.pnext].pick_a_card_complete_info()
        if not self.judge_legal(choice):
            raise Exception("What's wrong with your agency?")
        self.cards_remain[self.pnext].remove(choice)
        self.players[self.pnext].cards_list.remove(choice)
        self.cards_on_table.append(choice)
        #如果一墩结束
        if len(self.cards_on_table)==5:
            #判断赢家
            winner=1
            score_temp=ORDER_DICT2[self.cards_on_table[1][1]]
            if self.cards_on_table[2][0]==self.cards_on_table[1][0]\
            and ORDER_DICT2[self.cards_on_table[2][1]]>score_temp:
                winner=2
                score_temp=ORDER_DICT2[self.cards_on_table[2][1]]
            if self.cards_on_table[3][0]==self.cards_on_table[1][0]\
            and ORDER_DICT2[self.cards_on_table[3][1]]>score_temp:
                winner=3
                score_temp=ORDER_DICT2[self.cards_on_table[3][1]]
            if self.cards_on_table[4][0]==self.cards_on_table[1][0]\
            and ORDER_DICT2[self.cards_on_table[4][1]]>score_temp:
                winner=4
            self.pnext=(winner+self.pnext)%4
            #结算有分的牌
            for i in self.cards_on_table[1:]:
                if i in SCORE_DICT:
                    self.scores[self.pnext].append(i)
            #更新数据结构
            self.history.append(copy.copy(self.cards_on_table))
            self.cards_on_table=[self.pnext,]
        else:
            self.pnext=(self.pnext+1)%4
        return 0

    def clear(self):
        self.scores_num=[calc_score(i) for i in self.scores]
        """scores_temp=copy.copy(self.scores_num)
        c10=(i for i in range(4) if 'C10' in self.scores[i]).__next__()
        if len(self.scores[c10])==1:
            scores_temp[c10]=0
        else:
            scores_temp[c10]/=2
        try:
            assert sum(scores_temp)==-200
        except:
            log("clear score: %s, %s"%(self.scores,self.scores_num))"""
        if self.print_flag:
            log("game end: %s, %s"%(self.scores_num,self.scores))
        return self.scores_num

    def prepare_new(self):
        self.pstart=(self.pstart+1)%4
        self.pnext=self.pstart
        self.cards_on_table=[self.pnext,]
        self.scores=[[],[],[],[]]
        self.history=[]
        del self.scores_num
        del self.cards_remain

def gen_shuffle(num,perfix):
    cards=['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'SJ', 'SQ', 'SK', 'SA',
           'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'HJ', 'HQ', 'HK', 'HA',
           'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'DJ', 'DQ', 'DK', 'DA',
           'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'CJ', 'CQ', 'CK', 'CA']
    logfile="StdHands/%s_%d.hands"%(perfix,num)
    for i in range(num):
        random.shuffle(cards)
        log("No.%04d: %s"%(i,cards),logfile=logfile,fileonly=True)
        
def read_std_hands(filename):
    import re
    from ast import literal_eval
    p_cards=re.compile("No\\.[0-9]+: (.+)")
    f=open(filename,"r")
    stdhands=[]
    for l in f:
        s=p_cards.search(l)
        if s==None:
            log("failed to parse:\n%s"%(l),end="")
        else:
            stdhands.append(s.group(1))
    f.close()
    stdhands=[literal_eval(l) for l in stdhands]
    log("parsed %d hands, start with:\n%s"%(len(stdhands),stdhands[0]))
    return stdhands

if __name__=="__main__":
    gen_shuffle(1024,"random")
    read_std_hands("StdHands/random_0_1024.hands")