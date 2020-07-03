#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from MrIf import LOGFILE,log,cards_order,MrRandom,Human,MrIf
import random,itertools,numpy

SCORE_DICT={'SQ':-100,'DJ':100,'C10':0,
            'H2':0,'H3':0,'H4':0,'H5':-10,'H6':-10,'H7':-10,'H8':-10,'H9':-10,'H10':-10,
            'HJ':-20,'HQ':-30,'HK':-40,'HA':-50,'JP':-60,'JG':-70}
def calc_score(l):
    s=0
    has_score_flag=False
    c10_flag=False
    for i in l:
        if i=="C10":
            c10_flag=True
        else:
            s+=SCORE_DICT[i]
            has_score_flag=True
    if c10_flag==True:
        if has_score_flag==False: 
            s+=50
        else:
            s*=2
    return s

class OfflineInterface():
    """ONLY for 4 players"""
    def __init__(self,players):
        """players: list of robots or humans(represent by None)"""
        self.players=players
        self.pstart=0
        self.pnext=self.pstart #index of next player
        self.cards_on_table=[self.pnext,]
        self.scores = [[],[],[],[]]

    def shuffle(self,cards=None):
        if cards==None:
            cards=['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'SJ', 'SQ', 'SK', 'SA', 
                   'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'HJ', 'HQ', 'HK', 'HA', 
                   'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'DJ', 'DQ', 'DK', 'DA', 
                   'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'CJ', 'CQ', 'CK', 'CA']
            for i in range(4):
                random.shuffle(cards)
        else:
            assert len(cards)==52
        for i in range(4):
            shuffle_list=cards[i*13:i*13+13]
            shuffle_list.sort(key=cards_order)
            #log("shufflling %s with %s"%(self.players[i].name,shuffle_list))
            #self.players[i].receive_shuffle(shuffle_list)
            self.players[i].cards_list=shuffle_list
        return cards

    def step(self):
        self.players[self.pnext].cards_on_table=self.cards_on_table
        choice=self.players[self.pnext].pick_a_card()
        self.cards_on_table.append(choice)
        #self.players[self.pnext].pop_card(choice)
        self.players[self.pnext].cards_list.remove(choice)
        #log("%s played %s"%(self.players[self.pnext].name,self.cards_on_table,))
        if len(self.cards_on_table)==5:
            pass
            #log("%s"%(self.cards_on_table,))
        if len(self.cards_on_table)<5:
            self.pnext=(self.pnext+1)%4
        else:
            winner=0
            score_temp=cards_order(self.cards_on_table[1])
            for i in range(1,4):
                if self.cards_on_table[i+1][0]==self.cards_on_table[1][0]:
                    score_temp2=cards_order(self.cards_on_table[i+1])
                    if score_temp2>score_temp:
                        winner=i
                        score_temp=score_temp2
            self.pnext=(self.cards_on_table[0]+winner)%4
            for i in self.cards_on_table[1:]:
                if i in SCORE_DICT:
                    self.scores[self.pnext].append(i)
            self.cards_on_table=[self.pnext,]
            #log("winner is %s, %s"%(self.pnext,self.scores))

    def clear(self):
        self.scores_num=[calc_score(i) for i in self.scores]
        log("game end: %s, %s"%(self.scores_num,self.scores))
        return self.scores_num #[0,100,-100,50]
    
    def prepare_new(self):
        self.pstart=(self.pstart+1)%4
        self.pnext=self.pstart
        self.cards_on_table=[self.pnext,]
        self.scores = [[],[],[],[]]
        del self.scores_num

def stat_random():
    random0=MrRandom(0,0,"random0")
    random1=MrRandom(0,1,"random1")
    random2=MrRandom(0,2,"random2")
    random3=MrRandom(0,3,"random3")
    if0=MrIf(0,0,"if0")
    if1=MrIf(0,1,"if1")
    if2=MrIf(0,2,"if2")
    if3=MrIf(0,2,"if3")
    offlineinterface=OfflineInterface([if0,random1,if2,random3])
    stats=[]
    for k,l in itertools.product(range(16),range(16)):
        if l==0:
            cards=offlineinterface.shuffle()
        else:
            cards=cards[13:52]+cards[0:13]
            offlineinterface.shuffle(cards=cards)
        for i,j in itertools.product(range(13),range(4)):
            offlineinterface.step()
        stats.append(offlineinterface.clear())
        offlineinterface.prepare_new()
    log(stats)
    for i in range(4):
        pi=[j[i] for j in stats]
        log("%.2f %.2f"%(numpy.mean(pi),numpy.sqrt(numpy.var(pi)),))


def play_with_if():
    if0=MrIf(0,0,"if0")
    if1=MrIf(0,1,"if1")
    if2=MrIf(0,2,"if2")
    if3=MrIf(0,2,"if3")
    myself=Human(0,3,"myself")
    offlineinterface=OfflineInterface([if0,if1,if2,if3])
    offlineinterface.shuffle()
    for i in range(13):
        for j in range(4):
            offlineinterface.step()
            input()
    offlineinterface.clear()

if __name__=="__main__":
    stat_random()
    #play_with_if()

"""
"""