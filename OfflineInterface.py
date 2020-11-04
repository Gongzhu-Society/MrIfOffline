#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from Util import log,calc_score,cards_order
from Util import ORDER_DICT2,SCORE_DICT

from MrRandom import MrRandom,Human
from MrIf import MrIf
from MrGreed import MrGreed
#from MrTree import MrTree
#from MrNN import MrNN
#from MrNN_Trainer import NN_First,NN_Second,NN_Third,NN_Last

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
            list_temp.sort(key=cards_order)
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
            log("choice %s, %s illegal"%(choice,self.cards_on_table))
            input()
            self.step()
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

    def clear(self):
        self.scores_num=[calc_score(i) for i in self.scores]
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

def stat_ai():
    r=[MrRandom(room=0,place=i,name="random%d"%(i)) for i in range(4)]
    f=[MrIf(room=0,place=i,name="if%d"%(i)) for i in range(4)]
    g=[MrGreed(room=0,place=i,name='greed%d'%(i)) for i in range(4)]
    #n=[MrNN(room=0,place=i,name='net%d'%(i)) for i in range(4)]
    #for i in n:
    #    i.prepare_net([(NN_First,'NN_First_11_121012.ckpt'),(NN_Second,'NN_Second_9_126004.ckpt'),(NN_Third,'NN_Third_7_130996.ckpt'),(NN_Last,'NN_Last_5_135988.ckpt')])
    offlineinterface=OfflineInterface([g[0],f[1],g[2],f[3]],print_flag=False)
    stats=[]
    N1=128;N2=2
    tik=time.time()
    for k,l in itertools.product(range(N1),range(N2)):
        if l==0:
            cards=offlineinterface.shuffle()
        else:
            cards=cards[39:52]+cards[0:39]
            offlineinterface.shuffle(cards=cards)
        for i,j in itertools.product(range(13),range(4)):
            offlineinterface.step()
        stats.append(offlineinterface.clear())
        log("%d, %d: %s"%(k,l,stats[-1]))
        offlineinterface.prepare_new()
    tok=time.time()
    log("time consume: %d"%(tok-tik))

    #statistic
    for i in range(4):
        s_temp=[j[i] for j in stats]
        log("%dth player: %.2f %.2f"%(i,numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1)),),l=2)
    s_temp=[j[0]+j[2] for j in stats]
    log("0 2 player: %.2f %.2f"%(numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1)),),l=2)
    s_temp=[j[1]+j[3] for j in stats]
    log("1 3 player: %.2f %.2f"%(numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1)),),l=2)
    s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    log(" 0+2 - 1+3: %.2f %.2f"%(numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1)),),l=2)

if __name__=="__main__":
    stat_ai()