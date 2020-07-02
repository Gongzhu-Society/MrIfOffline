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
            self.players[i].receive_shuffle(shuffle_list)
        return cards

    def step(self):
        if len(self.cards_on_table)==1:
            suit="A" #maybe A stands for any?
        else:
            suit=self.cards_on_table[1][0]
        self.players[self.pnext].cards_on_table=self.cards_on_table
        choice=self.players[self.pnext].pick_a_card(suit)
        self.cards_on_table.append(choice)
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
        return self.scores_num
    
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
    for k,l in itertools.product(range(128),range(16)):
        if l==0:
            cards=offlineinterface.shuffle()
        else:
            cards=cards[13:52]+cards[0:13]
            offlineinterface.shuffle(cards=cards)
        for i,j in itertools.product(range(13),range(4)):
            offlineinterface.step()
        stats.append(offlineinterface.clear())
        offlineinterface.prepare_new()
    #log(stats)
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
四个Mr. Random打1024x16局：
20/07/01 23:05:30 214 [INFO,stat_random:114] -64.01 110.97
20/07/01 23:05:30 217 [INFO,stat_random:114] -63.44 111.01
20/07/01 23:05:30 220 [INFO,stat_random:114] -62.29 109.91
20/07/01 23:05:30 223 [INFO,stat_random:114] -64.00 110.39

教会Mr. If先把短的花色打光再贴牌
    如果随便出
        从所剩张数少的花色开始，如果没有“危险牌”，就出这个花色大的
        尽量不出猪、猪圈、变压器、比变压器大的、红桃AK、羊
    如果是贴牌，按危险列表依次贴，没有危险列表了，贴短的
三个Mr. Random和一个Mr. If打128x16局：
20/07/02 12:30:33 370 [INFO,stat_random:116] -35.82 96.22
20/07/02 12:30:33 371 [INFO,stat_random:116] -75.61 117.02
20/07/02 12:30:33 372 [INFO,stat_random:116] -72.72 112.31
20/07/02 12:30:33 372 [INFO,stat_random:116] -73.02 112.53
两个Mr. Random和两个Mr. If打128x16局（邻）：
20/07/02 12:38:59 021 [INFO,stat_random:119] -42.78 95.74
20/07/02 12:38:59 022 [INFO,stat_random:119] -43.41 100.92
20/07/02 12:38:59 022 [INFO,stat_random:119] -90.96 125.14
20/07/02 12:38:59 023 [INFO,stat_random:119] -85.41 122.00
两个Mr. Random和两个Mr. If打128x16局（对）：
20/07/02 14:18:51 948 [INFO,stat_random:119] -42.74 99.39
20/07/02 14:18:51 949 [INFO,stat_random:119] -88.81 121.30
20/07/02 14:18:51 950 [INFO,stat_random:119] -44.93 97.94
20/07/02 14:18:51 950 [INFO,stat_random:119] -82.64 123.54
一个Mr. Random和三个Mr. If打128x16局：
20/07/02 12:39:33 774 [INFO,stat_random:119] -53.20 107.56
20/07/02 12:39:33 774 [INFO,stat_random:119] -56.47 106.12
20/07/02 12:39:33 775 [INFO,stat_random:119] -52.94 109.28
20/07/02 12:39:33 776 [INFO,stat_random:119] -102.85 131.01
四个Mr. If打128x16局：
20/07/02 12:40:30 030 [INFO,stat_random:119] -63.34 114.67
20/07/02 12:40:30 031 [INFO,stat_random:119] -67.39 121.14
20/07/02 12:40:30 031 [INFO,stat_random:119] -68.96 122.06
20/07/02 12:40:30 032 [INFO,stat_random:119] -68.15 114.92

教会Mr. If红桃草花黑桃方片的基本打法
    如果是猪牌并且我的猪剩两张以上
        如果我有猪并且有人打过猪圈，贴猪
        如果我是最后一个，打除了猪之外最大的
        其他情况打不会得猪的
    如果是变压器并且草花剩两张以上
        类似于猪
    如果是羊并且剩两张以上
        如果我是最后一个，我有羊，并且前面的牌都比羊小，打羊
        其他情况打不是羊的最大的
    如果是红桃，尽可能躲，捡大的贴
三个Mr. Random和一个Mr. If打128x16局：
20/07/02 14:15:45 410 [INFO,stat_random:119] -2.30 69.36
20/07/02 14:15:45 411 [INFO,stat_random:119] -83.91 118.71
20/07/02 14:15:45 411 [INFO,stat_random:119] -88.05 121.81
20/07/02 14:15:45 412 [INFO,stat_random:119] -91.43 123.46
两个Mr. Random和两个Mr. If打128x16局（邻）：
20/07/02 14:16:11 780 [INFO,stat_random:119] -18.26 80.08
20/07/02 14:16:11 780 [INFO,stat_random:119] -17.73 79.66
20/07/02 14:16:11 781 [INFO,stat_random:119] -117.96 138.31
20/07/02 14:16:11 782 [INFO,stat_random:119] -125.02 141.88
两个Mr. Random和两个Mr. If打128x16局（对）：
20/07/02 14:17:38 259 [INFO,stat_random:119] -22.49 81.92
20/07/02 14:17:38 259 [INFO,stat_random:119] -115.66 141.13
20/07/02 14:17:38 260 [INFO,stat_random:119] -19.26 77.73
20/07/02 14:17:38 261 [INFO,stat_random:119] -118.54 140.22
一个Mr. Random和三个Mr. If打128x16局：
20/07/02 14:16:33 123 [INFO,stat_random:119] -42.94 96.30
20/07/02 14:16:33 123 [INFO,stat_random:119] -37.52 101.65
20/07/02 14:16:33 124 [INFO,stat_random:119] -38.59 96.96
20/07/02 14:16:33 125 [INFO,stat_random:119] -169.94 167.55
四个Mr. If打128x16局：
20/07/02 14:16:53 298 [INFO,stat_random:119] -69.90 121.73
20/07/02 14:16:53 298 [INFO,stat_random:119] -66.86 120.47
20/07/02 14:16:53 299 [INFO,stat_random:119] -70.22 121.32
20/07/02 14:16:53 300 [INFO,stat_random:119] -69.22 121.62
"""