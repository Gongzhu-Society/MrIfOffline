#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import time,sys,traceback,math,numpy
LOGLEVEL={0:"DEBUG",1:"INFO",2:"WARN",3:"ERR",4:"FATAL"}
LOGFILE=sys.argv[0].split(".")
LOGFILE[-1]="log"
LOGFILE=".".join(LOGFILE)
def log(msg,l=1,end="\n",logfile=None,fileonly=False):
    st=traceback.extract_stack()[-2]
    lstr=LOGLEVEL[l]
    now_str="%s %03d"%(time.strftime("%y/%m/%d %H:%M:%S",time.localtime()),math.modf(time.time())[0]*1000)
    if l<3:
        tempstr="%s [%s,%s:%d] %s%s"%(now_str,lstr,st.name,st.lineno,str(msg),end)
    else:
        tempstr="%s [%s,%s:%d] %s:\n%s%s"%(now_str,lstr,st.name,st.lineno,str(msg),traceback.format_exc(limit=5),end)
    if not fileonly:
        print(tempstr,end="")
    if l>=1 or fileonly:
        if logfile==None:
            logfile=LOGFILE
        with open(logfile,"a") as f:
            f.write(tempstr)

import random

ORDER_DICT1={'S':-300,'H':-200,'D':-100,'C':0,'J':-200}
ORDER_DICT2={'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'1':10,'J':11,'Q':12,'K':13,'A':14,'P':15,'G':16}
def cards_order(card):
    return ORDER_DICT1[card[0]]+ORDER_DICT2[card[1]]
def get_max_card(l):
    """请在使用时把cards_on_table中第一个int去掉"""
    max_card=None
    score_temp=-500
    for i in l:
        if i[0]==l[0][0]:
            score_temp2=cards_order(i)
            if score_temp2>score_temp:
                max_card=i
                score_temp=score_temp2
    return max_card,score_temp

class MrRandom():
    """ONLY for 4 players"""
    def __init__(self,room,place,name):
        self.place = place
        self.room = room
        self.name = name
        self.players_information = [None, None, None, None]
        self.cards_list = [] #cards in hand
        self.cards_dict = {"S": [], "H": [], "D": [], "C": []} 
        self.initial_cards = [] #cards initial
        self.history = [] #list of (int,str,str,str,str)
        self.cards_on_table = [] #[int,str,...]

    def receive_shuffle(self,cards):
        """接收洗牌"""
        self.cards_list=cards
        for i in self.cards_list:
            self.cards_dict[i[0]].append(i)
        #log("%s received shuffle: %s"%(self.name,self.cards_list))

    def pop_card(self,which):
        """确认手牌打出后会被调用，更新手牌的数据结构"""
        self.cards_list.remove(which)
        self.cards_dict[which[0]].remove(which)

    def pick_a_card(self,suit):
        try:
            assert len(self.cards_list)==sum([len(self.cards_dict[k]) for k in self.cards_dict])
        except:
            log("",l=3)
        #log("%s, %s, %s, %s"%(self.name,suit,self.cards_on_table,self.cards_list))
        if self.cards_dict.get(suit)==None or len(self.cards_dict[suit])==0:
            i=random.randint(0,len(self.cards_list)-1)
            choice=self.cards_list[i]
        else:
            i=random.randint(0,len(self.cards_dict[suit])-1)
            choice=self.cards_dict[suit][i]
        #log("%s plays %s"%(self.name,choice))
        return choice

    @staticmethod
    def family_name():
        return 'MrRandom'

class Human(MrRandom):
    def pick_a_card(self,suit):
        log("%s, %s, %s, %s"%(self.name,suit,self.cards_on_table,self.cards_list))
        while True:
            choice=input("your turn: ")
            if choice in self.cards_list:
                break
            else:
                log("%s is not your cards. "%(choice),end="")
        return choice

    @staticmethod
    def family_name():
        return 'Human'

def get_nonempty_min(l):
    if len(l)!=0:
        return len(l)
    else:
        return 100

class MrIf(MrRandom):
    """
    如果随便出
        从所剩张数少的花色开始，如果没有“危险牌”，就出这个花色大的
        尽量不出猪、猪圈、变压器、比变压器大的、红桃AK、羊
    如果是贴牌，按危险列表依次贴，没有危险列表了，贴短的
    如果是猪牌并且我的猪剩两张以上
        如果我有猪并且有人打过猪圈，贴猪
        如果我是最后一个并且前面没认出过猪，打除了猪之外最大的
        其他情况打不会得猪的
    如果是变压器并且草花剩两张以上
        类似于猪
    如果是羊并且剩两张以上
        如果我是最后一个，我有羊，并且前面的牌都比羊小，打羊
        其他情况打不是羊的最大的
    如果是红桃，尽可能躲，捡大的贴
    """
    def pick_a_card(self,suit):
        try:
            assert len(self.cards_list)==sum([len(self.cards_dict[k]) for k in self.cards_dict])
        except:
            log("",l=3)
        #log("%s, %s, %s, %s"%(self.name,suit,self.cards_on_table,self.cards_list))
        #如果随便出
        if suit=="A":
            list_temp=[self.cards_dict[k] for k in self.cards_dict]
            list_temp.sort(key=get_nonempty_min)
            #log(list_temp)
            for i in range(4):
                if len(list_temp[i])==0:
                    continue
                suit_temp=list_temp[i][0][0]
                #log("thinking %s"%(suit_temp))
                if suit_temp=="S" and ("SQ" not in self.cards_list) \
                and ("SK" not in self.cards_list) and ("SA" not in self.cards_list):
                    choice=self.cards_dict["S"][-1]
                    return choice
                if suit_temp=="H" and ("HQ" not in self.cards_list) \
                and ("HK" not in self.cards_list) and ("HA" not in self.cards_list):
                    choice=self.cards_dict["H"][-1]
                    return choice
                if suit_temp=="C" and ("C10" not in self.cards_list) \
                and ("CJ" not in self.cards_list) and ("CQ" not in self.cards_list)\
                and ("CK" not in self.cards_list) and ("CA" not in self.cards_list):
                    choice=self.cards_dict["C"][-1]
                    return choice
                if suit_temp=="D" and ("DJ" not in self.cards_list):
                    choice=self.cards_dict["D"][-1]
                    return choice
            for i in range(5):
                choice=random.choice(self.cards_list)
                if choice not in ("SQ","SK","SA","HA","HK","C10","CJ","CQ","CK","CA","DJ"):
                    return choice
        #如果是贴牌
        elif len(self.cards_dict[suit])==0:
            for i in ("SQ","HA","SA","SK","HK","C10","CA","HQ","HJ","CK","CQ","CJ","H10","H9","H8","H7","H6","H5"):
                if i in self.cards_list:
                    choice=i
                    return choice
            list_temp=[self.cards_dict[k] for k in self.cards_dict]
            list_temp.sort(key=get_nonempty_min)
            for i in range(4):
                if len(list_temp[i])==0:
                    continue
                suit_temp=list_temp[i][0][0]
                choice=self.cards_dict[suit_temp][-1]
                return choice
        #如果只有这一张
        elif len(self.cards_dict[suit])==1:
            choice=self.cards_dict[suit][-1]
            return choice

        #如果是猪并且剩好几张猪牌
        if suit=="S":
            if ("SQ" in self.cards_list) and (("SK" in self.cards_on_table) or ("SA" in self.cards_on_table)):
                choice="SQ"
                return choice
            if len(self.cards_on_table)==4 and ("SQ" not in self.cards_on_table):
                choice=self.cards_dict["S"][-1]
                if choice=="SQ":
                    choice=self.cards_dict["S"][-2]
                return choice
            else:
                if "SA" in self.cards_on_table[1:]:
                    max_pig=cards_order("SA")
                elif "SK" in self.cards_on_table[1:]:
                    max_pig=cards_order("SK")
                else:
                    max_pig=cards_order("SQ")
                for i in self.cards_dict["S"][::-1]:
                    if cards_order(i)<max_pig:
                        choice=i
                        return choice
                else:
                    choice=self.cards_dict["S"][-1]
                    return choice
        #如果是变压器并且草花剩两张以上
        if suit=="C":
            if ("C10" in self.cards_list)\
            and (("CJ" in self.cards_on_table) or ("CQ" in self.cards_on_table) or\
                 ("CK" in self.cards_on_table) or ("CA" in self.cards_on_table)):
                choice="C10"
                return choice
            if len(self.cards_on_table)==4 and ("C10" not in self.cards_on_table):
                choice=self.cards_dict["C"][-1]
                if choice=="C10":
                    choice=self.cards_dict["C"][-2]
                return choice
            else:
                if "CA" in self.cards_on_table[1:]:
                    max_club=cards_order("CA")
                elif "CK" in self.cards_on_table[1:]:
                    max_club=cards_order("CK")
                elif "CQ" in self.cards_on_table[1:]:
                    max_club=cards_order("CQ")
                elif "CJ" in self.cards_on_table[1:]:
                    max_club=cards_order("CJ")
                else:
                    max_club=cards_order("C10")
                for i in self.cards_dict["C"][::-1]:
                    if cards_order(i)<max_club:
                        choice=i
                        return choice
                else:
                    choice=self.cards_dict["C"][-1]
                    return choice
        #如果是羊并且剩两张以上
        if suit=="D":
            if len(self.cards_on_table)==4 and ("DJ" in self.cards_dict["D"])\
            and ("DA" not in self.cards_on_table) and ("DK" not in self.cards_on_table)\
            and ("DQ" not in self.cards_on_table):
                choice="DJ"
                return choice
            choice=self.cards_dict["D"][-1]
            if choice=="DJ":
                choice=self.cards_dict["D"][-2]
            return choice
        #如果是红桃
        if suit=="H":
            max_heart=-1000
            for i in self.cards_on_table[1:]:
                if i[0]=="H" and cards_order(i)>max_heart:
                    max_heart=cards_order(i)
            for i in self.cards_dict["H"][::-1]:
                if cards_order(i)<max_heart:
                    choice=i
                    return choice
        #log("cannot be decided by rules")
        return MrRandom.pick_a_card(self,suit)

    @staticmethod
    def family_name():
        return 'MrIf'

def test_avoid_C10():
    if0=MrIf(0,0,"if0")
    if0.receive_shuffle(['D2', 'D7', 'DQ', 'C2', 'C4', 'C5', 'C6', 'C9', 'CQ'])
    if0.cards_on_table=[3, 'C8', 'C3', 'C10']
    log(if0.pick_a_card("C"))

if __name__=="__main__":
    test_avoid_C10()

# git push https://github.com/Gongzhu-Society/MrIfOffline.git
# git pull https://github.com/Gongzhu-Society/MrIfOffline.git
# git add .
# git commit