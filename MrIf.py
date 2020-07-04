#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import ORDER_DICT1,ORDER_DICT2
from MrRandom import MrRandom
import random

"""def get_max_card(l):
    请在使用时把cards_on_table中第一个int去掉
    max_card=None
    score_temp=-500
    for i in l:
        if i[0]==l[0][0]:
            score_temp2=cards_order(i)
            if score_temp2>score_temp:
                max_card=i
                score_temp=score_temp2
    return max_card,score_temp"""

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
    def pick_a_card(self):
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place,"self.place and self.cards_on_table contrdict"
        if len(self.cards_on_table)==1:
            suit="A"
        else:
            suit=self.cards_on_table[1][0]
        cards_dict={"S":[],"H":[],"D":[],"C":[]}
        for i in self.cards_list:
            cards_dict[i[0]].append(i)
        #log("%s, %s, %s, %s"%(self.name,suit,self.cards_on_table,self.cards_list))
        #如果随便出
        if suit=="A":
            list_temp=[cards_dict[k] for k in cards_dict]
            list_temp.sort(key=get_nonempty_min)
            #log(list_temp)
            for i in range(4):
                if len(list_temp[i])==0:
                    continue
                suit_temp=list_temp[i][0][0]
                #log("thinking %s"%(suit_temp))
                if suit_temp=="S" and ("SQ" not in self.cards_list)\
                and ("SK" not in self.cards_list) and ("SA" not in self.cards_list):
                    choice=cards_dict["S"][-1]
                    return choice
                if suit_temp=="H" and ("HQ" not in self.cards_list)\
                and ("HK" not in self.cards_list) and ("HA" not in self.cards_list):
                    choice=cards_dict["H"][-1]
                    return choice
                if suit_temp=="C" and ("C10" not in self.cards_list)\
                and ("CJ" not in self.cards_list) and ("CQ" not in self.cards_list)\
                and ("CK" not in self.cards_list) and ("CA" not in self.cards_list):
                    choice=cards_dict["C"][-1]
                    return choice
                if suit_temp=="D" and ("DJ" not in self.cards_list):
                    choice=cards_dict["D"][-1]
                    return choice
            for i in range(5):
                choice=random.choice(self.cards_list)
                if choice not in ("SQ","SK","SA","HA","HK","C10","CJ","CQ","CK","CA","DJ"):
                    return choice
        #如果是贴牌
        elif len(cards_dict[suit])==0:
            for i in ("SQ","HA","SA","SK","HK","C10","CA","HQ","HJ","CK","CQ","CJ","H10","H9","H8","H7","H6","H5"):
                if i in self.cards_list:
                    choice=i
                    return choice
            list_temp=[cards_dict[k] for k in cards_dict]
            list_temp.sort(key=get_nonempty_min)
            for i in range(4):
                if len(list_temp[i])==0:
                    continue
                suit_temp=list_temp[i][0][0]
                choice=cards_dict[suit_temp][-1]
                return choice
        #如果只有这一张
        elif len(cards_dict[suit])==1:
            choice=cards_dict[suit][-1]
            return choice

        #如果是猪并且剩好几张猪牌
        if suit=="S":
            if ("SQ" in self.cards_list) and (("SK" in self.cards_on_table) or ("SA" in self.cards_on_table)):
                choice="SQ"
                return choice
            if len(self.cards_on_table)==4 and ("SQ" not in self.cards_on_table):
                choice=cards_dict["S"][-1]
                if choice=="SQ":
                    choice=cards_dict["S"][-2]
                return choice
            else:
                if "SA" in self.cards_on_table[1:]:
                    max_pig=cards_order("SA")
                elif "SK" in self.cards_on_table[1:]:
                    max_pig=cards_order("SK")
                else:
                    max_pig=cards_order("SQ")
                for i in cards_dict["S"][::-1]:
                    if cards_order(i)<max_pig:
                        choice=i
                        return choice
                else:
                    choice=cards_dict["S"][-1]
                    return choice
        #如果是变压器并且草花剩两张以上
        if suit=="C":
            if ("C10" in self.cards_list)\
            and (("CJ" in self.cards_on_table) or ("CQ" in self.cards_on_table) or\
                 ("CK" in self.cards_on_table) or ("CA" in self.cards_on_table)):
                choice="C10"
                return choice
            if len(self.cards_on_table)==4 and ("C10" not in self.cards_on_table):
                choice=cards_dict["C"][-1]
                if choice=="C10":
                    choice=cards_dict["C"][-2]
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
                for i in cards_dict["C"][::-1]:
                    if cards_order(i)<max_club:
                        choice=i
                        return choice
                else:
                    choice=cards_dict["C"][-1]
                    return choice
        #如果是羊并且剩两张以上
        if suit=="D":
            if len(self.cards_on_table)==4 and ("DJ" in cards_dict["D"])\
            and ("DA" not in self.cards_on_table) and ("DK" not in self.cards_on_table)\
            and ("DQ" not in self.cards_on_table):
                choice="DJ"
                return choice
            choice=cards_dict["D"][-1]
            if choice=="DJ":
                choice=cards_dict["D"][-2]
            return choice
        #如果是红桃
        if suit=="H":
            max_heart=-1000
            for i in self.cards_on_table[1:]:
                if i[0]=="H" and cards_order(i)>max_heart:
                    max_heart=cards_order(i)
            for i in cards_dict["H"][::-1]:
                if cards_order(i)<max_heart:
                    choice=i
                    return choice
        #log("cannot be decided by rules")
        return MrRandom.pick_a_card(self)

    @staticmethod
    def family_name():
        return 'MrIf'

def test_avoid_C10():
    if0=MrIf(room=0,place=0,name="if0")
    if0.cards_list=['D2', 'D7', 'DQ', 'C2', 'C4', 'C5', 'C6', 'C9', 'CQ']
    if0.cards_on_table=[1, 'C8', 'C3', 'C10']
    log(if0.pick_a_card())

if __name__=="__main__":
    test_avoid_C10()