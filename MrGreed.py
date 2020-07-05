#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import ORDER_DICT2,INIT_CARDS
from MrRandom import MrRandom
import random

def nonempty_len(l):
    r=len(l)
    if r==0:
        r=100
    return r

class MrGreed(MrRandom):
    SCORE_DICT={'SQ':-100,'DJ':100,'C10':-60,
            'H2':0,'H3':0,'H4':0,'H5':-10,'H6':-10,'H7':-10,'H8':-10,'H9':-10,'H10':-10,
            'HJ':-20,'HQ':-30,'HK':-40,'HA':-50,'JP':-60,'JG':-70}
    BURDEN_DICT={'SA':45,'SK':35,'SJ':9,'S10':8,'S9':7,'S8':6,'S7':5,'S6':4,'S5':3,'S4':2,'S3':1,
                 'CA':16,'CK':15,'CQ':9,'CJ':8,'C9':7,'C8':6,'C7':5,'C6':4,'C5':3,'C4':2,'C3':1,
                 'DA':-3,'DK':-2,'DQ':-1,'D10':8,'D9':7,'D8':6,'D7':5,'D6':4,'D5':3,'D4':2,'D3':1,
                 'H10':6,'H9':5,'H8':4,'H7':3,'H6':2,'H5':1,'H4':3,'H3':2,'H2':1}
    N_THIRD=10
    N_SECOND=10
    N_FIRST=10

    def gen_legal_choice(suit,cards_dict,cards_list):
        if cards_dict.get(suit)==None or len(cards_dict[suit])==0:
            return cards_list
        else:
            return cards_dict[suit]

    def gen_cards_dict(cards_list):
        cards_dict={"S":[],"H":[],"D":[],"C":[]}
        for i in cards_list:
            cards_dict[i[0]].append(i)
        return cards_dict

    def clear_score(four_cards):
        winner=-1
        score_temp=ORDER_DICT2[four_cards[0][1]]
        if four_cards[1][0]==four_cards[0][0]\
        and ORDER_DICT2[four_cards[1][1]]>score_temp:
            winner=1
            score_temp=ORDER_DICT2[four_cards[1][1]]
        if four_cards[2][0]==four_cards[0][0]\
        and ORDER_DICT2[four_cards[2][1]]>score_temp:
            winner=-1
            score_temp=ORDER_DICT2[four_cards[2][1]]
        if four_cards[3][0]==four_cards[0][0]\
        and ORDER_DICT2[four_cards[3][1]]>score_temp:
            winner=1
        final_score=sum([MrGreed.SCORE_DICT.get(i,0) for i in four_cards])
        if 'C10' in four_cards:
            if 'SQ' in four_cards:
                final_score-=75
            if 'HA' in four_cards:
                final_score-=25
            if 'DJ' in four_cards:
                final_score+=25
        final_score*=winner
        return final_score

    def as_last_player(suit,four_cards,cards_dict,cards_list):
        '''return best four_cards directly'''
        if len(cards_dict[suit])==1:
            four_cards[3]=cards_dict[suit][0]
            return
        best_score=-1024
        four_cards_tmp=four_cards[0:3]+['']
        for c in MrGreed.gen_legal_choice(suit,cards_dict,cards_list):
            four_cards_tmp[3]=c
            score_temp=MrGreed.clear_score(four_cards_tmp)+MrGreed.BURDEN_DICT.get(c,0)
            if score_temp>best_score:
                four_cards[3]=four_cards_tmp[3]
                best_score=score_temp

    def as_third_player(suit,four_cards,cards_dict3,cards_list3,cards_dict4,cards_list4):
        '''return best four_cards directly'''
        if len(cards_dict3[suit])==1:
            four_cards[2]=cards_dict3[suit][0]
            MrGreed.as_last_player(suit,four_cards,cards_dict4,cards_list4)
            return
        best_score=-1024
        four_cards_tmp=four_cards[0:2]+['','']
        for c in MrGreed.gen_legal_choice(suit,cards_dict3,cards_list3):
            four_cards_tmp[2]=c
            MrGreed.as_last_player(suit,four_cards_tmp,cards_dict4,cards_list4)
            score_temp=-1*MrGreed.clear_score(four_cards_tmp)+MrGreed.BURDEN_DICT.get(c,0)
            if score_temp>best_score:
                four_cards[2]=four_cards_tmp[2]
                four_cards[3]=four_cards_tmp[3]
                best_score=score_temp

    def as_second_player(suit,four_cards,cards_dict2,cards_list2,cards_dict3,cards_list3,cards_dict4,cards_list4):
        if len(cards_dict2[suit])==1:
            four_cards[1]=cards_dict2[suit][0]
            MrGreed.as_third_player(suit,four_cards,cards_dict3,cards_list3,cards_dict4,cards_list4)
            return
        best_score=-1024
        four_cards_tmp=four_cards[0:1]+['','','']
        for c in MrGreed.gen_legal_choice(suit,cards_dict2,cards_list2):
            four_cards_tmp[1]=c
            MrGreed.as_third_player(suit,four_cards_tmp,cards_dict3,cards_list3,cards_dict4,cards_list4)
            score_temp=MrGreed.clear_score(four_cards_tmp)+MrGreed.BURDEN_DICT.get(c,0)
            if score_temp>best_score:
                four_cards[1]=four_cards_tmp[1]
                four_cards[2]=four_cards_tmp[2]
                four_cards[3]=four_cards_tmp[3]
                best_score=score_temp

    def calc_cards_remain(history,cards_on_table,cards_list):
        cards_remain=set(INIT_CARDS)
        for h in history:
            for c in h[1:5]:
                cards_remain.remove(c)
        for c in cards_on_table[1:]:
            cards_remain.remove(c)
        for c in cards_list:
            cards_remain.remove(c)
        return list(cards_remain)

    def pick_best_from_dlegal(d_legal):
        best_choice,best_score=d_legal.popitem()
        for k in d_legal:
            if d_legal[k]>best_score:
                best_choice=k
                best_score=d_legal[k]
        return best_choice

    def gen_void_info(history,cards_on_table):
        void_info=[{'S':False,'H':False,'D':False,'C':False},{'S':False,'H':False,'D':False,'C':False}\
                  ,{'S':False,'H':False,'D':False,'C':False},{'S':False,'H':False,'D':False,'C':False}]
        for h in history:
            for i,c in enumerate(h[2:5]):
                if c[0]!=h[1][0]:
                    void_info[(h[0]+i+1)%4][h[1][0]]=True
        for i,c in enumerate(cards_on_table[2:]):
            if c[0]!=cards_on_table[1][0]:
                void_info[(cards_on_table[0]+i+1)%4][cards_on_table[1][0]]=True
        return void_info

    def check_void_legal(void_info,seat,cards_dict):
        if void_info[seat]['S'] and len(cards_dict['S'])>0:
            return False
        if void_info[seat]['H'] and len(cards_dict['H'])>0:
            return False
        if void_info[seat]['D'] and len(cards_dict['D'])>0:
            return False
        if void_info[seat]['C'] and len(cards_dict['C'])>0:
            return False
        return True

    def check_void_legal_2(void_info,seat,cards_list):
        cards_str=''.join((i[0] for i in cards_list))
        if void_info[seat]['S'] and 'S' in cards_str:
            return False
        if void_info[seat]['H'] and 'H' in cards_str:
            return False
        if void_info[seat]['D'] and 'D' in cards_str:
            return False
        if void_info[seat]['C'] and 'C' in cards_str:
            return False
        return True

    def pick_a_card(self):
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place,"self.place and self.cards_on_table contrdict"
        #log("my turn %s %s"%(self.cards_on_table,self.cards_list))
        suit=self.decide_suit()
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        
        #如果别无选择
        if cards_dict.get(suit)!=None and len(cards_dict[suit])==1:
            choice=cards_dict[suit][0]
            #log("I have no choice but %s"%(choice))
            return choice
        #如果我是最后一个出的
        if len(self.cards_on_table)==4:
            four_cards=self.cards_on_table[1:4]+[""]
            MrGreed.as_last_player(suit,four_cards,cards_dict,self.cards_list)
            choice=four_cards[3]
            #log("%s, %s I choose %s"%(self.cards_on_table,self.cards_list,choice))
            return choice
        
        #其他情况要估计先验概率了
        void_info=MrGreed.gen_void_info(self.history,self.cards_on_table)
        #log("get void info: %s"%(void_info,))
        #计算别人可能还有什么牌
        cards_remain=MrGreed.calc_cards_remain(self.history,self.cards_on_table,self.cards_list)
        #生成一个合法操作字典
        d_legal={}
        for c in MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list):
            d_legal[c]=0
        #如果我是倒数第二个
        if len(self.cards_on_table)==3:
            assert len(cards_remain)==3*len(self.cards_list)-2
            four_cards=self.cards_on_table[1:3]+['','']
            ax=0
            while ax<MrGreed.N_THIRD:
                random.shuffle(cards_remain)
                cards_list_temp_p2=cards_remain[len(self.cards_list):2*len(self.cards_list)-1]
                if MrGreed.check_void_legal_2(void_info,(self.place+2)%4,cards_list_temp_p2)==False:
                    #log("drop illegal scenario(+2): %s"%(cards_list_temp_p2))
                    continue
                cards_list_temp_p3=cards_remain[2*len(self.cards_list)-1:3*len(self.cards_list)-2]
                if MrGreed.check_void_legal_2(void_info,(self.place+3)%4,cards_list_temp_p3)==False:
                    #log("drop illegal scenario(+3): %s"%(cards_list_temp_p3))
                    continue
                cards_list_temp=cards_remain[0:len(self.cards_list)]
                cards_list_temp.sort(key=cards_order)
                cards_dict_temp=MrGreed.gen_cards_dict(cards_list_temp)
                if MrGreed.check_void_legal(void_info,(self.place+1)%4,cards_dict_temp)==False:
                    #log("drop illegal scenario(+1): %s"%(cards_list_temp))
                    continue
                ax+=1
                #log("gen scenario: %s"%(cards_list_temp))
                for c in d_legal:
                    four_cards[2]=c
                    MrGreed.as_last_player(suit,four_cards,cards_dict_temp,cards_list_temp)
                    score=-1*MrGreed.clear_score(four_cards)+MrGreed.BURDEN_DICT.get(c,0)
                    #log("If I choose %s: %s, %d"%(c,four_cards,score))
                    d_legal[c]+=score
            #log(d_legal)
            best_choice=MrGreed.pick_best_from_dlegal(d_legal)
            return best_choice
        #如果我是倒数第三个
        elif len(self.cards_on_table)==2:
            assert len(cards_remain)==3*len(self.cards_list)-1
            four_cards=[self.cards_on_table[1],'','','']
            ax=0
            while ax<MrGreed.N_SECOND:
                random.shuffle(cards_remain)
                cards_list_temp1=cards_remain[2*len(self.cards_list):3*len(self.cards_list)-1]
                if MrGreed.check_void_legal_2(void_info,(self.place+3)%4,cards_list_temp1)==False:
                    #log("drop illegal scenario(+3): %s"%(cards_list_temp1))
                    continue
                cards_list_temp3=cards_remain[0:len(self.cards_list)]
                cards_list_temp3.sort(key=cards_order)
                cards_dict_temp3=MrGreed.gen_cards_dict(cards_list_temp3)
                if MrGreed.check_void_legal(void_info,(self.place+1)%4,cards_dict_temp3)==False:
                    #log("drop illegal scenario(+1): %s"%(cards_list_temp3))
                    continue
                cards_list_temp4=cards_remain[len(self.cards_list):2*len(self.cards_list)]
                cards_list_temp4.sort(key=cards_order)
                cards_dict_temp4=MrGreed.gen_cards_dict(cards_list_temp4)
                if MrGreed.check_void_legal(void_info,(self.place+2)%4,cards_dict_temp4)==False:
                    #log("drop illegal scenario(+2): %s"%(cards_list_temp4))
                    continue
                ax+=1
                #log("gen scenario: %s, %s"%(cards_list_temp3,cards_list_temp4))
                for c in d_legal:
                    four_cards[1]=c
                    MrGreed.as_third_player(suit,four_cards,cards_dict_temp3,cards_list_temp3,cards_dict_temp4,cards_list_temp4)
                    score=MrGreed.clear_score(four_cards)+MrGreed.BURDEN_DICT.get(c,0)
                    #log("If I choose %s: %s, %d"%(c,four_cards,score))
                    d_legal[c]+=score
            #log(d_legal)
            best_choice=MrGreed.pick_best_from_dlegal(d_legal)
            return best_choice
        #如果我是第一个出
        elif len(self.cards_on_table)==1:
            assert len(cards_remain)==3*len(self.cards_list)
            four_cards=['','','','']
            ax=0
            while ax<MrGreed.N_FIRST:
                random.shuffle(cards_remain)
                cards_list_temp2=cards_remain[0:len(self.cards_list)]
                cards_list_temp2.sort(key=cards_order)
                cards_dict_temp2=MrGreed.gen_cards_dict(cards_list_temp2)
                if MrGreed.check_void_legal_2(void_info,(self.place+1)%4,cards_list_temp2)==False:
                    #log("drop illegal scenario(+1): %s"%(cards_list_temp2))
                    continue
                cards_list_temp3=cards_remain[len(self.cards_list):len(self.cards_list)*2]
                cards_list_temp3.sort(key=cards_order)
                cards_dict_temp3=MrGreed.gen_cards_dict(cards_list_temp3)
                if MrGreed.check_void_legal(void_info,(self.place+2)%4,cards_dict_temp3)==False:
                    #log("drop illegal scenario(+2): %s"%(cards_list_temp3))
                    continue
                cards_list_temp4=cards_remain[len(self.cards_list)*2:len(self.cards_list)*3]
                cards_list_temp4.sort(key=cards_order)
                cards_dict_temp4=MrGreed.gen_cards_dict(cards_list_temp4)
                if MrGreed.check_void_legal(void_info,(self.place+3)%4,cards_dict_temp4)==False:
                    #log("drop illegal scenario(+3): %s"%(cards_list_temp4))
                    continue
                ax+=1
                #log("gen scenario: %s, %s, %s"%(cards_list_temp2,cards_list_temp3,cards_list_temp4))
                for c in d_legal:
                    four_cards[0]=c
                    MrGreed.as_second_player(c[0],four_cards,cards_dict_temp2,cards_list_temp2\
                                                            ,cards_dict_temp3,cards_list_temp3\
                                                            ,cards_dict_temp4,cards_list_temp4)
                    score=-1*MrGreed.clear_score(four_cards)+MrGreed.BURDEN_DICT.get(c,0)
                    #log("If I choose %s: %s, %d"%(c,four_cards,score))
                    d_legal[c]+=score
            #log(d_legal)
            best_choice=MrGreed.pick_best_from_dlegal(d_legal)
            return best_choice
            """list_temp=[cards_dict[k] for k in cards_dict]
            list_temp.sort(key=nonempty_len)
            for i in range(2):
                if len(list_temp[i])==0:
                    continue
                suit_temp=list_temp[i][0][0]
                #log("thinking %s"%(suit_temp))
                if suit_temp=="S" and ("SQ" not in self.cards_list)\
                and ("SK" not in self.cards_list) and ("SA" not in self.cards_list):
                    choice=cards_dict["S"][-1]
                    return choice
                elif suit_temp=="H" and ("HQ" not in self.cards_list)\
                and ("HK" not in self.cards_list) and ("HA" not in self.cards_list):
                    choice=cards_dict["H"][-1]
                    return choice
                elif suit_temp=="C" and ("C10" not in self.cards_list)\
                and ("CJ" not in self.cards_list) and ("CQ" not in self.cards_list)\
                and ("CK" not in self.cards_list) and ("CA" not in self.cards_list):
                    choice=cards_dict["C"][-1]
                    return choice
            cards_set=set(self.cards_list)
            for c in ("SQ","SK","SA","HA","HK","HQ","C10","CJ","CQ","CK","CA"):
                cards_set.discard(c)
            if len(cards_set)>0:
                return random.choice(list(cards_set))
            else:
                return random.choice(self.cards_list)"""
        log("I cannot decide")
        if cards_dict.get(suit)==None or len(cards_dict[suit])==0:
            i=random.randint(0,len(self.cards_list)-1)
            choice=self.cards_list[i]
        else:
            i=random.randint(0,len(cards_dict[suit])-1)
            choice=cards_dict[suit][i]
        return choice

    @staticmethod
    def family_name():
        return 'MrGreed'


def test_1st():
    g0=MrGreed(room=0,place=0,name="if0")
    g0.history=[(0,'C2','C3','C4','C5'),(3,'C6','C7','C8','D2'),(2,'D3','D4','D5','D6'),(1,'S2','S3','S4','S5')]
    g0.cards_list=['H6','H7','HJ','DJ','DK','C9','CJ','D7','D10']
    g0.cards_on_table=[0,]
    log(g0.cards_list)
    log(g0.pick_a_card())

def test_2nd():
    g0=MrGreed(room=0,place=0,name="if0")
    g0.history=[(0,'C2','C3','C4','C5'),(3,'C6','C7','D2','C8'),(2,'D3','D4','D5','D6'),(1,'S2','S3','S4','S5')]
    g0.cards_list=['H6','H7','HJ','HA','SQ','DJ','DK']
    #g0.cards_on_table=[3,'HQ']
    g0.cards_on_table=[3,'CK']
    log(g0.cards_list)
    log(g0.pick_a_card())

def test_3rd():
    g0=MrGreed(room=0,place=0,name="if0")
    g0.history=[(0,'C2','C3','C4','C5'),(3,'C6','C7','D2','C8'),(2,'D3','D4','D5','D6'),(1,'S2','S3','S4','S5')]
    g0.cards_list=['H6','H7','HJ','HA','SQ','DJ','DK']
    g0.cards_on_table=[2,'H8','HQ']
    #g0.cards_on_table=[2,'CA','CJ']
    log(g0.cards_list)
    log(g0.pick_a_card())

def test_last():
    g0=MrGreed(room=0,place=0,name="if0")
    #红桃就是躲
    g0.cards_list=['H3','H7','HJ','HA']
    g0.cards_on_table=[1, 'HQ', 'H8', 'H2']
    #羊要给队友
    #g0.cards_list=['D2','D7','DJ','S5','S8','S10']
    #g0.cards_on_table=[1, 'D8', 'DA', 'HJ']
    #不能把猪给队友
    #g0.cards_list=['D2','D7','DQ','S5','S8','SQ']
    #g0.cards_on_table=[1, 'S8', 'SA', 'S2']
    #有 好东西/坏东西 要 拿到/避开
    #g0.cards_list=['D2','D7','DQ','S5','S7','S10']
    #g0.cards_on_table=[1, 'S8', 'S2', 'DJ']
    #不得猪的情况下尽可能出大的
    #g0.cards_list=['D2','D7','DQ','S5','S8','S10']
    #g0.cards_on_table=[1, 'S8', 'SJ', 'S2']
    #无药可救时出大的
    #g0.cards_list=['D2','D7','DQ','S5','S8','S10']
    #g0.cards_on_table=[1, 'S8', 'SQ', 'S2']
    log(g0.pick_a_card())

if __name__=="__main__":
    #test_last()
    #test_3rd()
    #test_2nd()
    test_1st()