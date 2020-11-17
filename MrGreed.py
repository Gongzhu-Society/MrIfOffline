#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import ORDER_DICT2,INIT_CARDS,SCORE_DICT
from MrRandom import MrRandom
from MrO_Trainer import cards_in_hand_oh,score_oh,four_cards_oh
import random,itertools,copy,torch

class MrGreed(MrRandom):
    BURDEN_DICT={'SA':11,'SK':9,'SQ':8,'SJ':7,'S10':6,'S9':5,'S8':4,'S7':3,'S6':2,'S5':1,'S4':1,
                 'CA':11,'CK':9,'CQ':8,'CJ':7,'C10':6,'C9':5,'C8':4,'C7':3,'C6':2,'C5':1,'C4':1,
                 'DA':11,'DK':9,'DQ':8,'DJ':7,'D10':6,'D9':5,'D8':4,'D7':3,'D6':2,'D5':1,'D4':1,
                 'H10':6,'H9':5,'H8':4,'H7':3,'H6':2,'H5':1,'H4':1}
    BURDEN_DICT_S={'SA':50,'SK':30}
    BURDEN_DICT_D={'DA':-30,'DK':-20,'DQ':-10}
    BURDEN_DICT_C={'CA':0.4,'CK':0.3,'CQ':0.2,'CJ':0.1} #ratio of burden, see calc_relief
    SHORT_PREFERENCE=30 #will multiply (average suit count)-(my suit count), if play first
    N_SAMPLE=5

    def gen_cards_dict(cards_list):
        cards_dict={"S":[],"H":[],"D":[],"C":[]}
        for i in cards_list:
            cards_dict[i[0]].append(i)
        return cards_dict

    def gen_legal_choice(suit,cards_dict,cards_list):
        if cards_dict.get(suit)==None or len(cards_dict[suit])==0:
            return cards_list
        else:
            return cards_dict[suit]

    def gen_fmt_scores(scards):
        """
            generate formatted scores which is easy for computation
            [0,0,False,False] stands for [score, #hearts, C10 flag, has score flag]
        """
        fmt_score_list=[[0,0,False,False],[0,0,False,False],[0,0,False,False],[0,0,False,False]]
        for i,cs in enumerate(scards):
            for c in cs:
                fmt_score_list[i][0]+=SCORE_DICT[c]
                if c=='C10':
                    fmt_score_list[i][2]=True
                else:
                    fmt_score_list[i][3]=True
                    if c[0]=='H':
                        fmt_score_list[i][1]+=1
        return fmt_score_list

    def calc_score_change(fmt_score,four_cards,scs_rmn_avg):
        """
            calculate the marginal score change for the winner of this trick
            will be called by clear_score
            fmt_score  : formated score. see gen_fmt_scores
            four_cards : four cards for this trick
            scs_rmn_avg: scored cards remained average
        """
        delta_score=0
        for c in four_cards:
            delta_score+=SCORE_DICT.get(c,0)
        if fmt_score[2]:
            delta_score*=2
        elif 'C10' in four_cards:
            delta_score+=delta_score+fmt_score[0]+scs_rmn_avg
        return delta_score

    def clear_score(four_cards,fmt_score_list,trick_start,scs_rmn_avg):
        """
            judge the winner of this trick and then calc_score_change
            see calc_score_change for the meaning of parameters
        """
        winner=0
        score_temp=ORDER_DICT2[four_cards[0][1]]
        if four_cards[1][0]==four_cards[0][0]\
        and ORDER_DICT2[four_cards[1][1]]>score_temp:
            winner=1
            score_temp=ORDER_DICT2[four_cards[1][1]]
        if four_cards[2][0]==four_cards[0][0]\
        and ORDER_DICT2[four_cards[2][1]]>score_temp:
            winner=2
            score_temp=ORDER_DICT2[four_cards[2][1]]
        if four_cards[3][0]==four_cards[0][0]\
        and ORDER_DICT2[four_cards[3][1]]>score_temp:
            winner=3
        delta_score=MrGreed.calc_score_change(fmt_score_list[(trick_start+winner)%4],four_cards,scs_rmn_avg)
        if winner%2==0:
            delta_score*=-1
        return delta_score

    def gen_impc_dict(scards,cards_on_table):
        """
            generate important cards dict.
            important cards include SQ, DJ, C10 so far.
            played is True, not is False.
            scards: I cannot remember
        """
        impc_dict={'SQ':False,'DJ':False,'C10':False}
        for i in scards:
            for c in i:
                if c=='SQ':
                    impc_dict['SQ']=True
                elif c=='DJ':
                    impc_dict['DJ']=True
                elif c=='C10':
                    impc_dict['C10']=True
        for c in cards_on_table:
            if c=='SQ':
                impc_dict['SQ']=True
            elif c=='DJ':
                impc_dict['DJ']=True
            elif c=='C10':
                impc_dict['C10']=True
        return impc_dict

    def diy_impc_dict(impc_dict_base,cards_list):
        """
            modify impc_dict w.r.t cards in hand
        """
        impc_dict=copy.copy(impc_dict_base)
        if 'SQ' in cards_list:
            impc_dict['SQ']=True
        if 'DJ' in cards_list:
            impc_dict['DJ']=True
        if 'C10' in cards_list:
            impc_dict['C10']=True
        return impc_dict

    def calc_relief(card,impc_dict,scs_rmn_avg,my_score):
        """
            If there is SQ, SA and SK have an effective "burden". Getting ride of them is a "relief".
            impc_dict: see gen_impc_dict
        """
        relief=MrGreed.BURDEN_DICT.get(card,0)
        if impc_dict['SQ']==False:
            relief+=MrGreed.BURDEN_DICT_S.get(card,0)
        if impc_dict['DJ']==False:
            relief+=MrGreed.BURDEN_DICT_D.get(card,0)
        if impc_dict['C10']==False:
            relief+=int(-1*MrGreed.BURDEN_DICT_C.get(card,0)*(scs_rmn_avg+my_score))
        return relief

    def as_last_player(suit,four_cards,cards_dict,cards_list,
        fmt_score_list,trick_start,scs_rmn_avg,impc_dict_base,myplace):
        impc_dict=MrGreed.diy_impc_dict(impc_dict_base,cards_list)
        best_score=-65535
        four_cards_tmp=four_cards[0:3]+['']
        for c in MrGreed.gen_legal_choice(suit,cards_dict,cards_list):
            four_cards_tmp[3]=c
            score_temp=MrGreed.clear_score(four_cards_tmp,fmt_score_list,trick_start,scs_rmn_avg)\
                      +MrGreed.calc_relief(c,impc_dict,scs_rmn_avg,fmt_score_list[myplace][0])
            if score_temp>best_score:
                four_cards[3]=four_cards_tmp[3]
                best_score=score_temp

    def as_third_player(suit,four_cards,cards_dict3,cards_list3,cards_dict4,cards_list4,
        fmt_score_list,trick_start,scs_rmn_avg,impc_dict_base,myplace):
        impc_dict=MrGreed.diy_impc_dict(impc_dict_base,cards_list3)
        best_score=-65535
        four_cards_tmp=four_cards[0:2]+['','']
        for c in MrGreed.gen_legal_choice(suit,cards_dict3,cards_list3):
            four_cards_tmp[2]=c
            MrGreed.as_last_player(suit,four_cards_tmp,cards_dict4,cards_list4
                                  ,fmt_score_list,trick_start,scs_rmn_avg,impc_dict,myplace)
            score_temp=-1*MrGreed.clear_score(four_cards_tmp,fmt_score_list,trick_start,scs_rmn_avg)\
                       +MrGreed.calc_relief(c,impc_dict,scs_rmn_avg,fmt_score_list[myplace][0])
            if score_temp>best_score:
                four_cards[2:4]=four_cards_tmp[2:4]
                best_score=score_temp

    def as_second_player(suit,four_cards,cards_dict2,cards_list2,cards_dict3,cards_list3,cards_dict4,cards_list4,
        fmt_score_list,trick_start,scs_rmn_avg,impc_dict_base,myplace):
        impc_dict=MrGreed.diy_impc_dict(impc_dict_base,cards_list2)
        best_score=-65535
        four_cards_tmp=[four_cards[0],'','','']
        for c in MrGreed.gen_legal_choice(suit,cards_dict2,cards_list2):
            four_cards_tmp[1]=c
            MrGreed.as_third_player(suit,four_cards_tmp,cards_dict3,cards_list3,cards_dict4,cards_list4
                                   ,fmt_score_list,trick_start,scs_rmn_avg,impc_dict,myplace)
            score_temp=MrGreed.clear_score(four_cards_tmp,fmt_score_list,trick_start,scs_rmn_avg)\
                      +MrGreed.calc_relief(c,impc_dict,scs_rmn_avg,fmt_score_list[myplace][0])
            if score_temp>best_score:
                four_cards[1:4]=four_cards_tmp[1:4]
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
        """
            pick best choice from d_legal
            will be called by pick_a_card only once
        """
        best_choice,best_score=d_legal.popitem()
        for k in d_legal:
            if d_legal[k]>best_score:
                best_choice=k
                best_score=d_legal[k]
        return best_choice

    def gen_void_info(myseat,history,cards_on_table):
        """
            generate void info
            will be called in pick_a_card and used as the input for check_void_info in gen_scenario
        """
        void_info=[{'S':False,'H':False,'D':False,'C':False},{'S':False,'H':False,'D':False,'C':False}\
                  ,{'S':False,'H':False,'D':False,'C':False}]
        for h in history:
            for i,c in enumerate(h[2:5]):
                seat=(h[0]+i-myseat)%4
                if seat!=3 and c[0]!=h[1][0]:
                    void_info[seat][h[1][0]]=True
        for i,c in enumerate(cards_on_table[2:]):
            seat=(cards_on_table[0]+i-myseat)%4
            if seat!=3 and c[0]!=cards_on_table[1][0]:
                void_info[seat][cards_on_table[1][0]]=True
        return void_info

    def gen_scenario(void_info,cards_remain,lens):
        bx=0
        while True:
            bx+=1
            random.shuffle(cards_remain)
            if MrGreed.check_void_legal(cards_remain[0:lens[0]],cards_remain[lens[0]:lens[0]+lens[1]],cards_remain[lens[0]+lens[1]:],void_info):
                break
        cards_list_list=[cards_remain[0:lens[0]],cards_remain[lens[0]:lens[0]+lens[1]],cards_remain[lens[0]+lens[1]:sum(lens)]]
        a2b=[];a2c=[]
        for c in cards_list_list[0]:
            if not void_info[1][c[0]]:
                a2b.append(c)
            if not void_info[2][c[0]]:
                a2c.append(c)
        b2c=[];b2a=[]
        for c in cards_list_list[1]:
            if not void_info[2][c[0]]:
                b2c.append(c)
            if not void_info[0][c[0]]:
                b2a.append(c)
        c2a=[];c2b=[]
        for c in cards_list_list[2]:
            if not void_info[0][c[0]]:
                c2a.append(c)
            if not void_info[1][c[0]]:
                c2b.append(c)
        exchange_info=[[b2c,c2b],[c2a,a2c],[a2b,b2a]]
        return cards_list_list,exchange_info,bx

    def alter_scenario(cards_list_list,exchange_info,void_info):
        fsi_exc=[]
        if len(exchange_info[0][0])!=0 and len(exchange_info[0][1])!=0:
            fsi_exc.append(0)
        if len(exchange_info[1][0])!=0 and len(exchange_info[1][1])!=0:
            fsi_exc.append(1)
        if len(exchange_info[2][0])!=0 and len(exchange_info[2][1])!=0:
            fsi_exc.append(2)
        if len(fsi_exc)==0:
            return 1
        exc_ind=random.choice(fsi_exc)
        #from (exc_ind+1)%3 to (exc_ind+2)%3
        c_ind_0=random.randrange(0,len(exchange_info[exc_ind][0]))
        c_ind_1=random.randrange(0,len(exchange_info[exc_ind][1]))
        c0=exchange_info[exc_ind][0].pop(c_ind_0)
        c1=exchange_info[exc_ind][1].pop(c_ind_1)
        exchange_info[exc_ind][0].append(c1)
        exchange_info[exc_ind][1].append(c0)
        if not void_info[exc_ind][c0[0]]:
            exchange_info[(exc_ind+2)%3][1].remove(c0)
            exchange_info[(exc_ind+1)%3][0].append(c0)
        if not void_info[exc_ind][c1[0]]:
            exchange_info[(exc_ind+2)%3][1].append(c1)
            exchange_info[(exc_ind+1)%3][0].remove(c1)
        cards_list_list[(exc_ind+1)%3].remove(c0)
        cards_list_list[(exc_ind+1)%3].append(c1)
        cards_list_list[(exc_ind+2)%3].remove(c1)
        cards_list_list[(exc_ind+2)%3].append(c0)
        #assert MrGreed.check_void_legal(cards_list_list[0],cards_list_list[1],cards_list_list[2],void_info)
        return 0

    def check_void_legal(cards_list1,cards_list2,cards_list3,void_info):
        """
            check the senario generated agree with void info or not
            will be called by gen_scenario, asserted in alter_scenario
        """
        for i in range(3):
            s_temp=''.join((i[0] for i in cards_list1))
            if void_info[0]['S'] and 'S' in s_temp:
                return False
            if void_info[0]['H'] and 'H' in s_temp:
                return False
            if void_info[0]['D'] and 'D' in s_temp:
                return False
            if void_info[0]['C'] and 'C' in s_temp:
                return False
            s_temp=''.join((i[0] for i in cards_list2))
            if void_info[1]['S'] and 'S' in s_temp:
                return False
            if void_info[1]['H'] and 'H' in s_temp:
                return False
            if void_info[1]['D'] and 'D' in s_temp:
                return False
            if void_info[1]['C'] and 'C' in s_temp:
                return False
            s_temp=''.join((i[0] for i in cards_list3))
            if void_info[2]['S'] and 'S' in s_temp:
                return False
            if void_info[2]['H'] and 'H' in s_temp:
                return False
            if void_info[2]['D'] and 'D' in s_temp:
                return False
            if void_info[2]['C'] and 'C' in s_temp:
                return False
        return True

    def pick_a_card_pure(self):
        #确认桌上牌的数量和自己坐的位置相符
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place
        #utility datas
        suit=self.decide_suit()
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        #如果别无选择
        if cards_dict.get(suit)!=None and len(cards_dict[suit])==1:
            choice=cards_dict[suit][0]
            return choice
        #more utility datas
        fmt_score_list=MrGreed.gen_fmt_scores(self.scores)
        impc_dict_base=MrGreed.gen_impc_dict(self.scores,self.cards_on_table)
        scs_rmn_avg=(-200-sum([i[0] for i in fmt_score_list]))//4
        #如果我是最后一个出的
        if len(self.cards_on_table)==4:
            four_cards=self.cards_on_table[1:4]+[""]
            MrGreed.as_last_player(suit,four_cards,cards_dict,self.cards_list
                                  ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict_base,self.place)
            choice=four_cards[3]
            return choice

        #more utility datas
        cards_remain=MrGreed.calc_cards_remain(self.history,self.cards_on_table,self.cards_list)
        assert len(cards_remain)==3*len(self.cards_list)-(len(self.cards_on_table)-1) #确认别人手里牌的数量和我手里的还有桌上牌的数量相符
        void_info=MrGreed.gen_void_info(self.place,self.history,self.cards_on_table)
        impc_dict_mine=MrGreed.diy_impc_dict(impc_dict_base,self.cards_list)
        #如果我是倒数第二个
        if len(self.cards_on_table)==3:
            lens=[len(self.cards_list),len(self.cards_list)-1,len(self.cards_list)-1]
            four_cards=self.cards_on_table[1:3]+['','']
        #如果我是倒数第三个
        elif len(self.cards_on_table)==2:
            lens=[len(self.cards_list),len(self.cards_list),len(self.cards_list)-1]
            four_cards=[self.cards_on_table[1],'','','']
        #如果我是第一个出
        elif len(self.cards_on_table)==1:
            lens=[len(self.cards_list),len(self.cards_list),len(self.cards_list)]
            four_cards=['','','','']
        expire_date=0 #for sampling
        d_legal={c:0 for c in MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)} #dict of legal choice
        for ax in range(MrGreed.N_SAMPLE):
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
            if len(self.cards_on_table)==3:
                cards_list_1=cards_list_list[0]
                cards_dict_1=MrGreed.gen_cards_dict(cards_list_1)
                for c in d_legal:
                    four_cards[2]=c
                    MrGreed.as_last_player(suit,four_cards,cards_dict_1,cards_list_1
                                          ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict_base,self.place)
                    score=-1*MrGreed.clear_score(four_cards,fmt_score_list,self.cards_on_table[0],scs_rmn_avg)\
                          +MrGreed.calc_relief(c,impc_dict_mine,scs_rmn_avg,fmt_score_list[self.place][0])
                    d_legal[c]+=score
            elif len(self.cards_on_table)==2:
                cards_list_1=cards_list_list[0]
                cards_dict_1=MrGreed.gen_cards_dict(cards_list_1)
                cards_list_2=cards_list_list[1]
                cards_dict_2=MrGreed.gen_cards_dict(cards_list_2)
                for c in d_legal:
                    four_cards[1]=c
                    MrGreed.as_third_player(suit,four_cards,cards_dict_1,cards_list_1,cards_dict_2,cards_list_2
                                           ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict_base,self.place)
                    score=MrGreed.clear_score(four_cards,fmt_score_list,self.cards_on_table[0],scs_rmn_avg)\
                          +MrGreed.calc_relief(c,impc_dict_mine,scs_rmn_avg,fmt_score_list[self.place][0])
                    d_legal[c]+=score
            elif len(self.cards_on_table)==1:
                cards_list_1=cards_list_list[0]
                cards_dict_1=MrGreed.gen_cards_dict(cards_list_1)
                cards_list_2=cards_list_list[1]
                cards_dict_2=MrGreed.gen_cards_dict(cards_list_2)
                cards_list_3=cards_list_list[2]
                cards_dict_3=MrGreed.gen_cards_dict(cards_list_3)
                for c in d_legal:
                    four_cards[0]=c
                    MrGreed.as_second_player(c[0],four_cards,cards_dict_1,cards_list_1,cards_dict_2,cards_list_2,cards_dict_3,cards_list_3
                                            ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict_base,self.place)
                    score=-1*MrGreed.clear_score(four_cards,fmt_score_list,self.cards_on_table[0],scs_rmn_avg)\
                         +MrGreed.calc_relief(c,impc_dict_mine,scs_rmn_avg,fmt_score_list[self.place][0])
                    d_legal[c]+=score
        if len(self.cards_on_table)==1 and len(self.history)<3:
            suit_ct={'S':0,'H':0,'D':0,'C':0}
            for h,i in itertools.product(self.history,range(1,5)):
                suit_ct[h[i][0]]+=1
            d_suit_extra={'S':0,'H':0,'D':0,'C':0}
            for s in 'SHDC':
                my_len=len(cards_dict[s])
                avg_len=(13-suit_ct[s]-my_len)/3
                d_suit_extra[s]=int((avg_len-my_len)*MrGreed.SHORT_PREFERENCE*MrGreed.N_SAMPLE)
            for c in d_legal:
                d_legal[c]+=d_suit_extra[c[0]]
        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice

    def pick_a_card_record_for_o(self):
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place
        suit=self.decide_suit()
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        fmt_score_list=MrGreed.gen_fmt_scores(self.scores)
        scs_rmn_avg=(-200-sum([i[0] for i in fmt_score_list]))//4
        cards_remain=MrGreed.calc_cards_remain(self.history,self.cards_on_table,self.cards_list)
        void_info=MrGreed.gen_void_info(self.place,self.history,self.cards_on_table)
        impc_dict_base=MrGreed.gen_impc_dict(self.scores,self.cards_on_table)
        impc_dict_mine=MrGreed.diy_impc_dict(impc_dict_base,self.cards_list)
        expire_date=0 #for sampling
        d_legal={c:0 for c in MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)} #dict of legal choice
        if len(self.cards_on_table)==4:
            lens=[len(self.cards_list)-1,len(self.cards_list)-1,len(self.cards_list)-1]
            four_cards=self.cards_on_table[1:4]+['']
        elif len(self.cards_on_table)==3:
            lens=[len(self.cards_list),len(self.cards_list)-1,len(self.cards_list)-1]
            four_cards=self.cards_on_table[1:3]+['','']
        elif len(self.cards_on_table)==2:
            lens=[len(self.cards_list),len(self.cards_list),len(self.cards_list)-1]
            four_cards=[self.cards_on_table[1],'','','']
        elif len(self.cards_on_table)==1:
            lens=[len(self.cards_list),len(self.cards_list),len(self.cards_list)]
            four_cards=['','','','']
        if len(self.cards_on_table)==1 and len(self.history)<3:
            suit_ct={'S':0,'H':0,'D':0,'C':0}
            for h,i in itertools.product(self.history,range(1,5)):
                suit_ct[h[i][0]]+=1
            d_suit_extra={'S':0,'H':0,'D':0,'C':0}
            for s in 'SHDC':
                my_len=len(cards_dict[s])
                avg_len=(13-suit_ct[s]-my_len)/3
                d_suit_extra[s]=int((avg_len-my_len)*MrGreed.SHORT_PREFERENCE)
            del suit_ct
        for ax in range(MrGreed.N_SAMPLE):
            #sampling
            if expire_date==0:
                cards_list_list,exchange_info,bx=MrGreed.gen_scenario(void_info,cards_remain,lens)
                expire_date=max(bx-5,0)
            else:
                exhausted_flag=MrGreed.alter_scenario(cards_list_list,exchange_info,void_info)
                if exhausted_flag==1:
                    break
                expire_date-=1
            cards_list_1=cards_list_list[0]
            cards_dict_1=MrGreed.gen_cards_dict(cards_list_1)
            cards_list_2=cards_list_list[1]
            cards_dict_2=MrGreed.gen_cards_dict(cards_list_2)
            cards_list_3=cards_list_list[2]
            cards_dict_3=MrGreed.gen_cards_dict(cards_list_3)
            #decide
            for c in d_legal:
                if len(self.cards_on_table)==4:
                    four_cards[3]=c
                    score=MrGreed.clear_score(four_cards,fmt_score_list,self.cards_on_table[0],scs_rmn_avg)
                elif len(self.cards_on_table)==3:
                    four_cards[2]=c
                    MrGreed.as_last_player(suit,four_cards,cards_dict_1,cards_list_1
                                          ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict_base,self.place)
                    score=-1*MrGreed.clear_score(four_cards,fmt_score_list,self.cards_on_table[0],scs_rmn_avg)
                elif len(self.cards_on_table)==2:
                    four_cards[1]=c
                    MrGreed.as_third_player(suit,four_cards,cards_dict_1,cards_list_1,cards_dict_2,cards_list_2
                                           ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict_base,self.place)
                    score=MrGreed.clear_score(four_cards,fmt_score_list,self.cards_on_table[0],scs_rmn_avg)
                elif len(self.cards_on_table)==1:
                    four_cards[0]=c
                    MrGreed.as_second_player(c[0],four_cards,cards_dict_1,cards_list_1,cards_dict_2,cards_list_2,cards_dict_3,cards_list_3
                                            ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict_base,self.place)
                    score=-1*MrGreed.clear_score(four_cards,fmt_score_list,self.cards_on_table[0],scs_rmn_avg)
                    if len(self.history)<3:
                        score+=d_suit_extra[c[0]]
                score+=MrGreed.calc_relief(c,impc_dict_mine,scs_rmn_avg,fmt_score_list[self.place][0])
                score+=fmt_score_list[self.place][0]+scs_rmn_avg
                d_legal[c]+=score
                #记录, 只有有的选才记录
                if len(d_legal)>1:
                    netin=[[],]
                    netin[0].append(cards_in_hand_oh(cards_list_list,self.cards_list))
                    netin[0].append(four_cards_oh(self.cards_on_table,c))
                    netin[0].append(score_oh(self.score,self.place))
                    netin[0]=torch.cat(netin[0])
                    netin.append(torch.tensor(score))
                    global for_o
                    for_o[len(self.cards_on_table)-1].append(netin)
                del score
            #clear temp_data
            del cards_list_1,cards_dict_1,cards_list_2,cards_dict_2,cards_list_3,cards_dict_3
        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice

    def pick_a_card(self):
        return self.pick_a_card_pure()
        #return self.pick_a_card_record_for_o()

    @staticmethod
    def family_name():
        return 'MrGreed'

def gen_data_for_o(N1=2,N2=1,save=False):
    from OfflineInterface import OfflineInterface
    import pickle
    global for_o
    for_o=([],[],[],[])
    g=[MrGreed(room=0,place=i,name='greed%d'%(i)) for i in range(4)]
    offlineinterface=OfflineInterface(g,print_flag=False)
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
        print(".",end=" ",flush=True)
        offlineinterface.prepare_new()
    print("")
    if save:
        with open("Greed_%d.data"%(N1),'wb') as f:
            pickle.dump(for_o,f)
        log("saved")
    return for_o

def optimize_target(paras):
    """
        will be called by optimize_para to optimize parameters of MrGreed
        should import:
    from MrIf import MrIf
    from OfflineInterface import OfflineInterface
    import numpy
    """
    print(paras,end=" ",flush=True)
    g0=MrGreed(room=0,place=0,name='greed0')
    g2=MrGreed(room=0,place=2,name='greed2')
    f1=MrIf(room=0,place=1,name="if1")
    f3=MrIf(room=0,place=3,name="if3")
    for g in [g0,g2]:
        g.SHORT_PREFERENCE=paras[0]*100
    offlineinterface=OfflineInterface([g0,f1,g2,f3],print_flag=False)
    N1=256;N2=2
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
        offlineinterface.prepare_new()
    s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    print("%.2f %.2f"%(numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1)),))
    return numpy.mean(s_temp)

def optimize_para():
    """
        want to optimize MrGreed's parameter by scipy.optimize, but failed
    """
    import scipy.optimize
    init_para=(8,)
    res=scipy.optimize.minimize(optimize_target,init_para,options={'eps':1})#,bounds=np.array([2,None]))
    print(res)

if __name__=="__main__":
    #optimize_para()
    gen_data_for_o(N1=32,save=True)