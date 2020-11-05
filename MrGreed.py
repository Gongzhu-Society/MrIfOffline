#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import ORDER_DICT2,INIT_CARDS,SCORE_DICT
from MrRandom import MrRandom
import random,numpy

print_level=0

class MrGreed(MrRandom):
    BURDEN_DICT={'SA':11,'SK':9,'SQ':8,'SJ':7,'S10':6,'S9':5,'S8':4,'S7':3,'S6':2,'S5':1,'S4':1,
                 'CA':11,'CK':9,'CQ':8,'CJ':7,'C10':6,'C9':5,'C8':4,'C7':3,'C6':2,'C5':1,'C4':1,
                 'DA':11,'DK':9,'DQ':8,'DJ':7,'D10':6,'D9':5,'D8':4,'D7':3,'D6':2,'D5':1,'D4':1,
                 'H10':6,'H9':5,'H8':4,'H7':3,'H6':2,'H5':1,'H4':1}
    BURDEN_DICT_S={'SA':40,'SK':30}
    BURDEN_DICT_D={'DA':-30,'DK':-20,'DQ':-10}
    BURDEN_DICT_C={'CA':0.4,'CK':0.3,'CQ':0.2,'CJ':0.1} #ratio of burden, see calc_relief
    N_SAMPLE=20

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
        fmt_score_list,trick_start,scs_rmn_avg,impc_dict,myplace):
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
        fmt_score_list,trick_start,scs_rmn_avg,impc_dict,myplace):
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
        fmt_score_list,trick_start,scs_rmn_avg,impc_dict,myplace):
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
        best_choice,best_score=d_legal.popitem()
        for k in d_legal:
            if d_legal[k]>best_score:
                best_choice=k
                best_score=d_legal[k]
        return best_choice

    def gen_void_info(myseat,history,cards_on_table):
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
        global print_level
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

    def pick_a_card(self):
        global print_level
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place
        suit=self.decide_suit()
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)

        #如果别无选择
        if cards_dict.get(suit)!=None and len(cards_dict[suit])==1:
            choice=cards_dict[suit][0]
            if print_level>=1:
                log("I have no choice but %s"%(choice))
            return choice


        fmt_score_list=MrGreed.gen_fmt_scores(self.scores)
        impc_dict=MrGreed.gen_impc_dict(self.scores,self.cards_on_table) #TODO: relief!!!!
        for c in self.cards_list:
            if c=='SQ':
                impc_dict['SQ']=True
            elif c=='DJ':
                impc_dict['DJ']=True
            elif c=='C10':
                impc_dict['C10']=True
        scs_rmn_avg=(-200-sum([i[0] for i in fmt_score_list]))//4

        #如果我是最后一个出的
        if len(self.cards_on_table)==4:
            four_cards=self.cards_on_table[1:4]+[""]
            MrGreed.as_last_player(suit,four_cards,cards_dict,self.cards_list
                                  ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict,self.place)
            choice=four_cards[3]
            return choice

        #其他情况要估计先验概率了
        void_info=MrGreed.gen_void_info(self.place,self.history,self.cards_on_table)
        cards_remain=MrGreed.calc_cards_remain(self.history,self.cards_on_table,self.cards_list)
        d_legal={} #wtf???
        for c in MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list):
            d_legal[c]=0
        expire_date=0 #wtf???
        #如果我是倒数第二个
        if len(self.cards_on_table)==3:
            assert len(cards_remain)==3*len(self.cards_list)-2
            lens=[len(self.cards_list),len(self.cards_list)-1,len(self.cards_list)-1]
            four_cards=self.cards_on_table[1:3]+['','']
            for ax in range(MrGreed.N_SAMPLE):
                if expire_date==0:
                    cards_list_list,exchange_info,bx=MrGreed.gen_scenario(void_info,cards_remain,lens)
                    expire_date=max(bx-5,0)
                    #log("expire_date set: %d"%(expire_date))
                else:
                    exhausted_flag=MrGreed.alter_scenario(cards_list_list,exchange_info,void_info)
                    if exhausted_flag==1:
                        #log("exhausted, break ax=%d"%(ax))
                        break
                    expire_date-=1
                cards_list_1=cards_list_list[0]
                cards_dict_1=MrGreed.gen_cards_dict(cards_list_1)
                if print_level>=2:
                    log("gen scenario: %s"%(cards_list_1))
                for c in d_legal:
                    four_cards[2]=c
                    MrGreed.as_last_player(suit,four_cards,cards_dict_1,cards_list_1
                                          ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict,self.place)
                    score=-1*MrGreed.clear_score(four_cards,fmt_score_list,self.cards_on_table[0],scs_rmn_avg)\
                         +MrGreed.calc_relief(c,impc_dict,scs_rmn_avg,fmt_score_list[self.place][0])
                    if print_level>=2:
                        log("If I choose %s: %s, %d"%(c,four_cards,score))
                    d_legal[c]+=score
        #如果我是倒数第三个
        elif len(self.cards_on_table)==2:
            assert len(cards_remain)==3*len(self.cards_list)-1
            lens=[len(self.cards_list),len(self.cards_list),len(self.cards_list)-1]
            four_cards=[self.cards_on_table[1],'','','']
            for ax in range(MrGreed.N_SAMPLE):
                if expire_date==0:
                    cards_list_list,exchange_info,bx=MrGreed.gen_scenario(void_info,cards_remain,lens)
                    expire_date=max(bx-5,0)
                    #log("expire_date set: %d"%(expire_date))
                else:
                    exhausted_flag=MrGreed.alter_scenario(cards_list_list,exchange_info,void_info)
                    if exhausted_flag==1:
                        #log("exhausted, break ax=%d"%(ax))
                        break
                    expire_date-=1
                cards_list_1=cards_list_list[0]
                cards_dict_1=MrGreed.gen_cards_dict(cards_list_1)
                cards_list_2=cards_list_list[1]
                cards_dict_2=MrGreed.gen_cards_dict(cards_list_2)
                if print_level>=2:
                    log("gen scenario: %s, %s"%(cards_list_1,cards_list_2))
                for c in d_legal:
                    four_cards[1]=c
                    MrGreed.as_third_player(suit,four_cards,cards_dict_1,cards_list_1,cards_dict_2,cards_list_2
                                           ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict,self.place)
                    score=MrGreed.clear_score(four_cards,fmt_score_list,self.cards_on_table[0],scs_rmn_avg)\
                         +MrGreed.calc_relief(c,impc_dict,scs_rmn_avg,fmt_score_list[self.place][0])
                    if print_level>=2:
                        log("If I choose %s: %s, %d"%(c,four_cards,score))
                    d_legal[c]+=score
        #如果我是第一个出
        elif len(self.cards_on_table)==1:
            assert len(cards_remain)==3*len(self.cards_list)
            lens=[len(self.cards_list),len(self.cards_list),len(self.cards_list)]
            d_suit_extra={'S':0,'H':0,'D':0,'C':0}
            if len(self.history)<3:
                l_temp=[]
                for h in self.history:
                    for c in h[1:5]:
                        l_temp.append(c[0])
                s_temp=''.join(l_temp)
                for s in 'SHDC':
                    my_len=len(cards_dict[s])
                    avg_len=(13-s_temp.count(s)-my_len)/3
                    d_suit_extra[s]=int((avg_len-my_len)*20) #a_free_para_meter_here
            four_cards=['','','','']
            for ax in range(MrGreed.N_SAMPLE):
                if expire_date==0:
                    cards_list_list,exchange_info,bx=MrGreed.gen_scenario(void_info,cards_remain,lens)
                    expire_date=max(bx-5,0)
                    #log("expire_date set: %d"%(expire_date))
                else:
                    exhausted_flag=MrGreed.alter_scenario(cards_list_list,exchange_info,void_info)
                    if exhausted_flag==1:
                        #log("exhausted, break ax=%d"%(ax))
                        break
                    expire_date-=1
                cards_list_1=cards_list_list[0]
                cards_dict_1=MrGreed.gen_cards_dict(cards_list_1)
                cards_list_2=cards_list_list[1]
                cards_dict_2=MrGreed.gen_cards_dict(cards_list_2)
                cards_list_3=cards_list_list[2]
                cards_dict_3=MrGreed.gen_cards_dict(cards_list_3)
                if print_level>=2:
                    log("gen scenario: %s, %s, %s"%(cards_list_1,cards_list_2,cards_list_3))
                for c in d_legal:
                    four_cards[0]=c
                    MrGreed.as_second_player(c[0],four_cards,cards_dict_1,cards_list_1,cards_dict_2,cards_list_2,cards_dict_3,cards_list_3
                                            ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict,self.place)
                    score=-1*MrGreed.clear_score(four_cards,fmt_score_list,self.cards_on_table[0],scs_rmn_avg)\
                         +MrGreed.calc_relief(c,impc_dict,scs_rmn_avg,fmt_score_list[self.place][0])
                    if print_level>=2:
                        log("If I choose %s: %s, %d"%(c,four_cards,score))
                    d_legal[c]+=score
            for c in d_legal:
                d_legal[c]+=d_suit_extra[c[0]]*(ax+1)
        if print_level>=1:
            log(d_legal)
        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice

    @staticmethod
    def family_name():
        return 'MrGreed'

if __name__=="__main__":
    pass