#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import ORDER_DICT2,SCORE_DICT
from MrRandom import MrRandom
#from MrO_Trainer import cards_in_hand_oh,score_oh,four_cards_oh #a failed module
from ScenarioGenerator.ScenarioGen import ScenarioGen
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
            in absolute order!
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
        fmt_score_list,trick_start,scs_rmn_avg,impc_dict_base,myplace,need_details=False):
        impc_dict=MrGreed.diy_impc_dict(impc_dict_base,cards_list)
        best_score=-65535
        four_cards_tmp=four_cards[0:3]+['']
        if need_details:
            d_return={}
        for c in MrGreed.gen_legal_choice(suit,cards_dict,cards_list):
            four_cards_tmp[3]=c
            score_temp=MrGreed.clear_score(four_cards_tmp,fmt_score_list,trick_start,scs_rmn_avg)\
                      +MrGreed.calc_relief(c,impc_dict,scs_rmn_avg,fmt_score_list[myplace][0])
            if score_temp>best_score:
                four_cards[3]=four_cards_tmp[3]
                best_score=score_temp
            if need_details:
                d_return[c]=score_temp
        if need_details:
            return d_return #add d_return for training MrZeroTree

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

    def pick_best_from_dlegal(d_legal):
        """
            pick best choice from d_legal
            will be called by pick_a_card only once
        """
        best_score=float("-inf")
        for k,v in d_legal.items():
            if v>best_score:
                best_score=v
                best_choices=[k]
            elif v==best_score:
                best_choices.append(k)
        """if len(best_choices)>1:
            log("have to choice: %s"%(d_legal))
            for i in range(10):
                print(random.choice(best_choices))"""
        return random.choice(best_choices)

    def pick_a_card(self,need_details=False):
        #确认桌上牌的数量和自己坐的位置相符
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place
        #utility datas
        suit=self.decide_suit()
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        #如果别无选择
        if cards_dict.get(suit)!=None and len(cards_dict[suit])==1:
            choice=cards_dict[suit][0]
            if need_details:
                return choice,None
            else:
                return choice
        #more utility datas
        fmt_score_list=MrGreed.gen_fmt_scores(self.scores) #in absolute order， because self.scores is in absolute order
        impc_dict_base=MrGreed.gen_impc_dict(self.scores,self.cards_on_table)
        scs_rmn_avg=(-200-sum([i[0] for i in fmt_score_list]))//4

        #如果我是最后一个出的
        if len(self.cards_on_table)==4:
            four_cards=self.cards_on_table[1:4]+[""]
            if need_details:
                d_return=MrGreed.as_last_player(suit,four_cards,cards_dict,self.cards_list
                    ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict_base,self.place,need_details=True)
                choice=four_cards[3]
                return choice,d_return
            else:
                MrGreed.as_last_player(suit,four_cards,cards_dict,self.cards_list
                    ,fmt_score_list,self.cards_on_table[0],scs_rmn_avg,impc_dict_base,self.place)
                choice=four_cards[3]
                return choice

        #more utility datas
        impc_dict_mine=MrGreed.diy_impc_dict(impc_dict_base,self.cards_list)
        #如果我是倒数第二个
        if len(self.cards_on_table)==3:
            four_cards=self.cards_on_table[1:3]+['','']
        #如果我是倒数第三个
        elif len(self.cards_on_table)==2:
            four_cards=[self.cards_on_table[1],'','','']
        #如果我是第一个出
        elif len(self.cards_on_table)==1:
            four_cards=['','','','']
        #expire_date=0 #for sampling
        d_legal={c:0 for c in MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)} #dict of legal choice
        sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=MrGreed.N_SAMPLE)
        for cards_list_list in sce_gen:
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

        if need_details:
            #scores_had=[MrGreed.clear_fmt_score(fmt_score_list[(self.place+i)%4]) for i in range(4)]
            #scores_had=scores_had[0]+scores_had[2]-scores_had[1]-scores_had[3]
            d_return={c:d_legal[c]/MrGreed.N_SAMPLE for c in d_legal} #todo, no need for scs_rmn_avg
            #log("scores had: %s"%(scores_had))
            #log(d_return)
            #log(d_legal)
            return best_choice,d_return
        else:
            return best_choice

    def clear_fmt_score(fmt_score):
        """
            [0,0,False,False] stands for [score, #hearts, C10 flag, has score flag]
        """
        s=fmt_score[0]
        if fmt_score[1]==13:
            s+=400
        if fmt_score[2]:
            if fmt_score[3]:
                s*=2
            else:
                assert fmt_score[0]==0
                assert fmt_score[1]==0
                s=50
        return s

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

def bug_shi():
    g=MrGreed(room=0,place=1,name='shi')
    g.cards_on_table=[0,'C4']
    g.history=[[0, 'C10', 'C5', 'CA', 'C9'], [2, 'DK', 'D10', 'D7', 'D8'], [2, 'DJ', 'DQ', 'D5', 'DA'],
               [1, 'H5', 'H6', 'H4', 'H10'], [0, 'H8', 'H3', 'HJ', 'H9'], [2, 'CK', 'HA', 'CJ', 'CQ'],
               [2, 'S9', 'S3', 'S10', 'SJ'], [1, 'C2', 'C7', 'HK', 'C8'], [0, 'S6', 'S4', 'S5', 'H7']]
    g.scores=[['H5', 'H6', 'H4', 'H10', 'HK', 'H7'], ['DJ'], ['C10', 'H8', 'H3', 'HJ', 'H9', 'HA'], []]
    g.cards_list=['D4', 'D6', 'D9', 'HQ']
    print(g.pick_a_card())

if __name__=="__main__":
    #optimize_para()
    bug_shi()