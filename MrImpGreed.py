#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import ORDER_DICT2,SCORE_DICT
from MrGreed import MrGreed
from ScenarioGenerator.ImpScenarioGen import ImpScenarioGen
import itertools

class MrImpGreed(MrGreed):
    L_SAMPLE=2
    N_PER_IMP=5

    def pick_a_card(self,sce_gen=None):
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
        fmt_score_list=MrGreed.gen_fmt_scores(self.scores) #in absolute order， because self.scores is in absolute order
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
        if sce_gen==None:
            sce_gen=ImpScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,level=MrImpGreed.L_SAMPLE,num_per_imp=MrImpGreed.N_PER_IMP)
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

        return best_choice

    @staticmethod
    def family_name():
        return 'MrImpGreed'

def benchmark(print_process=True):
    from OfflineInterface import OfflineInterface
    import itertools,numpy

    ig=[MrImpGreed(room=255,place=i,name='impgreed%d'%(i)) for i in [0,2]]
    g=[MrGreed(room=255,place=i,name='greed%d'%(i)) for i in [1,3]]
    interface=OfflineInterface([ig[0],g[0],ig[1],g[1]],print_flag=False)

    N1=256;N2=2;
    log("%s v.s. %s for %dx%d"%(interface.players[0].__class__.__name__,interface.players[1].__class__.__name__,N1,N2))
    stats=[]
    for k,l in itertools.product(range(N1),range(N2)):
        if l==0:
            cards=interface.shuffle()
        else:
            cards=cards[39:52]+cards[0:39]
            interface.shuffle(cards=cards)
        for i,j in itertools.product(range(13),range(4)):
            interface.step()
            #input("continue...")
        stats.append(interface.clear())
        interface.prepare_new()
        if l==N2-1:
            if print_process:
                log("%2d %4d: %s"%(k,sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2,stats[-N2:]))
            else:
                print("%4d"%(sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2),end=" ",flush=True)
    s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    log(s_temp[0:8])
    s_temp=[sum(s_temp[i:i+N2])/N2 for i in range(0,len(s_temp),N2)]
    log(s_temp[0:4])
    log("benchmark result: %.2f %.2f"%(numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1))))

if __name__=="__main__":
    benchmark(print_process=True)