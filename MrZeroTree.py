#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,calc_score
from Util import ORDER_DICT,ORDER_DICT2,ORDER_DICT5,SCORE_DICT,INIT_CARDS
from MrRandom import MrRandom
from MrGreed import MrGreed
from MrZeroTreeSimple import GameState,MrZeroTreeSimple,MCTS_EXPL
from ScenarioGenerator.ScenarioGen import ScenarioGen
#from ScenarioGenerator.ImpScenarioGen import ImpScenarioGen
from OfflineInterface import OfflineInterface
from MCTS.mcts import mcts

import torch
import torch.nn.functional as F
import copy,math

print_level=0

class MrZeroTree(MrZeroTreeSimple):
    def __init__(self,room=0,place=0,name="default",pv_net=None,device=None,train_mode=False,
                 sample_b=5,sample_k=0,mcts_b=10,mcts_k=2):
        MrRandom.__init__(self,room,place,name)
        self.pv_net=pv_net
        self.device=device
        self.sample_b=sample_b
        self.sample_k=sample_k
        self.mcts_b=mcts_b
        self.mcts_k=mcts_k
        self.train_mode=train_mode
        if self.train_mode:
            self.train_datas=[]
    
    def possi_rectify_pvnet(self,cards_lists,scores,cards_on_table,pnext,legal_choice,choice,confidence):
        netin=MrZeroTree.prepare_ohs(cards_lists,cards_on_table,scores,pnext)
        with torch.no_grad():
            p,_=self.pv_net(netin.to(self.device))
        p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice]
        v_max=max((v for c,v in p_legal))
        p_legal=[(c,math.exp(v-v_max)) for c,v in p_legal]
        v_sum=sum((v for c,v in p_legal))
        p_legal=[(c,v/v_sum) for c,v in p_legal]
        assert (sum((v for c,v in p_legal))-1)<1e-5, sum((v for c,v in p_legal))

        p_choice=(v for c,v in p_legal if c==choice).__next__()
        #possi=confidence*p_choice+(1-confidence)/len(legal_choice)
        possi=confidence*p_choice*len(legal_choice)+(1-confidence)
        if print_level>=3:
            log(possi);input()
        return possi

    def possi_rectify_greed(self,cards_lists,scores,cards_on_table,pnext,legal_choice,choice,confidence=0.6):
        g=self.g_aux[pnext]
        g.cards_on_table=copy.copy(cards_on_table)
        g.scores=copy.deepcopy(scores)
        g.cards_list=copy.deepcopy(cards_lists[pnext])
        g_choice=g.pick_a_card(sce_gen=[[cards_lists[(pnext+1)%4],cards_lists[(pnext+2)%4],cards_lists[(pnext+3)%4]]])
        if choice==g_choice:
            #log("%s==%s"%(choice,g_choice))
            return confidence/(1-2*confidence*(1-confidence)) #renormalization factor: 1/(c**2+(1-c)**2)
        else:
            #log("%s!=%s"%(choice,g_choice))
            return (1-confidence)/(1-2*confidence*(1-confidence))
   
    def decide_rect_necessity(self,thisuit,suit,choice,pnext,cards_lists):
        """
            return True for necessary
        """
        """do_flag=False
                #if thisuit in ("S","A") and choice[0]=="S" and "SQ" in cards_lists_origin[pnext]:
                if thisuit in ("S","A") and choice[0]=="S"\
                and len(set(cards_lists_origin[pnext]).intersection(set(["SQ","SK","SA"])))>0:
                    do_flag=True
                #elif thisuit in ("D","A") and choice[0]=="D"\
                and "DJ" in cards_lists_origin[pnext]:
                elif thisuit in ("D",) and choice[0]=="D"\
                and len(set(cards_lists_origin[pnext]).intersection(set(["DJ","DA","DK","DQ"])))>0:
                    do_flag=True
                elif thisuit in ("C") and choice[0]=="C"\
                and "C10" in cards_lists_origin[pnext]:
                    do_flag=True
                elif thisuit in ("H",) and choice[0]=="H"\
                and len(set(cards_lists_origin[pnext]).intersection(set(["HA","HK","HQ"])))>0:
                    do_flag=True
                if (not do_flag) or (not same_flag):
                    continue"""
        """if len(cards_on_table)!=5:
                same_flag=False
            else:
                same_flag=True
                for i in range(4):
                    if (cards_on_table[0]+i)%4==self.place:
                        continue
                    if cards_on_table[i+1][0]!=cards_on_table[1][0]:
                        same_flag=False"""
        if thisuit==choice[0] and choice[1] not in "234":
            return True
        if thisuit=="A" and choice in ("SA","SK","DA","DK","DQ","HA","HK","HQ","HJ"):
            return True
    
        return False

    def possi_rectify(self,cards_lists,thisuit):
        """
            posterior probability rectify
            cards_lists is in absolute order
        """
        #log(cards_lists)
        cards_lists=copy.deepcopy(cards_lists)
        scores=copy.deepcopy(self.scores)
        result=1
        for history in [self.cards_on_table,]+self.history[::-1]:
            if len(history)==5:
                for c in history[1:]:
                    if c in SCORE_DICT:
                        scores[last_winner].remove(c)
            last_winner=history[0]
            cards_on_table=copy.copy(history)
            pnext=(cards_on_table[0]+len(history)-1)%4
            for i in range(len(cards_on_table)-1):
                pnext=(pnext-1)%4
                choice=cards_on_table.pop()
                cards_lists[pnext].append(choice)
                #不用修正我自己
                if pnext==self.place:
                    continue
                #决定是否需要修正
                if len(cards_on_table)==1:
                    suit="A"
                else:
                    suit=cards_on_table[1][0]
                cards_dict=MrGreed.gen_cards_dict(cards_lists[pnext])
                legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,cards_lists[pnext])
                if not self.decide_rect_necessity(thisuit,suit,choice,pnext,cards_lists):
                    continue
                result*=self.possi_rectify_greed(cards_lists,scores,cards_on_table,pnext,legal_choice,choice)
                #log("rectifying: %s %s %.4f"%(cards_on_table,choice,result));input()
        else:
            assert len(scores[0])==len(scores[1])==len(scores[2])==len(scores[3])==0, scores
            assert len(cards_lists[0])==len(cards_lists[1])==len(cards_lists[2])==len(cards_lists[3])==13, cards_lists
        return result

    def pick_a_card(self):
        #确认桌上牌的数量和自己坐的位置相符
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place
        #utility datas
        suit=self.decide_suit() #inherited from MrRandom
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        #如果别无选择
        if cards_dict.get(suit)!=None and len(cards_dict[suit])==1:
            choice=cards_dict[suit][0]
            if print_level>=1:
                log("I have no choice but %s"%(choice))
            return choice
        if  len(self.cards_list)==1:
            return self.cards_list[0]
        if print_level>=1:
            log("my turn: %s, %s, %s"%(self.cards_on_table,self.cards_list,self.scores))

        if self.sample_b>=0 and self.sample_k>=0:
            sce_num=self.sample_b+int(self.sample_k*len(self.cards_list))
            sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=sce_num)
            scenarios=[i for i in sce_gen]
        elif self.sample_b<0 and self.sample_k<0:
            input("not using")
            sce_gen=ImpScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,
                                   level=-1*self.sample_k,num_per_imp=-1*self.sample_b)
            scenarios=sce_gen.get_scenarios()

        scenarios_weight=[]
        cards_lists_list=[]
        for cll in scenarios:
            cards_lists=[None,None,None,None]
            cards_lists[self.place]=copy.copy(self.cards_list)
            for i in range(3):
                cards_lists[(self.place+i+1)%4]=cll[i]
            scenarios_weight.append(self.possi_rectify(cards_lists,suit))
            #scenarios_weight.append(1.0)
            cards_lists_list.append(cards_lists)
            if print_level>=3:
                log("weight: %.4e\n%s"%(scenarios_weight[-1],cards_lists))
        else:
            del scenarios
        if print_level>=2:
            log("scenarios_weight: %s"%(scenarios_weight,))
        weight_sum=sum(scenarios_weight)
        scenarios_weight=[i/weight_sum for i in scenarios_weight]
        if print_level>=2:
            log("scenarios_weight: %s"%(scenarios_weight,))
        
        legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)
        d_legal={c:0 for c in legal_choice}
        searchnum=self.mcts_b+self.mcts_k*len(legal_choice)
        for i,cards_lists in enumerate(cards_lists_list):
            #initialize gamestate
            gamestate=GameState(cards_lists,self.scores,self.cards_on_table,self.place)
            #mcts
            if self.mcts_k>=0:
                searcher=mcts(iterationLimit=searchnum,rolloutPolicy=self.pv_policy,
                              explorationConstant=MCTS_EXPL)
                searcher.search(initialState=gamestate)
                for action,node in searcher.root.children.items():
                    d_legal[action]+=scenarios_weight[i]*node.totalReward/node.numVisits
            elif self.mcts_k==-1:
                input("not using")
                netin=MrZeroTree.prepare_ohs(cards_lists,self.cards_on_table,self.scores,self.place)
                with torch.no_grad():
                    p,_=self.pv_net(netin.to(self.device))
                p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice]
                p_legal.sort(key=lambda x:x[1],reverse=True)
                d_legal[p_legal[0][0]]+=1
            else:
                raise Exception("reserved")  

        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice
    
    @staticmethod
    def family_name():
        return 'MrZeroTree'

if __name__=="__main__":
    pass

    """ 统计重复率
            netin=MrZeroTree.prepare_ohs(cards_lists,self.cards_on_table,self.scores,self.place)
            with torch.no_grad():
                p,_=self.pv_net(netin.to(self.device))
            p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice]
            v_max=max((v for c,v in p_legal))
            p_legal=[(c,math.exp(v-v_max)) for c,v in p_legal]
            v_sum=sum((v for c,v in p_legal))
            p_legal=[(c,v/v_sum) for c,v in p_legal]
            assert (sum((v for c,v in p_legal))-1)<1e-5, sum((v for c,v in p_legal))
            p_legal.sort(key=lambda x: x[1],reverse=True)
            
            choice=[(action,node.totalReward/node.numVisits) for action,node in searcher.root.children.items()]
            choice.sort(key=lambda x:x[1],reverse=True)
            choice=choice[0][0]
            print("%d"%(p_legal[0][0]==choice),end=", ",flush=True)
            #print("%.4f"%([v for c,v in p_legal if c==choice][0]),end=", ",flush=True)

            g=self.g_aux[self.place]
            g.cards_on_table=copy.copy(self.cards_on_table)
            g.history=copy.deepcopy(self.history)
            g.scores=copy.deepcopy(self.scores)
            g.cards_list=copy.deepcopy(self.cards_list)
            gc=g.pick_a_card(sce_gen=[scenarios[i]])
            print("%d"%(p_legal[0][0]==gc,),end=", ",flush=True)
            #print("%.4f"%([v for c,v in p_legal if c==gc][0]),end=", ",flush=True)"""

    """ 挑5个最好的
        scenarios_weight_bk=copy.copy(scenarios_weight)
        scenarios_weight_bk.sort(reverse=True)
        sel_num=min(4,len(scenarios_weight_bk)-1)
        weight_thres=scenarios_weight_bk[sel_num];del scenarios_weight_bk
        temp=[]
        for i,cards_lists in enumerate(cards_lists_list):
            if scenarios_weight[i]>=weight_thres:
                temp.append((scenarios_weight[i],cards_lists))
            if len(temp)>=sel_num+1:
                break
        del scenarios_weight,cards_lists_list
        scenarios_weight=[1.0]*(sel_num+1)
        cards_lists_list=[j for i,j in temp]
        assert len(scenarios_weight)==len(cards_lists_list), "%d %d"%(len(scenarios_weight),len(cards_lists_list))"""