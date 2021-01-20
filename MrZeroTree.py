#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,calc_score
from Util import ORDER_DICT,ORDER_DICT2,ORDER_DICT5,SCORE_DICT,INIT_CARDS
from MrRandom import MrRandom
from MrGreed import MrGreed
from MrZeroTreeSimple import GameState,MrZeroTreeSimple,MCTS_EXPL,BETA
from ScenarioGenerator.ScenarioGen import ScenarioGen
from ScenarioGenerator.ImpScenarioGen import ImpScenarioGen
from OfflineInterface import OfflineInterface
from MCTS.mcts import mcts

import torch
import torch.nn.functional as F
import copy,math

print_level=0
BETA_POST_RECT=0.015
log("BETA_POST_RECT: %.3f"%(BETA_POST_RECT,))

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

    def cards_lists_oh_post_rect(cards_lists,place):
        """
            return a 208-length one hot, in raletive order
            the order is [me,me+1,me+2,me+3]
        """
        oh=torch.zeros(52*4)
        for c in cards_lists[place]:
            oh[ORDER_DICT[c]]=1
        for i in range(1,4):
            for c in cards_lists[(place+i)%4]:
                oh[52*1+ORDER_DICT[c]]=1/3
                oh[52*2+ORDER_DICT[c]]=1/3
                oh[52*3+ORDER_DICT[c]]=1/3
        return oh

    def prepare_ohs_post_rect(cards_lists,cards_on_table,score_lists,place):
        oh_card=MrZeroTree.cards_lists_oh_post_rect(cards_lists,place)
        oh_score=MrZeroTreeSimple.score_lists_oh(score_lists,place)
        oh_table=MrZeroTreeSimple.four_cards_oh(cards_on_table,place)
        return torch.cat([oh_card,oh_score,oh_table])

    def possi_rectify_pvnet(self,cards_lists,scores,cards_on_table,pnext,legal_choice,choice,restrict_flag=True):
        netin=MrZeroTree.prepare_ohs_post_rect(cards_lists,cards_on_table,scores,pnext)
        with torch.no_grad():
            p,_=self.pv_net(netin.to(self.device))
        if restrict_flag:
            p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice if c[0]==choice[0]]
        else:
            p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice]
        v_max=max((v for c,v in p_legal))
        p_legal=[(c,1+BETA_POST_RECT*(v-v_max)/BETA) for c,v in p_legal]
        """p_legal=[(c,math.exp(BETA_POST_RECT/BETA*(v-v_max))) for c,v in p_legal]
        v_sum=sum((v for c,v in p_legal))
        p_legal=[(c,v/v_sum) for c,v in p_legal]
        assert (sum((v for c,v in p_legal))-1)<1e-5, sum((v for c,v in p_legal))"""
        if print_level>=4:
            log(["%s: %.4f"%(c,v) for c,v in p_legal])

        p_choice=(v for c,v in p_legal if c==choice).__next__()
        #possi=confidence*p_choice+(1-confidence)/len(legal_choice)
        #possi=confidence*p_choice*len(legal_choice)+(1-confidence)
        #possi=p_choice*len(p_legal)
        possi=max(p_choice,0.2)
        return possi

    def decide_rect_necessity(self,thisuit,suit,choice,pnext,cards_lists):
        """
            return True for necessary
        """
        # C: 69.4(4.5)
        #if thisuit==choice[0] and choice[1] not in "234567":
        #    return True

        # C2:
        #if thisuit==choice[0] and choice[1] not in "23456":
        #    return True

        # C3:
        #if thisuit==choice[0] and choice[1] not in "2345678":
        #    return True

        # C4:
        #if thisuit==choice[0] and choice[1] not in "2345":
        #    return True

        # C5:
        #if thisuit==choice[0] and choice[1] not in "23456789":
        #    return True

        # D: 68.0(4.6)
        #if thisuit=="A" and choice[1] not in "234567":
        #    return True

        # D2
        #if thisuit=="A" and choice[1] not in "2345":
        #    return True

        # D3
        if thisuit=="A" and choice[1] not in "23456789":
            return True

        # F: 68.4(4.7) 67.2(4.7)
        #if suit!="A" and choice[0]!=suit:
        #    return True

        # F2: 65.6(4.4)
        #if thisuit!="A" and suit!="A" and choice[0]!=suit:
        #    return True

        # F3: 65.3(4.8)
        #if thisuit=="A" and suit!="A" and choice[0]!=suit:
        #    return True

        # H=C+F+following: 68.8(4.6)
        #if thisuit=="A" and suit=="A":
        #    return True

        # G=C+D+F: 59.2(4.5) 60.2(4.6) WHY?
        # K=C+F: 66.4(4.6)
        # L=C+D: 75.6(4.8)
        # M=D+F:

        # J: 63.0(4.7)
        #if thisuit=="A" and choice[1] not in "234567":
        #    if suit=="A":
        #        return True
        #    elif choice[0]!=suit:
        #        return True

        # J2: 62.7(4.8) 65.6(4.7)
        #if thisuit=="A":
        #    if suit=="A":
        #        return True
        #    elif choice[0]!=suit:
        #        return True
        return False

    def possi_rectify(self,cards_lists,thisuit):
        """
            posterior probability rectify
            cards_lists is in absolute order
        """
        #input("should not happen")
        cards_lists=copy.deepcopy(cards_lists)
        scores=copy.deepcopy(self.scores)
        result=1.0
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
                #只修正前6圈
                #if len(cards_lists[pnext])<8:
                #    continue
                #决定是否需要修正
                if len(cards_on_table)==1:
                    suit="A"
                else:
                    suit=cards_on_table[1][0]
                cards_dict=MrGreed.gen_cards_dict(cards_lists[pnext])
                legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,cards_lists[pnext])
                if not self.decide_rect_necessity(thisuit,suit,choice,pnext,cards_lists):
                    continue
                possi_pvnet=self.possi_rectify_pvnet(cards_lists,scores,cards_on_table,pnext,legal_choice,choice)
                if print_level>=4:
                    log("rectify: %s %s: %.4f"%(cards_on_table,choice,possi_pvnet),end="");input()
                result*=possi_pvnet
        else:
            assert len(scores[0])==len(scores[1])==len(scores[2])==len(scores[3])==0, scores
            assert len(cards_lists[0])==len(cards_lists[1])==len(cards_lists[2])==len(cards_lists[3])==13, cards_lists
        return result
        #return 1.0+sum(result)+(sum(result)**2-sum([i**2 for i in result]))/2

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

        if self.sample_k>=0:
            sce_num=self.sample_b+int(self.sample_k*len(self.cards_list))
            assert self.sample_b>=0 and sce_num>0
            sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=sce_num)
            scenarios=[i for i in sce_gen]
        else:
            assert self.sample_k<0 and self.sample_b<0
            sce_gen=ImpScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,suit,
                                   level=-1*self.sample_k,num_per_imp=-1*self.sample_b)
            scenarios=sce_gen.get_scenarios()

        scenarios_weight=[]
        cards_lists_list=[]
        for cll in scenarios:
            if print_level>=3:
                log("analyzing: %s"%(cll))
            cards_lists=[None,None,None,None]
            cards_lists[self.place]=copy.copy(self.cards_list)
            for i in range(3):
                cards_lists[(self.place+i+1)%4]=cll[i]
            scenarios_weight.append(self.possi_rectify(cards_lists,suit))
            #scenarios_weight.append(1.0)
            cards_lists_list.append(cards_lists)
            if print_level>=3:
                log("weight: %.4f"%(scenarios_weight[-1]),end="");input()
        else:
            del scenarios
        if print_level>=2:
            log("scenarios_weight: %s"%(["%.4f"%(i) for i in scenarios_weight],))
        weight_sum=sum(scenarios_weight)
        scenarios_weight=[i/weight_sum for i in scenarios_weight]

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
            elif self.mcts_k==-2:
                assert self.sample_b==1 and self.sample_k==0 and self.mcts_b==0, "This is raw-policy mode"
                netin=MrZeroTree.prepare_ohs_post_rect(cards_lists,self.cards_on_table,self.scores,self.place)
                with torch.no_grad():
                    p,_=self.pv_net(netin.to(self.device))
                p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice]
                p_legal.sort(key=lambda x:x[1],reverse=True)
                return p_legal[0][0]
            else:
                raise Exception("reserved")

        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice

    @staticmethod
    def family_name():
        return 'MrZeroTree'

def example_DJ():
    from MrZ_NETs import PV_NET_2
    from MrImpGreed import MrImpGreed
    device_bench=torch.device("cuda:2")
    state_dict=torch.load("Zero-29th-25-11416629-720.pt",map_location=device_bench)
    pv_net=PV_NET_2()
    pv_net.load_state_dict(state_dict)
    pv_net.to(device_bench)
    zt3=MrZeroTree(room=255,place=3,name='zerotree3',pv_net=pv_net,device=device_bench,mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-2)

    zt3.cards_list=["HQ","HJ","H8","SA","S5","S4","S3","CQ","CJ","C4"]
    zt3.cards_on_table=[1,"DJ","D8"]
    zt3.history=[[0,"H3","H5","H4","H7"],[3,"S6","SJ","HK","S10"],[0,"DQ","DA","D9","D3"]]
    zt3.scores=[["HK"],[],[],["H3","H5","H4","H7"]]
    log(zt3.pick_a_card())
    return
    l=[zt3.pick_a_card() for i in range(20)]
    log("%d %d %d"%(len([i[0] for i in l if i[0]=="H"]),len([i[0] for i in l if i[0]=="C"]),len([i[0] for i in l if i[0]=="S"])))


def example_SQ():
    #not include so far
    from MrZ_NETs import PV_NET_2
    from MrImpGreed import MrImpGreed
    device_bench=torch.device("cuda:2")
    state_dict=torch.load("Zero-29th-25-11416629-720.pt",map_location=device_bench)
    pv_net=PV_NET_2()
    pv_net.load_state_dict(state_dict)
    pv_net.to(device_bench)
    zt3=MrZeroTree(room=255,place=3,name='zerotree3',pv_net=pv_net,device=device_bench,mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-2)

    zt3.cards_list=["HQ","HJ","H8","H7","SA","S6","S5","S4","S3","CQ","CJ","D3"]
    zt3.cards_on_table=[1,"S7","SJ"]
    zt3.history=[[1,"C9","C7","C4","H9"],]
    zt3.scores=[[],["H9"],[],[]]
    log(zt3.pick_a_card())

def example_SQ2():
    from MrZ_NETs import PV_NET_2
    from MrImpGreed import MrImpGreed
    device_bench=torch.device("cuda:2")
    state_dict=torch.load("Zero-29th-25-11416629-720.pt",map_location=device_bench)
    pv_net=PV_NET_2()
    pv_net.load_state_dict(state_dict)
    pv_net.to(device_bench)
    zt3=MrZeroTree(room=255,place=3,name='zerotree3',pv_net=pv_net,device=device_bench,mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-1)

    zt3.cards_list=["HQ","HJ","H8","H7","SA","SJ","S4","S3","CQ","D3","D10"]
    zt3.cards_on_table=[2,"S6"]
    zt3.history=[[0,"S7","SK","S10","S8"],[1,"C2","C9","C8","C7"]]
    zt3.scores=[[],[],[],[]]
    log(zt3.pick_a_card())

if __name__=="__main__":
    #example_DJ()
    example_SQ2()

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