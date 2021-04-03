#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,SCORE_DICT,calc_score,calc_score_midway
from MrRandom import MrRandom # Father class of all AIs
from MrGreed import MrGreed # For its Util funcs
from MrZeroTreeSimple import GameState
from ScenarioGenerator.ScenarioGen import ScenarioGen
from MCTS.mcts import abpruning,mcts
import time,copy,numpy

# print_level meaning
# 0: nothing
# 1:
# 2: gross search result
# 3: into each search cases
print_level=1

class MrAbTree(MrRandom):
    BURDEN_DICT={'SA':11,'SK':9,'SQ':8,'SJ':7,'S10':6,'S9':5,'S8':4,'S7':3,'S6':2,'S5':1,'S4':1,
                 'CA':11,'CK':9,'CQ':8,'CJ':7,'C10':6,'C9':5,'C8':4,'C7':3,'C6':2,'C5':1,'C4':1,
                 'DA':11,'DK':9,'DQ':8,'DJ':7,'D10':6,'D9':5,'D8':4,'D7':3,'D6':2,'D5':1,'D4':1,
                 'H10':6,'H9':5,'H8':4,'H7':3,'H6':2,'H5':1,'H4':1}
    BURDEN_DICT_S={'SA':50,'SK':30}
    BURDEN_DICT_D={'DA':-30,'DK':-20,'DQ':-10}
    BURDEN_DICT_C={'CA':0.4,'CK':0.3,'CQ':0.2,'CJ':0.1} #ratio of burden, see calc_relief
    SHORT_PREFERENCE=30 #will multiply (average suit count)-(my suit count), if play first

    def __init__(self,room=0,place=0,name="MrAbTree",sample_b=5,trick_deep=2):
        MrRandom.__init__(self,room,place,name)
        self.sample_b=sample_b
        self.trick_deep=trick_deep

    def calc_burden(cards_lists,play_for,score_remain_avg,score_lists):
        cards_list=cards_lists[play_for]
        burden=sum([MrAbTree.BURDEN_DICT.get(i,0) for i in cards_list])

        cards_remain=set()
        for i in range(1,4):
            cards_remain.update(cards_lists[(play_for+i)%4])

        if 'SQ' in cards_remain:
            burden+=sum([MrAbTree.BURDEN_DICT_S.get(i,0) for i in cards_list])
        if 'DJ' in cards_remain:
            burden+=sum([MrAbTree.BURDEN_DICT_D.get(i,0) for i in cards_list])
        if 'C10' in cards_remain:
            burden-=sum([MrAbTree.BURDEN_DICT_C.get(i,0)*score_remain_avg for i in cards_list])
        elif 'C10' in score_lists[play_for]:
            burden-=score_remain_avg
        return burden

    def ab_policy(self,state):
        assert len(state.cards_on_table)==1
        if state.isTerminal():
            scores=[calc_score(state.score_lists[(state.play_for+i)%4]) for i in range(4)]
            return scores[0]+scores[2]-scores[1]-scores[3]
        else:
            scards_played=sum([len(i) for i in state.score_lists])
            scores=[calc_score_midway(state.score_lists[(state.play_for+i)%4],scards_played) for i in range(4)]
            scores=scores[0]+scores[2]-scores[1]-scores[3]

            score_remain_avg=sum([SCORE_DICT.get(i,0)  for j in range(4) for i in state.score_lists[j]])/4
            burdens=[MrAbTree.calc_burden(state.cards_lists,(state.play_for+i)%4,score_remain_avg,state.score_lists) for i in range(4)]
            burdens=burdens[0]+burdens[2]-burdens[1]-burdens[3]
            #burden_0=MrAbTree.calc_burden(state.cards_lists,state.play_for,score_remain,state.score_lists)
            #burden_2=MrAbTree.calc_burden(state.cards_lists,(state.play_for+2)%4,score_remain,state.score_lists)
            #burdens=burden_0+burden_2
            #burdens=burden_0

            return scores-burdens*len(state.cards_lists[state.play_for])/12


    def pick_a_card(self):
        #utility datas
        suit=self.decide_suit() #inherited from MrRandom
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        # If there is no choice
        if cards_dict.get(suit)!=None and len(cards_dict[suit])==1:
            choice=cards_dict[suit][0]
            if print_level>=1:
                log("I have no choice but %s."%(choice))
            return choice
        # If this is the last trick
        if  len(self.cards_list)==1:
            if print_level>=1:
                log("There is only one card left.")
            return self.cards_list[0]

        #生成Scenario
        sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=self.sample_b)
        cards_lists_list=[]
        for cll in sce_gen:
            cards_lists=[None,None,None,None]
            cards_lists[self.place]=copy.copy(self.cards_list)
            for i in range(3):
                cards_lists[(self.place+i+1)%4]=cll[i]
            cards_lists_list.append(cards_lists)

        #对Scenario平均
        legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)
        d_legal={c:0 for c in legal_choice}
        if print_level>=1:
            log("my turn: %s, %s, %s"%(self.cards_on_table,legal_choice,self.scores))
        tree_deep=self.trick_deep*4-len(self.cards_on_table)+1
        searcher=abpruning(deep=tree_deep,rolloutPolicy=self.ab_policy)
        if print_level>=2:
            log("tree_deep: %d"%(tree_deep))

        for i,cards_lists in enumerate(cards_lists_list):
            if print_level>=3:
                log("considering case %d: %s"%(i,cards_lists));input()
            gamestate=GameState(cards_lists,self.scores,self.cards_on_table,self.history,self.place)
            searcher.search(initialState=gamestate)
            for action,val in searcher.children.items():
                d_legal[action]+=val
        if print_level>=2:
            log("searched %.1f cases"%(searcher.counter/self.sample_b))

        #best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        best_choice=max(d_legal.items(),key=lambda x:x[1])[0]
        if print_level>=1:
            log("%s: %s"%(best_choice,{k:"%.1f"%(v/self.sample_b) for k,v in d_legal.items()}),end="");input()
        return best_choice

def benchmark(handsfile):
    from OfflineInterface import OfflineInterface,read_std_hands,play_a_test

    g=[MrGreed(room=0,place=i,name='g%d'%(i)) for i in range(4)]
    abt=[MrAbTree(room=0,place=i,name='abt%d'%(i),trick_deep=1,sample_b=5) for i in range(4)]
    #interface=OfflineInterface([abt[0],g[1],abt[2],g[3]],print_flag=True)
    interface=OfflineInterface([g[0],abt[1],g[2],abt[3]],print_flag=True)
    N1=4;N2=2

    hands=read_std_hands(handsfile)

    tik=time.time()
    stats=[]
    for k,hand in hands[:N1]:
        stats.append(play_a_test(interface,hand,N2,step_int=False))
        print("%4d"%(stats[-1],),end=" ",flush=True)
        if (k+1)%(N1//4)==0:
            bench_stat(stats)
    tok=time.time()
    log("time consume: %ds"%(tok-tik))

    bench_stat(stats)

def bench_stat(stats,comments=None):
    print("")
    log("benchmark result: %.2f %.2f"%(numpy.mean(stats),numpy.sqrt(numpy.var(stats)/(len(stats)-1))))
    suc_ct=len([1 for i in stats if i>0])
    draw_ct=len([1 for i in stats if i==0])
    log("success rate: (%d+%d)/%d"%(suc_ct,draw_ct,len(stats)))
    if comments!=None:
        log(comments)

if __name__=="__main__":
    benchmark("StdHands/random_0_1024.hands")