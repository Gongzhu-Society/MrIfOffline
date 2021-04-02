#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
from MrRandom import MrRandom # Father class of all AIs
from MrGreed import MrGreed # For its Util funcs
from MrZeroTreeSimple import GameState
from ScenarioGenerator.ScenarioGen import ScenarioGen
from MCTS.mcts import abpruning,mcts
import time,copy,numpy

print_level=0

class MrAbTree(MrRandom):

    N_SAMPLE=5

    def __init__(self,room=0,place=0,name="MrAbTree",trick_deep=2):
        MrRandom.__init__(self,room,place,name)
        self.trick_deep=trick_deep

    def ab_policy(self,state):
        assert len(state.cards_on_table)==1
        #log("in abpolicy, cards_on_table: %s"%(state.cards_on_table));input()

        if state.isTerminal():
            return state.getReward_final()
        else:
            return state.getReward_midway()

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
        # Give some output
        if print_level>=1:
            log("my turn: %s, %s, %s"%(self.cards_on_table,self.cards_list,self.scores))

        #生成Scenario
        sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=MrAbTree.N_SAMPLE)
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
        tree_deep=self.trick_deep*4-len(self.cards_on_table)+1
        searcher=abpruning(deep=tree_deep,rolloutPolicy=self.ab_policy)
        if print_level>=2:
            log("tree_deep: %d"%(tree_deep))
        for i,cards_lists in enumerate(cards_lists_list):
            #initialize gamestate
            gamestate=GameState(cards_lists,self.scores,self.cards_on_table,self.history,self.place)
            searcher.search(initialState=gamestate)
            for action,val in searcher.children.items():
                d_legal[action]+=val
        if print_level>=2:
            log("searched %.1f cases"%(searcher.counter/MrAbTree.N_SAMPLE))

        #best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        best_choice=max(d_legal.items(),key=lambda x:x[1])[0]
        if print_level>=2:
            log("%s: %s"%(best_choice,d_legal),end="");input()
        return best_choice

def benchmark(handsfile):
    from OfflineInterface import OfflineInterface,read_std_hands,play_a_test

    g=[MrGreed(room=0,place=i,name='g%d'%(i)) for i in range(4)]
    abt=[MrAbTree(room=0,place=i,name='abt%d'%(i),trick_deep=2) for i in range(4)]
    interface=OfflineInterface([abt[0],g[1],abt[2],g[3]],print_flag=False)
    N1=8;N2=2
    log("trick_deep: %d"%(interface.players[0].trick_deep,))

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