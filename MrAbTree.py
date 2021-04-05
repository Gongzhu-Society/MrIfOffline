#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,SCORE_DICT,calc_score,calc_score_midway
from MrRandom import MrRandom # Father class of all AIs
from MrGreed import MrGreed # For its Util funcs
from MrZeroTreeSimple import GameState
from ScenarioGenerator.ScenarioGen import ScenarioGen
from MCTS.mcts import abpruning,mcts
import time,copy,numpy
from multiprocessing import Process,Queue

# print_level meaning
# 0: nothing
# 1:
# 2: gross search result
# 3: into each search cases
print_level=0

class MrAbTree(MrRandom):
    BURDEN_DICT={'SA':11,'SK':9,'SQ':8,'SJ':7,'S10':6,'S9':5,'S8':4,'S7':3,'S6':2,'S5':1,'S4':1,
                 'CA':11,'CK':9,'CQ':8,'CJ':7,'C10':6,'C9':5,'C8':4,'C7':3,'C6':2,'C5':1,'C4':1,
                 'DA':11,'DK':9,'DQ':8,'DJ':7,'D10':6,'D9':5,'D8':4,'D7':3,'D6':2,'D5':1,'D4':1,
                 'H10':6,'H9':5,'H8':4,'H7':3,'H6':2,'H5':1,'H4':1}
    BURDEN_DICT_S={'SA':50,'SK':30}
    BURDEN_DICT_D={'DA':-30,'DK':-20,'DQ':-10}
    BURDEN_DICT_C={'CA':0.4,'CK':0.3,'CQ':0.2,'CJ':0.1} #ratio of burden, see calc_relief
    #SHORT_PREFE=0.1
    SHORT_PREFE=30
    SHORT_POSSI=(None,
                (0.0000,),(0.0000,),(0.2229,),(0.4439,),
                (0.6164,0.0000),(0.7381,0.0000),(0.8162,0.1245),(0.8847,0.2847),
                (0.9198,0.4495,0.0000),(0.9456,0.5764,0.0000),(0.9628,0.6959,0.0852),(0.9722,0.7730,0.2161),
                (0.9853,0.8374,0.3561,0.0000))

    def __init__(self,room=0,place=0,name="MrAbTree",sample_b=5,trick_deep=2,multi_proc=True):
        MrRandom.__init__(self,room,place,name)
        self.sample_b=sample_b
        self.trick_deep=trick_deep
        self.multi_proc=multi_proc

    def calc_burden(cards_lists,play_for,score_remain_avg,score_lists,suits_remain_tot,short_flag=False):
        cards_list=cards_lists[play_for]
        burden=sum([MrAbTree.BURDEN_DICT.get(i,0) for i in cards_list])

        cards_remain_others=set([c for j in range(1,4) for c in cards_lists[(play_for+j)%4]])

        if 'SQ' in cards_remain_others:
            burden+=sum([MrAbTree.BURDEN_DICT_S.get(i,0) for i in cards_list])
        burden_j=0
        if 'DJ' in cards_remain_others:
            #burden+=sum([MrAbTree.BURDEN_DICT_D.get(i,0) for i in cards_list])
            burden_j=sum([MrAbTree.BURDEN_DICT_D.get(i,0) for i in cards_list])
        burden_c10=0
        if 'C10' in cards_remain_others:
            burden-=sum([MrAbTree.BURDEN_DICT_C.get(i,0)*score_remain_avg for i in cards_list])
        elif 'C10' in score_lists[play_for]:
            #burden-=score_remain_avg
            burden_c10=score_remain_avg

        if short_flag:
            short_factor=0
            for s in "SHDC":
                s_num=sum([1 for c in cards_list if c[0]==s])
                s_num_oppo=(sum([1 for c in cards_lists[(play_for+1)%4] if c[0]==s])+sum([1 for c in cards_lists[(play_for+3)%4] if c[0]==s]))/2
                if s_num<suits_remain_tot[s]/4 and s_num<s_num_oppo:
                    short_factor+=MrAbTree.SHORT_POSSI[suits_remain_tot[s]][s_num]
            burden-=short_factor*MrAbTree.SHORT_PREFE*len(cards_list)/13

        return burden+burden_j-burden_c10

    def ab_policy(self,state):
        assert len(state.cards_on_table)==1
        if state.isTerminal():
            scores=[calc_score(state.score_lists[(state.play_for+i)%4]) for i in range(4)]
            return scores[0]+scores[2]-scores[1]-scores[3]
        else:
            scards_played=sum([len(i) for i in state.score_lists])
            scores=[calc_score_midway(state.score_lists[(state.play_for+i)%4],scards_played) for i in range(4)]
            scores=scores[0]+scores[2]-scores[1]-scores[3]

            cards_remain=set([i for j in range(4) for i in state.score_lists[j]])
            score_remain_avg=sum([SCORE_DICT.get(c,0) for c in cards_remain])/4
            suits_remain_tot={s:sum([1 for c in cards_remain if c[0]==s]) for s in "SHDC"}

            burdens=[MrAbTree.calc_burden(state.cards_lists,(state.play_for+i)%4,score_remain_avg,state.score_lists,suits_remain_tot,short_flag=(i==0)) for i in range(4)]
            burdens=burdens[0]+burdens[2]-burdens[1]-burdens[3]
            #burden_0=MrAbTree.calc_burden(state.cards_lists,state.play_for,score_remain,state.score_lists)
            #burden_2=MrAbTree.calc_burden(state.cards_lists,(state.play_for+2)%4,score_remain,state.score_lists)
            #burdens=burden_0+burden_2
            #burdens=burden_0

            return scores-burdens*len(state.cards_lists[state.play_for])/12

    def search_a_case(self,cards_lists,searcher,data_q):
        gamestate=GameState(cards_lists,self.scores,self.cards_on_table,self.history,self.place)
        searcher.search(initialState=gamestate)
        data_q.put(searcher.children)

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

        if self.multi_proc and len(cards_lists_list)>2 and tree_deep>=7 and len(self.cards_list)>2:
            plist=[]
            data_q=Queue()
            for i,cards_lists in enumerate(cards_lists_list):
                plist.append(Process(target=self.search_a_case,args=(cards_lists,searcher,data_q)))
                plist[-1].start()
            for p in plist:
                p.join()
            for i in range(len(plist)):
                d_case=data_q.get(True)
                for action,val in d_case.items():
                    d_legal[action]+=val
        else:
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
    #abt=[MrAbTree(room=0,place=i,name='abt%d'%(i),trick_deep=2,sample_b=5,multi_proc=True) for i in range(4)]
    #abt=[MrAbTree(room=0,place=i,name='abt%d'%(i),trick_deep=1,sample_b=1,multi_proc=False) for i in range(4)]
    abt=[MrAbTree(room=0,place=i,name='abt%d'%(i),trick_deep=1,sample_b=5,multi_proc=True) for i in range(4)]
    interface=OfflineInterface([abt[0],g[1],abt[2],g[3]],print_flag=False)
    #interface=OfflineInterface([g[0],g[1],g[2],g[3]],print_flag=False)
    N1=128;N2=2
    log(interface)

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