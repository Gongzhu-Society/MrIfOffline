#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,calc_score
from Util import ORDER_DICT,ORDER_DICT2,ORDER_DICT5,SCORE_DICT,INIT_CARDS
from MrRandom import MrRandom
from MrGreed import MrGreed
from ScenarioGenerator.ScenarioGen import ScenarioGen
from ScenarioGenerator.ImpScenarioGen import ImpScenarioGen
from OfflineInterface import OfflineInterface
from MCTS.mcts import mcts

import torch
import torch.nn.functional as F
import copy,gc,math

class GameState():
    def __init__(self,cards_lists,score_lists,cards_on_table,play_for):
        self.cards_lists=cards_lists
        self.cards_on_table=cards_on_table
        self.score_lists=score_lists
        self.play_for=play_for

        #decide cards_dicts, suit and pnext
        self.cards_dicts=[MrGreed.gen_cards_dict(i) for i in self.cards_lists]
        if len(self.cards_on_table)==1:
            self.suit="A"
        else:
            self.suit=self.cards_on_table[1][0]
        self.pnext=(self.cards_on_table[0]+len(self.cards_on_table)-1)%4
        self.remain_card_num=sum([len(i) for i in self.cards_lists])

    def getCurrentPlayer(self):
        if (self.pnext-self.play_for)%2==0:
            return 1
        else:
            return -1

    def getPossibleActions(self):
        return MrGreed.gen_legal_choice(self.suit,self.cards_dicts[self.pnext],self.cards_lists[self.pnext])

    def takeAction(self,action):
        #log(action)
        neo_state=copy.deepcopy(self)
        neo_state.cards_lists[neo_state.pnext].remove(action)
        neo_state.cards_dicts[neo_state.pnext][action[0]].remove(action)
        neo_state.remain_card_num-=1
        neo_state.cards_on_table.append(action)
        #log(neo_state.cards_on_table)
        #input()
        assert len(neo_state.cards_on_table)<=5
        if len(neo_state.cards_on_table)<5:
            neo_state.pnext=(neo_state.pnext+1)%4
            if len(neo_state.cards_on_table)==2:
                neo_state.suit=neo_state.cards_on_table[1][0]
        else:
            #decide pnext
            score_temp=-1024
            for i in range(4):
                if neo_state.cards_on_table[i+1][0]==neo_state.cards_on_table[1][0] and ORDER_DICT2[neo_state.cards_on_table[i+1][1]]>score_temp:
                    winner=i #in relative order
                    score_temp=ORDER_DICT2[neo_state.cards_on_table[i+1][1]]
            neo_state.pnext=(neo_state.cards_on_table[0]+winner)%4
            #clear scores
            neo_state.score_lists[neo_state.pnext]+=[c for c in neo_state.cards_on_table[1:] if c in SCORE_DICT]
            #clean table
            neo_state.cards_on_table=[neo_state.pnext,]
            neo_state.suit='A'
        return neo_state

    def isTerminal(self):
        if self.remain_card_num==0:
            return True
        else:
            return False

    def getReward(self):
        #assert sum([len(i) for i in self.score_lists])==16
        scores=[calc_score(self.score_lists[(self.play_for+i)%4]) for i in range(4)]
        return scores[0]+scores[2]-scores[1]-scores[3]

print_level=0
BETA=0.2
MCTS_EXPL=30

class MrZeroTree(MrRandom):
    def __init__(self,room=0,place=0,name="default",pv_net=None,device=None,train_mode=False,
                 sample_b=None,sample_k=None,mcts_b=None,mcts_k=None):
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

    def cards_lists_oh(cards_lists,place):
        """
            return a 208-length one hot, in raletive order
            the order is [me,me+1,me+2,me+3]
        """
        oh=torch.zeros(52*4)#,dtype=torch.uint8)
        for i in range(4):
            for c in cards_lists[(place+i)%4]:
                oh[52*i+ORDER_DICT[c]]=1
        return oh

    def score_lists_oh(score_lists,place):
        """
            return a 64-length one hot, in relative order
            the order is [me,me+1,me+2,me+3]
        """
        oh=torch.zeros(16*4)#,dtype=torch.uint8)
        for i in range(4):
            for c in score_lists[(place+i)%4]:
                oh[16*i+ORDER_DICT5[c]]=1
        return oh

    def four_cards_oh(cards_on_table,place):
        """
            return a 156-legth oh, in anti-relative order
            the order is [me-1,me-2,me-3]
        """
        assert (cards_on_table[0]+len(cards_on_table)-1)%4==place
        """oh=torch.zeros(52*3)
        for i,c in enumerate(cards_on_table[:0:-1]):
            oh[52*i+ORDER_DICT[c]]=1"""
        oh=torch.zeros(54*3)
        for i,c in enumerate(cards_on_table[:0:-1]):
            index=54*i+ORDER_DICT[c]#TODO +1 !!!
            oh[index-1:index+2]=1
        """oh=torch.zeros(54*3+20*4)#,dtype=torch.uint8)
        for i,c in enumerate(cards_on_table[:0:-1]):
            index=54*i+ORDER_DICT[c]
            oh[index-1:index+2]=1
        oh[54*3+20*len(cards_on_table)-13:54*3+20*len(cards_on_table)]=1"""
        return oh

    def prepare_ohs(cards_lists,cards_on_table,score_lists,place):
        """
            double the time of four_cards for it to focus
        """
        oh_card=MrZeroTree.cards_lists_oh(cards_lists,place)
        oh_score=MrZeroTree.score_lists_oh(score_lists,place)
        oh_table=MrZeroTree.four_cards_oh(cards_on_table,place)
        return torch.cat([oh_card,oh_score,oh_table])

    def pv_policy(self,state):
        if state.isTerminal():
            return state.getReward()
        else:
            netin=MrZeroTree.prepare_ohs(state.cards_lists,state.cards_on_table,state.score_lists,state.pnext)
            with torch.no_grad():
                _,v=self.pv_net(netin.to(self.device))
            return v.item()*state.getCurrentPlayer()+state.getReward()

    def possi_rectify_sub(self,cards_lists,scores,cards_on_table,pnext,suit,choice,confidence):
        cards_list=cards_lists[pnext]
        cards_dict=MrGreed.gen_cards_dict(cards_list)
        legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,cards_list)

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
        if len(legal_choice)==1:
            assert abs(p_choice-1)<1e-5, p_choice
            possi=confidence
        elif len(legal_choice)>1:
            possi=confidence*p_choice+(1-confidence)*(1-p_choice)/(len(legal_choice)-1)
        return possi

    def possi_rectify(self,cards_lists,c1=0.5,c2=0.55):
        """
            posterior probability rectify
            cards_lists is in absolute order
        """
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
                if pnext==self.place:
                    continue
                elif (pnext-self.place)%2==0:
                    confidence=c2
                else:
                    confidence=c1
                if len(cards_on_table)==1:
                    suit="A"
                else:
                    suit=cards_on_table[1][0]
                if print_level>=3:
                    log("%s\n%s\n%s %s %s %s %.2f"%(cards_lists,scores,cards_on_table,pnext,suit,choice,confidence))
                result*=self.possi_rectify_sub(cards_lists,scores,cards_on_table,pnext,suit,choice,confidence)
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

        legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)
        d_legal={c:0 for c in legal_choice}

        #sce_num=self.sample_b+int(self.sample_k*len(self.cards_list))
        #sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=sce_num)
        #scenarios=[i for i in sce_gen] #
        sce_gen=ImpScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,level=2,num_per_imp=2)
        scenarios=sce_gen.get_scenarios()
        scenarios_weight=[]
        cards_lists_list=[]
        for cll in scenarios:
            cards_lists=[None,None,None,None]
            cards_lists[self.place]=copy.copy(self.cards_list)
            for i in range(3):
                cards_lists[(self.place+i+1)%4]=cll[i]
            if print_level>=3:
                log("%s\n%s"%(self.history,cards_lists))
            scenarios_weight.append(self.possi_rectify(cards_lists))
            #scenarios_weight.append(1.0)
            if print_level>=3:
                log("weight: %.4e\n%s"%(scenarios_weight[-1],cards_lists))
            cards_lists_list.append(cards_lists)
        weight_sum=sum(scenarios_weight)
        scenarios_weight=[i/weight_sum for i in scenarios_weight]
        assert abs(sum(scenarios_weight)-1)<1e-5, scenarios_weight
        if print_level>=2:
            print(scenarios_weight)
        for i,cards_lists in enumerate(cards_lists_list):
            #initialize gamestate
            gamestate=GameState(cards_lists,self.scores,self.cards_on_table,self.place)
            #mcts
            if self.mcts_k>=0:
                searchnum=self.mcts_b+self.mcts_k*len(legal_choice)
                searcher=mcts(iterationLimit=searchnum,rolloutPolicy=self.pv_policy,
                              explorationConstant=MCTS_EXPL)
                searcher.search(initialState=gamestate)
                for action,node in searcher.root.children.items():
                    d_legal[action]+=scenarios_weight[i]*node.totalReward/node.numVisits
            elif self.mcts_k==-1:
                netin=MrZeroTree.prepare_ohs(cards_lists,self.cards_on_table,self.scores,self.place)
                with torch.no_grad():
                    p,_=self.pv_net(netin.to(self.device))
                p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice]
                p_legal.sort(key=lambda x:x[1],reverse=True)
                d_legal[p_legal[0][0]]+=1
            else:
                raise Exception("reserved")

            """netin=MrZeroTree.prepare_ohs(cards_lists,self.cards_on_table,self.scores,self.place)
            with torch.no_grad():
                p,_=self.pv_net(netin.to(self.device))
            p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice]
            v_max=max((v for c,v in p_legal))
            p_legal=[(c,math.exp(v-v_max)) for c,v in p_legal]
            v_sum=sum((v for c,v in p_legal))
            p_legal=[(c,v/v_sum) for c,v in p_legal]
            assert (sum((v for c,v in p_legal))-1)<1e-5, sum((v for c,v in p_legal))"""

            """choice=[(action,node.totalReward/node.numVisits) for action,node in searcher.root.children.items()]
            choice.sort(key=lambda x:x[1],reverse=True)
            choice=choice[0][0]
            print("%.4f"%([v for c,v in p_legal if c==choice][0]),end=", ",flush=True)"""

            """g=self.g_aux[self.place]
            g.cards_on_table=copy.copy(self.cards_on_table)
            g.history=copy.deepcopy(self.history)
            g.scores=copy.deepcopy(self.scores)
            g.cards_list=copy.deepcopy(self.cards_list)
            gc=g.pick_a_card(sce_gen=[scenarios[i]])
            print("%.4f"%([v for c,v in p_legal if c==gc][0]),end=", ",flush=True)"""

        if print_level>=2:
            log(d_legal)
        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice

    def pick_a_card_complete_info(self):
        #确认桌上牌的数量和自己坐的位置相符
        #assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place

        #initialize gamestate
        #assert self.cards_list==self.cards_remain[self.place]
        gamestate=GameState(self.cards_remain,self.scores,self.cards_on_table,self.place)

        #mcts
        suit=self.decide_suit()
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)
        searchnum=self.mcts_b+self.mcts_k*len(legal_choice)
        searcher=mcts(iterationLimit=searchnum,rolloutPolicy=self.pv_policy,
                        explorationConstant=MCTS_EXPL)
        searcher.search(initialState=gamestate)
        d_legal_temp={action: node.totalReward/node.numVisits for action,node in searcher.root.children.items()}
        #save data for train
        value_max=max(d_legal_temp.values())
        target_p=torch.zeros(52)
        legal_mask=torch.zeros(52)
        for k,v in d_legal_temp.items():
            target_p[ORDER_DICT[k]]=math.exp(BETA*(v-value_max))
            legal_mask[ORDER_DICT[k]]=1
        target_p/=target_p.sum()
        target_v=torch.tensor(value_max-gamestate.getReward())
        netin=MrZeroTree.prepare_ohs(self.cards_remain,self.cards_on_table,self.scores,self.place)
        self.train_datas.append((netin,target_p,target_v,legal_mask))
        best_choice=MrGreed.pick_best_from_dlegal(d_legal_temp)
        return best_choice

BENCH_SMP_B=5
BENCH_SMP_K=0

def benchmark(save_name,epoch,device_num,print_process=False):
    """
        benchmark raw network against MrGreed
    """
    import itertools,numpy

    N1=512;N2=2;
    log("start benchmark against MrGreed for %dx%d"%(N1,N2))

    device_bench=torch.device("cuda:%d"%(device_num))
    pv_net=torch.load(save_name,map_location=device_bench)
    #pv_net.to(device_bench)

    zt=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_bench,
                   mcts_b=0,mcts_k=1,sample_b=BENCH_SMP_B,sample_k=BENCH_SMP_K) for i in [0,2]]
    g=[MrGreed(room=255,place=i,name='greed%d'%(i)) for i in [1,3]]
    interface=OfflineInterface([zt[0],g[0],zt[1],g[1]],print_flag=False)

    stats=[]
    for k,l in itertools.product(range(N1),range(N2)):
        if l==0:
            cards=interface.shuffle()
        else:
            cards=cards[39:52]+cards[0:39]
            interface.shuffle(cards=cards)
        for i,j in itertools.product(range(13),range(4)):
            interface.step()
        stats.append(interface.clear())
        interface.prepare_new()
        if print_process and l==N2-1:
            print("%4d"%(sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2),end=" ",flush=True)
    s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    s_temp=[sum(s_temp[i:i+N2])/N2 for i in range(0,len(s_temp),N2)]
    log("benchmark at epoch %s's result: %.2f %.2f"%(epoch,numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1))))

def prepare_train_data_complete_info(pv_net,device_num,data_rounds,train_b,train_k,data_queue):
    device_train=torch.device("cuda:%d"%(device_num))
    pv_net.to(device_train)
    zt=[MrZeroTree(room=0,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_train,train_mode=True,
                   mcts_b=train_b,mcts_k=train_k) for i in range(4)]
    interface=OfflineInterface(zt,print_flag=False)
    stats=[]
    for k in range(data_rounds):
        cards=interface.shuffle()
        for i in range(52):
            interface.step_complete_info()
        stats.append(interface.clear())
        interface.prepare_new()

    for i in range(4):
        data_queue.put(zt[i].train_datas,block=False)

    del stats,interface,zt,pv_net,device_train

def clean_worker(*args,**kwargs):
    prepare_train_data_complete_info(*args,**kwargs)
    gc.collect()

def prepare_data(pv_net,device_num,data_rounds,train_b,train_k):
    device_train=torch.device("cuda:%d"%(device_num))
    pv_net.to(device_train)
    zt=[MrZeroTree(room=0,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_train,train_mode=True,
                   mcts_b=train_b,mcts_k=train_k) for i in range(4)]
    interface=OfflineInterface(zt,print_flag=False)
    stats=[]
    for k in range(data_rounds):
        cards=interface.shuffle()
        for i in range(52):
            interface.step_complete_info()
        stats.append(interface.clear())
        interface.prepare_new()

    return zt[0].train_datas+zt[1].train_datas+zt[2].train_datas+zt[3].train_datas

if __name__=="__main__":
    pass