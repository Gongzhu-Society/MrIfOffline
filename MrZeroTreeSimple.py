#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,calc_score, calc_score_midway
from Util import ORDER_DICT,ORDER_DICT2,ORDER_DICT5,SCORE_DICT,INIT_CARDS
from MrRandom import MrRandom
from MrGreed import MrGreed
from ScenarioGenerator.ScenarioGen import ScenarioGen
#from MCTS.mcts import mcts #abort mcts
from MCTS.mcts import abpruning, mcts, ismcts
from OfflineInterface import OfflineInterface

import torch
import torch.nn.functional as F
import copy,math

class GameState():
    def __init__(self,cards_lists,score_lists,cards_on_table,history,play_for,mode=0):
        self.cards_lists=cards_lists
        self.cards_on_table=cards_on_table
        self.history = history
        self.score_lists=score_lists
        self.play_for=play_for
        self.mode = mode # 0 for original version, 1 for information set monte carlo tree search
        #decide cards_dicts, suit and pnext
        self.cards_dicts=[MrGreed.gen_cards_dict(i) for i in self.cards_lists]
        #print(cards_lists)
        #print(self.cards_dicts)
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
        if self.mode < 1:
            neo_state.cards_lists[neo_state.pnext].remove(action)
            neo_state.cards_dicts[neo_state.pnext][action[0]].remove(action)
        elif self.mode == 1:
            sce_gen = ScenarioGen(self.pnext, self.history, self.cards_on_table, self.cards_lists[self.play_for], number=1)
            cards_lists_list = []
            for cll in sce_gen:
                cards_lists = [None, None, None, None]
                cards_lists[self.pnext] = copy.copy(self.cards_lists[self.pnext])
                for i in range(3):
                    cards_lists[(self.pnext + i + 1) % 4] = cll[i]
                cards_lists_list.append(cards_lists)
            #print(cards_lists_list)
            neo_state.cards_lists = cards_lists_list[0]
            neo_state.cards_dicts = [MrGreed.gen_cards_dict(i) for i in neo_state.cards_lists]
            neo_state.cards_lists[neo_state.pnext].remove(action)
            neo_state.cards_dicts[neo_state.pnext][action[0]].remove(action)
        else:
            raise Exception("Error: mode {} of game state does not exist!".format(self.mode))
        neo_state.remain_card_num-=1
        neo_state.cards_on_table.append(action)

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

        if len(neo_state.history)>0 and len(neo_state.history[-1])<5:
            neo_state.history[-1].append(action)
        else:
            neo_state.history.append([self.play_for,action])
        return neo_state

    def isTerminal(self):
        if self.remain_card_num==0:
            return True
        else:
            return False

    def getReward_final(self):
        assert sum([len(i) for i in self.score_lists])==16
        scores=[calc_score(self.score_lists[(self.play_for+i)%4]) for i in range(4)]
        return scores[0]+scores[2]-scores[1]-scores[3]

    def getReward_midway(self, mode=1):
        if mode == 0:
            scores=[calc_score(self.score_lists[(self.play_for+i)%4]) for i in range(4)]
            return scores[0]+scores[2]-scores[1]-scores[3]
        log("warning: not using!",l=2)
        scards_played=sum([len(i) for i in self.score_lists])
        scores=[calc_score_midway(self.score_lists[(self.play_for+i)%4],scards_played) for i in range(4)]
        return scores[0]+scores[2]-scores[1]-scores[3]

    def resample(self):
        sce_gen = ScenarioGen(self.pnext, self.history, self.cards_on_table, self.cards_lists[self.play_for], number=1)
        cards_lists_list = []
        for cll in sce_gen:
            cards_lists = [None, None, None, None]
            cards_lists[self.pnext] = copy.copy(self.cards_lists[self.pnext])
            for i in range(3):
                cards_lists[(self.pnext + i + 1) % 4] = cll[i]
            cards_lists_list.append(cards_lists)
        # print(cards_lists_list)
        self.cards_lists = cards_lists_list[0]
        self.cards_dicts = [MrGreed.gen_cards_dict(i) for i in self.cards_lists]

    def renew_hidden_information(self, hidden_info):
        self.cards_lists = copy.deepcopy(hidden_info)
        self.cards_dicts = [MrGreed.gen_cards_dict(i) for i in self.cards_lists]

    def next_hidden_information(self, action):
        cl = copy.deepcopy(self.cards_lists)
        cl[self.pnext].remove(action)
        return self.cards_lists

print_level=0
BETA=0.2 #for pareparing train data
MCTS_EXPL=30

class MrZeroTreeSimple(MrRandom):
    def __init__(self,room=0,place=0,name="default",pv_net=None,device=None,train_mode=False,
                 sample_b=10,sample_k=1,mcts_b=20,mcts_k=2,tree_deep=3,searcher='mcts',args={}):
        MrRandom.__init__(self,room,place,name)
        if isinstance(device,str):
            self.device=torch.device(device)
        else:
            self.device=device

        if isinstance(pv_net,str):
            self.load_pv_net(net_para_loc=pv_net,args=args)
        else:
            self.pv_net=pv_net

        self.sample_b=sample_b
        self.sample_k=sample_k
        self.mcts_b=mcts_b
        self.mcts_k=mcts_k
        self.tree_deep=tree_deep
        self.train_mode=train_mode
        self.args=args
        self.searcher = args['searcher']
        if self.train_mode:
            self.train_datas=[]

    def load_gs_net(self, net_para_loc=None, args={}):
        from MrZ_NETs import PV_NET_2, PV_NET_3, PV_NET_4, PV_NET_5, Guessing_net_1, RES_NET_18

        if True:
            self.gs_net = Guessing_net_1()

        try:
            self.gs_net.load_state_dict(torch.load(net_para_loc, map_location=self.device))
            log("load data from %s" % (net_para_loc))
        except FileNotFoundError:
            self.gs_net.load_state_dict(torch.load("../" + net_para_loc, map_location=self.device))
            log("load data from %s" % (net_para_loc))
        self.gs_net.to(self.device)

    def load_pv_net(self,net_para_loc=None,args={}):
        from MrZ_NETs import PV_NET_2, PV_NET_3, PV_NET_4, PV_NET_5, PV_NET_TRANSFORMER_1, Guessing_net_1, RES_NET_18

        if args['pv_net'] in {'PV_NET_4'}:
            self.pv_net=PV_NET_4()#RES_NET_18()#PV_NET_2()
        elif args['pv_net'] in {'PV_NET_5'}:
            self.pv_net = PV_NET_5()
        elif args['pv_net'] in {'PV_NET_TRANSFORMER_1'}:
            self.pv_net=PV_NET_TRANSFORMER_1()
        elif net_para_loc.starts_with('PV_NET_2'):
            self.pv_net=PV_NET_2()
        else:
            self.pv_net = PV_NET_3()

        try:
            self.pv_net.load_state_dict(torch.load(net_para_loc,map_location=self.device))
            log("load data from %s"%(net_para_loc))
        except FileNotFoundError:
            self.pv_net.load_state_dict(torch.load("../"+net_para_loc,map_location=self.device))
            log("load data from %s"%(net_para_loc))
        self.pv_net.to(self.device)

    def cards_lists_oh(cards_lists,place):
        """
            return a 208-length one hot, in raletive order
            the order is [me,me+1,me+2,me+3]
        """
        oh=torch.zeros((4,52))#,dtype=torch.uint8)
        for i in range(4):
            for c in cards_lists[(place+i)%4]:
                oh[i,ORDER_DICT[c]]=1
        return oh

    def score_lists_oh(score_lists,place):
        """
            return a 64-length one hot, in relative order
            the order is [me,me+1,me+2,me+3]
        """
        oh=torch.zeros((4,52))#,dtype=torch.uint8)
        for i in range(4):
            for c in score_lists[(place+i)%4]:
                oh[i,ORDER_DICT[c]]=1
        return oh

    def four_cards_oh_perhaps_problematic(cards_on_table,place):
        """
            return a 156-legth oh, in anti-relative order
            the order is [me-1,me-2,me-3]
        """
        '''
        #assert (cards_on_table[0]+len(cards_on_table)-1)%4==place
        oh=torch.zeros((3,54)) # 这应该是54，因为大牌往往是重要的，您不能diffuse了小牌但是忽略Ace
        for i,c in enumerate(cards_on_table[:0:-1]):
            l_index = ORDER_DICT[c]
            u_index = ORDER_DICT[c]+3
            oh[i,l_index:u_index]=1

        '''
        oh=torch.zeros((3,52))
        for i,c in enumerate(cards_on_table[:0:-1]):
            oh[i,ORDER_DICT[c]]=1
        return oh


    """def four_cards_oh(cards_on_table,place):
        assert (cards_on_table[0]+len(cards_on_table)-1)%4==place
        oh=torch.zeros(54*3)
        for i,c in enumerate(cards_on_table[:0:-1]):
            index=54*i+ORDER_DICT[c]#TODO +1 !!!
            oh[index-1:index+2]=1
        return oh"""
    def four_cards_oh(cards_on_table,place):
        """
            return a 156-legth oh, in anti-relative order
            the order is [me-1,me-2,me-3]
        """
        assert (cards_on_table[0]+len(cards_on_table)-1)%4==place
        """oh=torch.zeros(52*3)
        for i,c in enumerate(cards_on_table[:0:-1]):
            oh[52*i+ORDER_DICT[c]]=1"""
        oh=torch.zeros((3,52))
        for i,c in enumerate(cards_on_table[:0:-1]):
            l_index=max(ORDER_DICT[c]-1,0)#TODO +1 !!!
            u_index = min(ORDER_DICT[c]+2,52)
            oh[i,l_index:u_index]=1
        """oh=torch.zeros(54*3+20*4)#,dtype=torch.uint8)
        for i,c in enumerate(cards_on_table[:0:-1]):
            index=54*i+ORDER_DICT[c]
            oh[index-1:index+2]=1
        oh[54*3+20*len(cards_on_table)-13:54*3+20*len(cards_on_table)]=1"""
        return oh

    def history_oh(history, place):
        oh = torch.zeros((52, 56))  # ,dtype=torch.uint8)
        ct = 0

        for h in history:
            for p in range(len(h)-1):
                oh[ct, (h[0] + p - place ) % 4] = 1
                oh[ct, 4 + ORDER_DICT[h[p+1]]] = 1
                ct+=1
        return oh

    def prepare_ohs(cards_lists,cards_on_table,score_lists,history,place):
        oh_card=MrZeroTreeSimple.cards_lists_oh(cards_lists,place)
        oh_score=MrZeroTreeSimple.score_lists_oh(score_lists,place)
        oh_table=MrZeroTreeSimple.four_cards_oh(cards_on_table,place)
        oh_history=MrZeroTreeSimple.history_oh(history,place)
        a12 = torch.cat((oh_card,oh_score,oh_table),0)
        a1 = torch.cat((torch.zeros(11,4),a12),1)
        return torch.cat((a1,oh_history),0).unsqueeze(0)
        """oh_card=MrZeroTreeSimple.cards_lists_oh(cards_lists,place)
        oh_score=MrZeroTreeSimple.score_lists_oh(score_lists,place)
        oh_table=MrZeroTreeSimple.four_cards_oh(cards_on_table,place)
        return torch.cat([oh_card,oh_score,oh_table])"""

    def pv_policy(self,state):
        if state.isTerminal():
            return state.getReward_final()
        else:
            netin=MrZeroTreeSimple.prepare_ohs(state.cards_lists,state.cards_on_table,state.score_lists,state.history,state.pnext)
            with torch.no_grad():
                #_,v=self.pv_net(netin.to(self.device))
                _,v=self.pv_net(netin.to(self.device).unsqueeze(0))

            return v[0].item()*state.getCurrentPlayer()+state.getReward_midway(mode=self.args['calc_score_mode'])

    def pick_a_card(self):
        #input("in pick a card")
        #确认桌上牌的数量和自己坐的位置相符
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place
        #utility datas
        suit=self.decide_suit() #inherited from MrRandom
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        #如果别无选择
        if cards_dict.get(suit)!=None and len(cards_dict[suit])==1:
            choice=cards_dict[suit][0]
            if print_level>=1:
                log("I have no choice but %s."%(choice))
            return choice
        if  len(self.cards_list)==1:
            if print_level>=1:
                log("There is only one card left.")
            return self.cards_list[0]
        if print_level>=1:
            log("my turn: %s, %s, %s"%(self.cards_on_table,self.cards_list,self.scores))
        #生成Scenario
        sce_num=self.sample_b+int(self.sample_k*len(self.cards_list))
        sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=sce_num)
        cards_lists_list=[]
        for cll in sce_gen:
            cards_lists=[None,None,None,None]
            cards_lists[self.place]=copy.copy(self.cards_list)
            for i in range(3):
                cards_lists[(self.place+i+1)%4]=cll[i]
            cards_lists_list.append(cards_lists)
        #MCTS并对Scenario平均
        legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)
        d_legal={c:0 for c in legal_choice}
        searchnum=self.mcts_b+self.mcts_k*len(legal_choice)
        for i,cards_lists in enumerate(cards_lists_list):
            #initialize gamestate
            gamestate=GameState(cards_lists,self.scores,self.cards_on_table,self.history,self.place,mode=0)
            # ismcts, mcts, and abprune
            if self.searcher in {'ismcts'} and self.mcts_k>=0:
                searcher = ismcts(iterationLimit=searchnum, rolloutPolicy=self.pv_policy,
                                    explorationConstant=MCTS_EXPL)
                searcher.search(initialState=gamestate)
                for action, node in searcher.root.children.items():
                    d_legal[action] += (node.totalReward / node.numVisits) / len(cards_lists_list)
            elif self.searcher in {'mcts'} and self.mcts_k>=0:
                searcher=mcts(iterationLimit=searchnum,rolloutPolicy=self.pv_policy,
                                  explorationConstant=MCTS_EXPL)
                searcher.search(initialState=gamestate)
                for action,node in searcher.root.children.items():
                    d_legal[action]+=(node.totalReward/node.numVisits)/len(cards_lists_list)
            elif self.searcher in {'mcts', 'ismcts'} and self.mcts_k==-1:
                input("not using this mode")
                netin=MrZeroTreeSimple.prepare_ohs(cards_lists,self.cards_on_table,self.scores,self.history,self.place)
                with torch.no_grad():
                    p,_=self.pv_net(netin.to(self.device).unsqueeze(0))[0]
                p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice]
                p_legal.sort(key=lambda x:x[1],reverse=True)
                d_legal[p_legal[0][0]]+=1
            elif self.searcher in {'mcts', 'ismcts'}:
                raise Exception("reserved")
                #挑选出最好的并返回
                #d_legal={k:v\ for k,v in d_legal.items()}
            else:
                searcher=abpruning(deep=self.tree_deep,n_killer=2,rolloutPolicy=self.pv_policy)
                searcher.search(initialState=gamestate)
                for action,val in searcher.children.items():
                    d_legal[action]+=val

        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice

    def pick_a_card_complete_info(self):
        #确认桌上牌的数量和自己坐的位置相符
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place

        #initialize gamestate
        #assert self.cards_list==self.cards_remain[self.place]
        gamestate=GameState(self.cards_remain,self.scores,self.cards_on_table,self.history,self.place)

        #mcts
        suit=self.decide_suit()
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)
        if self.searcher in {'mcts'}:
            searchnum=self.mcts_b+self.mcts_k*len(legal_choice)
            searcher=mcts(iterationLimit=searchnum,rolloutPolicy=self.pv_policy,
                            explorationConstant=MCTS_EXPL)
            searcher.search(initialState=gamestate)
        elif self.searcher in {'ismcts'}:
            searchnum=self.mcts_b+self.mcts_k*len(legal_choice)
            searcher=ismcts(iterationLimit=searchnum,rolloutPolicy=self.pv_policy,
                            explorationConstant=MCTS_EXPL)
            searcher.search(initialState=gamestate)
        else:
            searcher=abpruning(deep=self.tree_deep,n_killer=2,rolloutPolicy=self.pv_policy)
            searcher.search(initialState=gamestate)
        if self.searcher in {'mcts','ismcts'}:
            d_legal_temp={action: node.totalReward/node.numVisits for action,node in searcher.root.children.items()}
        else:
            d_legal_temp={action: val for action,val in searcher.children.items()}
        #save data for train
        if self.train_mode:
            value_max=max(d_legal_temp.values())
            target_p=torch.zeros(52)
            legal_mask=torch.zeros(52)
            for k,v in d_legal_temp.items():
                target_p[ORDER_DICT[k]]=math.exp(BETA*(v-value_max))
                legal_mask[ORDER_DICT[k]]=1
            target_p/=target_p.sum()
            if gamestate.isTerminal():
                target_v=torch.tensor(value_max-gamestate.getReward_final())
            else:
                target_v=torch.tensor(value_max-gamestate.getReward_midway(mode=self.args['calc_score_mode']))

            netin=MrZeroTreeSimple.prepare_ohs(self.cards_remain,self.cards_on_table,self.scores,self.history,self.place)
            self.train_datas.append((netin,target_p,target_v,legal_mask))
        best_choice=MrGreed.pick_best_from_dlegal(d_legal_temp)
        return best_choice

    @staticmethod
    def family_name():
        return 'MrZeroTreeSimple'

BENCH_SMP_B=5
BENCH_SMP_K=0

def benchmark(save_name,epoch,device_num, print_process=False,args={}):
    """
        benchmark raw network against MrGreed
        will be called by trainer
    """
    import itertools,numpy

    N1=args['benchmark_N1'];N2=2;log("start benchmark against MrGreed for %dx%d"%(N1,N2))
    if device_num < 0:
        zt=[MrZeroTreeSimple(room=255,place=i,name='zerotree%d'%(i),pv_net=save_name,device="cpu",
                   mcts_b=0,mcts_k=1,sample_b=BENCH_SMP_B,sample_k=BENCH_SMP_K,args=args) for i in [0,2]]
        log('Using CPU!')
    else:
        zt=[MrZeroTreeSimple(room=255,place=i,name='zerotree%d'%(i),pv_net=save_name,device="cuda:%d"%(device_num),
                   mcts_b=0,mcts_k=1,sample_b=BENCH_SMP_B,sample_k=BENCH_SMP_K,args=args) for i in [0,2]]
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
        # print("%4d"%(stats[-1],),end=" ",flush=True)
        interface.prepare_new()
        if print_process and l==N2-1:
            print("%4d"%(sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2),end=" ",flush=True)
    s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    s_temp=[sum(s_temp[i:i+N2])/N2 for i in range(0,len(s_temp),N2)]
    log("benchmark at epoch %s's result: %.2f %.2f"%(epoch,numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1))))

def prepare_data_queue(pv_net,device_num,data_rounds,train_b,train_k,data_queue,args={}):
    input("not using")
    device_train=torch.device("cuda:%d"%(device_num))
    pv_net.to(device_train)
    zt=[MrZeroTreeSimple(room=0,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_train,train_mode=True,
                   mcts_b=train_b,mcts_k=train_k,args=args) for i in range(4)]
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


def prepare_data(pv_net,device_num,data_rounds,train_b,train_k,args={}):
    device_train=torch.device("cuda:%d"%(device_num))
    pv_net.to(device_train)
    zt=[MrZeroTreeSimple(room=0,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_train,train_mode=True,
                   mcts_b=train_b,mcts_k=train_k,args=args) for i in range(4)]
    interface=OfflineInterface(zt,print_flag=False)
    stats=[]
    for k in range(data_rounds):
        cards=interface.shuffle()
        for i in range(52):
            interface.step_complete_info()
        stats.append(interface.clear())
        interface.prepare_new()
    return zt[0].train_datas+zt[1].train_datas+zt[2].train_datas+zt[3].train_datas

def prepare_inference_data(pv_net,device_num, data_rounds,print_process=False,args={}):
    """
        play against MrGreed
        train guesser
    """
    import itertools,numpy

    recordings = []
    if device_num < 0:
        zt=[MrZeroTreeSimple(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net,device="cpu",
                   mcts_b=0,mcts_k=1,sample_b=BENCH_SMP_B,sample_k=BENCH_SMP_K,args=args) for i in [0,2]]
        log('Using CPU!')
    else:
        zt=[MrZeroTreeSimple(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net,device="cuda:%d"%(device_num),
                   mcts_b=0,mcts_k=1,sample_b=BENCH_SMP_B,sample_k=BENCH_SMP_K,args=args) for i in [0,2]]
    g=[MrGreed(room=255,place=i,name='greed%d'%(i)) for i in [1,3]]
    interface=OfflineInterface([zt[0],g[0],zt[1],g[1]],print_flag=False,record_history=True)

    stats=[]
    for k in range(data_rounds):
        cards=interface.shuffle()
        for i in range(52):
            interface.step()
        stats.append(interface.clear())
        recordings += [s for s in interface.recording if s[0] in [0,2]]
        interface.prepare_new()
    return recordings

if __name__=="__main__":
    pass
