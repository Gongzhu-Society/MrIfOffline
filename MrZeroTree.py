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
import copy,math,time,random,numpy

print_level=0
BETA_POST_RECT=0.015
log("BETA_POST_RECT: %.3f, BETA: %.2f"%(BETA_POST_RECT,BETA))

class MrZeroTree(MrZeroTreeSimple):
    def __init__(self,room=0,place=0,name="default",pv_net=None,device=None,train_mode=False,
                 sample_b=-3,sample_k=-3,mcts_b=50,mcts_k=2):
        MrRandom.__init__(self,room,place,name)

        if device==None:
            devnum=torch.cuda.device_count()
            self.device=torch.device("cuda:%d"%(random.randint(0,devnum-1)))
            #self.device=torch.device("cuda:3")
        else:
            self.device=device
        if pv_net==None:
            #net_para_loc="/home/spinor/youran/MrIfOffline/Zero-29th-25-11416629-720.pt"
            net_para_loc="./Zero-29th-25-11416629-720.pt"
            from MrZ_NETs import PV_NET_2
            self.pv_net=PV_NET_2()
            self.pv_net.load_state_dict(torch.load(net_para_loc,map_location=self.device))
            self.pv_net.to(self.device)
        else:
            self.pv_net=pv_net

        self.sample_b=sample_b
        self.sample_k=sample_k
        self.mcts_b=mcts_b
        self.mcts_k=mcts_k
        self.train_mode=train_mode
        if self.train_mode:
            self.train_datas=[]
        #self.int_method_printed_flag=False

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

    def wasserstein(l):
        return (l[0]-l[1]).abs().sum()+(l[1]-l[2]).abs().sum()+(l[0]-l[2]).abs().sum()
    
    def select_interact_cards(self,legal_choice,level=2):
        input("useless")
        oh_score=MrZeroTreeSimple.score_lists_oh(self.scores,self.place)
        oh_table=MrZeroTreeSimple.four_cards_oh(self.cards_on_table,self.place)
        cards_remain=ScenarioGen.gen_cards_remain(self.history,self.cards_on_table,self.cards_list)
        #cards_remain.sort()
        oh_card=torch.zeros(52*4)
        for c in self.cards_list:
            oh_card[ORDER_DICT[c]]=1
        for c in cards_remain:
            oh_card[52*1+ORDER_DICT[c]]=1/3
            oh_card[52*2+ORDER_DICT[c]]=1/3
            oh_card[52*3+ORDER_DICT[c]]=1/3
        l_re=[]
        for c in cards_remain:
            p_temp=[]
            for i in range(1,4):
                oh_card_cp=oh_card.clone()
                for j in range(1,4):
                    if j==i:
                        oh_card_cp[52*j+ORDER_DICT[c]]=1
                    else:
                        oh_card_cp[52*j+ORDER_DICT[c]]=0
                netin=torch.cat([oh_card_cp,oh_score,oh_table])
                with torch.no_grad():
                    p,_=self.pv_net(netin.to(self.device))
                p_legal=torch.tensor([p[ORDER_DICT[c]] for c in legal_choice])
                p_legal-=p_legal.max()
                p_temp.append(p_legal)
            l_re.append((c,MrZeroTree.wasserstein(p_temp).item()))
        l_re.sort(key=lambda x:x[1],reverse=True)
        l_re=l_re[0:min(len(l_re),level)]
        return [c for c,v in l_re]
    
    def possi_rectify_pvnet(self,cards_lists,scores,cards_on_table,pnext,legal_choice,choice):
        netin=MrZeroTree.prepare_ohs_post_rect(cards_lists,cards_on_table,scores,pnext)
        with torch.no_grad():
            p,_=self.pv_net(netin.to(self.device))
        #p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice if c[0]!="C"] #G on Feb 9th
        p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice if c[0]==choice[0]] #Important!
        #p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice] #Before Jan 19th
        v_max=max((v for c,v in p_legal))

        p_line=[(c,1+BETA_POST_RECT*(v-v_max)/BETA) for c,v in p_legal]
        possi_line=max((v for c,v in p_line if c==choice).__next__(),0.1)

        """p_exp=[(c,math.exp(BETA_POST_RECT/BETA*(v-v_max))) for c,v in p_legal]
        v_sum=sum((v for c,v in p_exp))
        p_exp=[(c,v/v_sum) for c,v in p_exp]
        possi_exp=(v for c,v in p_exp if c==choice).__next__()"""

        #log(["%s: %.4f, %.4f"%(p_legal[i][0],p_line[i][1],p_exp[i][1]) for i in range(len(p_legal))])
        return possi_line

    def decide_rect_necessity(self,thisuit,choice):
        #if choice[0]=="C":
        #    return -2
        # C
        if thisuit==choice[0] and choice[1] not in "234567":
            return 3
        # D
        if thisuit=="A" and choice[1] not in "234567":
            return 4
        return -1

    def possi_rectify_old(self,cards_lists,thisuit):
        """
            posterior probability rectify
            cards_lists is in absolute order
        """
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
                #决定是否需要修正
                nece=self.decide_rect_necessity(thisuit,choice)
                if nece<0:
                    continue
                
                suit=cards_on_table[1][0] if len(cards_on_table)>1 else "A"
                cards_dict=MrGreed.gen_cards_dict(cards_lists[pnext])
                legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,cards_lists[pnext])
                possi_pvnet=self.possi_rectify_pvnet(cards_lists,scores,cards_on_table,pnext,legal_choice,choice)
                if print_level>=4:
                    log("rectify %s(%d): %.4e"%(choice,nece,possi_pvnet),end="");input()
                result*=possi_pvnet
        else:
            assert len(scores[0])==len(scores[1])==len(scores[2])==len(scores[3])==0, "scores left not zero: %s"%(scores,)
            assert len(cards_lists[0])==len(cards_lists[1])==len(cards_lists[2])==len(cards_lists[3])==13, "cards_lists not equal 4x13: %s"%(cards_lists,)
        if print_level>=3:
            log("final cards possi: %.4e"%(result))
        return result
    
    def possi_rectify(self,cards_lists,thisuit):
        pass

    def public_info(self):
        """
            collect public information for possi_rectfy, including:
                cards_played,
                scores at different stage,
                break suits at different stage
        """
        cards_played=[[],[],[],[]] #absolute order
        scores=[[],[],[],[]]
        void_info=[{'S':False,'H':False,'D':False,'C':False},{'S':False,'H':False,'D':False,'C':False},
                   {'S':False,'H':False,'D':False,'C':False},{'S':False,'H':False,'D':False,'C':False},]
        scores_stage=[]
        void_info_stage=[]
        for r_num,history in enumertae(self.history+[self.cards_on_table]):
            pnext=history[0]
            for c in history[1:]:
                cards_played[pnext].append(c)
                if pnext!=history[0] and c[0]!=history[1][0]:
                    void_info[pnext][history[1][0]]=True
                scores_stage.append(copy.deepcopy(scores))
                void_info_stage.append(copy.deepcopy(void_info))
                pnext=(pnext+1)%4
            if r_num<len(h_temp)-1:
                winner=self.history[r_num+1][0]
            elif r_num==len(h_temp)-1:
                winner=self.cards_on_table[0]
            else:
                continue
            for c in history[1:]:
                if c in SCORE_DICT:
                    scores[winner].append(c)
        num_cards_played=sum([len(i) for i in cards_played])
        assert len(scores_stage)==num_cards_played
        assert len(void_info_stage)==num_cards_played
        return cards_played,scores_stage,void_info_stage
                
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
        #imp_cards=self.select_interact_cards(legal_choice)
            
        if self.sample_k>=0:
            sce_num=self.sample_b+int(self.sample_k*len(self.cards_list))
            assert self.sample_b>=0 and sce_num>0
            sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=sce_num)
            scenarios=[i for i in sce_gen]
        else:
            assert self.sample_k<0 and self.sample_b<0
            sce_gen=ImpScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,suit,
                                   level=-1*self.sample_k,num_per_imp=-1*self.sample_b)
                                   #imp_cards=imp_cards,num_per_imp=-1*self.sample_b)
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

            scenarios_weight.append(1.0)
            #scenarios_weight.append(self.possi_rectify(cards_lists,suit))

            #scenarios_weight[-1]*=self.int_equ_class(cards_lists,suit)
            #scenarios_weight[-1]*=self.int_equ_class_li(cards_lists,suit)

            cards_lists_list.append(cards_lists)
        else:
            del scenarios
        if print_level>=2:
            log("scenarios_weight: %s"%(["%.4e"%(i) for i in scenarios_weight],))
        weight_sum=sum(scenarios_weight)
        scenarios_weight=[i/weight_sum for i in scenarios_weight]
        assert (sum(scenarios_weight)-1)<1e-6, "scenario weight is %.8f: %s"%(sum(scenarios_weight),scenarios_weight,)

        #legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)
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

        if print_level>=2:
            log("d_legal: %s"%({k:float("%.1f"%(v)) for k,v in d_legal.items()}))
            #time.sleep(5+10*random.random())

        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        """
        if len(legal_choice)>1:
            g=self.g_aux[self.place]
            g.cards_on_table=copy.copy(self.cards_on_table)
            g.history=copy.deepcopy(self.history)
            g.scores=copy.deepcopy(self.scores)
            g.cards_list=copy.deepcopy(self.cards_list)
            gc=g.pick_a_card()

            netin=MrZeroTree.prepare_ohs(cards_lists,self.cards_on_table,self.scores,self.place)
            with torch.no_grad():
                p,_=self.pv_net(netin.to(self.device))

            p_legal=[(c,p[ORDER_DICT[c]].item()) for c in legal_choice if c[0]==gc[0]]
            v_max=max((v for c,v in p_legal))
            p_legal=[(c,1+BETA_POST_RECT*(v-v_max)/BETA) for c,v in p_legal]
            p_legal.sort(key=lambda x:x[1],reverse=True)
            p_choice=(v for c,v in p_legal if c==gc).__next__()
            possi=max(p_choice,0.2)
            log("greed, %s, %s, %s, %.4f"%(gc,suit,gc==p_legal[0][0],possi),logfile="stat_sim.txt",fileonly=True)

            p_legal=[(c,p[ORDER_DICT[c]].item()) for c in legal_choice if c[0]==best_choice[0]]
            v_max=max((v for c,v in p_legal))
            p_legal=[(c,1+BETA_POST_RECT*(v-v_max)/BETA) for c,v in p_legal]
            p_legal.sort(key=lambda x:x[1],reverse=True)
            p_choice=(v for c,v in p_legal if c==best_choice).__next__()
            possi=max(p_choice,0.2)
            log("zerotree, %s, %s, %s, %.4f"%(best_choice,suit,best_choice==p_legal[0][0],possi),logfile="stat_sim.txt",fileonly=True)"""


        return best_choice

    @staticmethod
    def family_name():
        return 'MrZeroTree'

def example_DJ():
    zt3=MrZeroTree(room=255,place=3,name='zerotree3',mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-2)

    zt3.cards_list=["HQ","HJ","H8","SA","S5","S4","S3","CQ","CJ","C4"]
    zt3.cards_on_table=[1,"DJ","D8"]
    zt3.history=[[0,"H3","H5","H4","H7"],[3,"S6","SJ","HK","S10"],[0,"DQ","DA","D9","D3"]]
    zt3.scores=[["HK"],[],[],["H3","H5","H4","H7"]]
    
    """cards_dict=MrGreed.gen_cards_dict(zt3.cards_list)
    legal_choice=MrGreed.gen_legal_choice("D",cards_dict,zt3.cards_list)
    zt3.select_interact_cards(legal_choice)"""
    cards_played,scores_stage,void_info_stage=zt3.public_info()
    log(cards_played)
    #log(zt3.pick_a_card())
    return
    l=[zt3.pick_a_card() for i in range(20)]
    log("%d %d %d"%(len([i[0] for i in l if i[0]=="H"]),len([i[0] for i in l if i[0]=="C"]),len([i[0] for i in l if i[0]=="S"])))


def example_SQ():
    zt3=MrZeroTree(room=255,place=3,name='zerotree3',mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-2)

    zt3.cards_list=["HQ","HJ","H8","H7","SA","S6","S5","S4","S3","CQ","CJ","D3"]
    zt3.cards_on_table=[1,"S7","SJ"]
    zt3.history=[[1,"C9","C7","C4","H9"],]
    zt3.scores=[[],["H9"],[],[]]
    
    cards_dict=MrGreed.gen_cards_dict(zt3.cards_list)
    legal_choice=MrGreed.gen_legal_choice("S",cards_dict,zt3.cards_list)
    zt3.select_interact_cards(legal_choice)
    return
    log(zt3.pick_a_card())

def example_SQ2():
    from MrZ_NETs import PV_NET_2
    #from MrImpGreed import MrImpGreed
    device_bench=torch.device("cuda:2")
    state_dict=torch.load("Zero-29th-25-11416629-720.pt",map_location=device_bench)
    pv_net=PV_NET_2()
    pv_net.load_state_dict(state_dict)
    pv_net.to(device_bench)
    zt3=MrZeroTree(room=255,place=3,name='zerotree3',pv_net=pv_net,device=device_bench,mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-1)

    zt3.cards_list=["HQ","HJ","H8","H7","SA","S3","CJ","C4","CQ","D3","D10"]
    zt3.cards_on_table=[2,"S6"]
    zt3.history=[[0,"S7","SK","S10","S8"],[1,"C2","C9","C8","C7"]]
    zt3.scores=[[],[],[],[]]
    log(zt3.pick_a_card())

def burdens():
    from MrZ_NETs import PV_NET_2
    device_bench=torch.device("cuda:0")
    state_dict=torch.load("Zero-29th-25-11416629-720.pt",map_location=device_bench)
    pv_net=PV_NET_2()
    pv_net.load_state_dict(state_dict)
    pv_net.to(device_bench)
    zt3=MrZeroTree(room=255,place=3,name='zerotree3',pv_net=pv_net,device=device_bench,mcts_b=0,mcts_k=1,sample_b=100,sample_k=0)

    """
    {'SA': 40.89, 'SK': 40.31, 'S5': -13.70}
    {'DA': 115.56, 'DK': 126.05, 'DQ': 127.73, 'D5': 135.93}
    {'CA': -133.03, 'CK': -133.34, 'CQ': -133.53, 'CJ': -136.75, 'C5': -152.01}
    {'CQ': -93.09, 'CJ': -94.13, 'C5': -104.02}
    """

    #zt3.cards_list=["SA","SK","S5","H6","C2","C3","C4","H2","H3","H4","D2","D3","D4"]
    #zt3.cards_on_table=[0,"S2","S3","S4"]

    #zt3.cards_list=["DA","DK","DQ","D5","C2","C3","C4","H2","H3","H4","S2","S3","S4"]
    #zt3.cards_on_table=[0,"D2","D3","D4"]
    """zt3.cards_list=["D10","H10","CQ","CJ","C5","D8","D9","H5","HK","HQ","SK","SA","S9"]
    zt3.cards_on_table=[0,"C2","C3","C4"]
    zt3.history=[]
    zt3.scores=[[],[],[],[]]
    log(zt3.pick_a_card())"""

    """
    suit_tested="S"s
    cards_on_table=[0,"S10","S4","S7"]
    ('S3', '61.75'), ('SK', '72.67'), ('SA', '72.54')
    ('S3', '66.94'), ('SA', '72.08')
    ('S3', '67.26'), ('SK', '70.49')
    """
    suit_tested="S"
    cards_on_table=[0,"D3","D4","D5"]
    cards_tested=["SA","SK","S3"]

    oh_card=torch.zeros(52*4)
    for c in cards_tested:
        oh_card[ORDER_DICT[c]]=1
    for c in INIT_CARDS:
        if c in cards_tested or c in cards_on_table:
            continue
        if c.startswith(suit_tested):
            oh_card[52*1+ORDER_DICT[c]]=1/3
            oh_card[52*2+ORDER_DICT[c]]=1/3
            oh_card[52*3+ORDER_DICT[c]]=1/3
        else:
            oh_card[52*0+ORDER_DICT[c]]=1/4
            oh_card[52*1+ORDER_DICT[c]]=1/4
            oh_card[52*2+ORDER_DICT[c]]=1/4
            oh_card[52*3+ORDER_DICT[c]]=1/4
    #print(oh_card)
    print(oh_card.sum())
    oh_score=torch.zeros(16*4)
    oh_table=MrZeroTreeSimple.four_cards_oh(cards_on_table,3)
    netin=torch.cat([oh_card,oh_score,oh_table])
    with torch.no_grad():
        p,v=zt3.pv_net(netin.to(zt3.device))
    p_legal=[(c,"%.2f"%(p[ORDER_DICT[c]].item()/BETA)) for c in INIT_CARDS if (c[0]==suit_tested and c in cards_tested) or c[0]!=suit_tested]
    log("%.2f: %s"%(v,p_legal))

def irrelevant_cards():
    from MrZ_NETs import PV_NET_2
    device=torch.device("cpu")
    state_dict=torch.load("Zero-29th-25-11416629-720.pt")
    pv_net=PV_NET_2()
    pv_net.load_state_dict(state_dict)
    pv_net.to(device)

    """
    [('S5', 0.9464), ('S10', 1.0), ('SJ', 0.9598)]
    [('S5', 0.9424), ('S10', 1.0)]

    [('S8', 0.9585), ('S10', 1.0), ('SQ', 0.495), ('SJ', 0.9528)]
    [('S8', 0.956), ('S10', 1.0), ('SQ', 0.4818)]

    [('S7', 1.0), ('SK', 0.8331), ('SJ', 0.9677)]
    [('S7', 1.0), ('SK', 0.8302)]
    """

    suit_tested="S"
    cards_on_table=[0,]
    cards_tested=["S7","SK"]
    place=0

    oh_card=torch.zeros(52*4)
    for c in cards_tested:
        oh_card[ORDER_DICT[c]]=1
    for c in INIT_CARDS:
        if c in cards_tested or c in cards_on_table:
            continue
        if c.startswith(suit_tested):
            oh_card[52*1+ORDER_DICT[c]]=1/3
            oh_card[52*2+ORDER_DICT[c]]=1/3
            oh_card[52*3+ORDER_DICT[c]]=1/3
        else:
            oh_card[52*0+ORDER_DICT[c]]=1/4
            oh_card[52*1+ORDER_DICT[c]]=1/4
            oh_card[52*2+ORDER_DICT[c]]=1/4
            oh_card[52*3+ORDER_DICT[c]]=1/4
    #print(oh_card)
    print(oh_card.sum())
    oh_score=torch.zeros(16*4)
    oh_table=MrZeroTreeSimple.four_cards_oh(cards_on_table,place)
    netin=torch.cat([oh_card,oh_score,oh_table])
    with torch.no_grad():
        p,v=pv_net(netin)
    p_legal=[(c,p[ORDER_DICT[c]]) for c in cards_tested]
    v_max=max((v for c,v in p_legal))
    p_line=[(c,1+BETA_POST_RECT*(v-v_max)/BETA) for c,v in p_legal]
    log([(c,float("%.4f"%(v))) for c,v in p_line])

if __name__=="__main__":
    #example_DJ()
    example_SQ()
    #example_SQ2()
    #burdens()
    #irrelevant_cards()

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

    """
        # C4="2345": 65.7(4.8)
        # C2="23456": 67.5(4.8)
        # C ="234567": 69.4(4.5)
        # C3="2345678": 61.5(4.6)
        # C5="23456789": 65.8(4.7)

        # C6="34567": 63.37(4.65)
        # C ="234567": 69.4(4.5)

        # D2="2345": 66.5(4.5)
        # D4="23456": 72.0(4.5)
        # D ="234567": 68.0(4.6)
        # D5="2345678": 67.3(4.5)
        # D3="23456789": 61.8(4.7)

        # D8="": 63.9(4.5)
        # D7="3456": 69.24 4.62
        # D6="34567": 68.46 4.55
        # D ="234567": 68.0(4.6)

        # H=C+F+following: 68.8(4.6)
        #if thisuit=="A" and suit=="A":
        #    return True

        # N=DxH: 61.3(4.4)
        #if thisuit=="A":
        #    if suit=="A":
        #        return True
        #    elif choice[1] not in "234567":
        #        return True

        # H=if True: 64.8(4.7)
        # G=C+D+F: 59.2(4.5) 60.2(4.6) WHY?
        # K=C+F: 66.4(4.6)
        # L=C+D: 75.6(4.8)
        # L2=C+D4: 66.6(4.7)
        # M=D+F: 64.4(4.6)

        # 修正贴牌的想法不错，但是和其他修正相容性不好。
        # 但相容性不好可能是retrict_flag导致的，这还有待研究
        # F: 68.4(4.7) 67.2(4.7)
        #if suit!="A" and choice[0]!=suit:
        # F2: 65.6(4.4)
        #if thisuit!="A" and suit!="A" and choice[0]!=suit:
        # F3: 65.3(4.8)
        #if thisuit=="A" and suit!="A" and choice[0]!=suit:
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
        #        return True"""