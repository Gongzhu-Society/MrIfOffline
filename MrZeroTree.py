#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,calc_score
from Util import ORDER_DICT,ORDER_DICT2,ORDER_DICT5,SCORE_DICT
from MrRandom import MrRandom
from MrGreed import MrGreed
from ScenarioGen import ScenarioGen
from OfflineInterface import OfflineInterface

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from MCTS.mcts import mcts
from torch.multiprocessing import Process,Queue
import copy,itertools,numpy,time,math

print_level=0

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
        assert sum([len(i) for i in self.score_lists])==16
        scores=[calc_score(self.score_lists[(self.play_for+i)%4]) for i in range(4)]
        return scores[0]+scores[2]-scores[1]-scores[3]

class PV_NET(nn.Module):
    """
        return 52 policy and 1 value
    """

    VALUE_RENORMAL=200

    def __init__(self):
        super(PV_NET,self).__init__()
        #cards in four player(52*4), two cards on table(52*3*2), scores in four players
        #totally 584
        self.fc0=nn.Linear(52*4+(52*3+13*4)*2+16*4,1024)
        self.fc1=nn.Linear(1024,1024)
        self.fc2=nn.Linear(1024,256)
        self.fc3=nn.Linear(256,256)
        self.fc4=nn.Linear(256,256)
        self.fc5=nn.Linear(256,256)
        self.fc6=nn.Linear(256,256)
        self.fc7=nn.Linear(256,256)
        self.fc8=nn.Linear(256,256)
        self.fcp=nn.Linear(256,52)
        self.fcv=nn.Linear(256,1)

    def forward(self, x):
        x=F.relu(self.fc0(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc4(F.relu(self.fc3(x))))+x
        x=F.relu(self.fc6(F.relu(self.fc5(x))))+x
        x=F.relu(self.fc8(F.relu(self.fc7(x))))+x
        p=self.fcp(x)
        v=self.fcv(x)*PV_NET.VALUE_RENORMAL
        return p,v

    def num_paras(self):
        return sum([p.numel() for p in self.parameters()])

    def num_layers(self):
        ax=0
        for name,child in self.named_children():
            ax+=1
        return ax

    def __str__(self):
        stru=[]
        for name,child in self.named_children():
            if 'weight' in child.state_dict():
                stru.append(tuple(child.state_dict()['weight'].t().size()))
                #stru.append(child.state_dict()['weight'].shape)
        return "%s %s %s"%(self.__class__.__name__,stru,self.num_paras())

class MrZeroTree(MrRandom):
    def __init__(self,room=0,place=0,name="default",pv_net=None,N_SAMPLE=5,device=None,train_mode=False,BETA=None):
        MrRandom.__init__(self,room,place,name)
        self.pv_net=pv_net
        self.train_mode=train_mode
        self.N_SAMPLE=N_SAMPLE
        self.device=device
        if self.train_mode:
            self.train_datas=[]
            self.BETA=BETA


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
        oh=torch.zeros(52*3+13*4)#,dtype=torch.uint8)
        for i,c in enumerate(cards_on_table[:0:-1]):
            oh[52*i+ORDER_DICT[c]]=1
        oh[52*3+13*len(cards_on_table)-13:52*3+13*len(cards_on_table)]=1
        return oh

    def prepare_ohs(cards_lists,cards_on_table,score_lists,place):
        """
            double the time of four_cards for it to focus
        """
        oh_card=MrZeroTree.cards_lists_oh(cards_lists,place)
        oh_score=MrZeroTree.score_lists_oh(score_lists,place)
        oh_table=MrZeroTree.four_cards_oh(cards_on_table,place)
        return torch.cat([oh_card,oh_score,oh_table,oh_table])

    def pv_policy(self,state):
        if state.isTerminal():
            return state.getReward()
        else:
            netin=MrZeroTree.prepare_ohs(state.cards_lists,state.cards_on_table,state.score_lists,state.pnext)
            with torch.no_grad():
                _,v=self.pv_net(netin.to(self.device))
            return v.item()

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

        if print_level>=1:
            log("my turn: %s, %s, %s"%(self.cards_on_table,self.cards_list,self.scores))
        legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)
        d_legal={c:0 for c in legal_choice} #dict of legal choice
        sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=self.N_SAMPLE,METHOD1_PREFERENCE=100)
        for cards_list_list in sce_gen:
            #initialize gamestate
            cards_lists=[None,None,None,None]
            cards_lists[self.place]=copy.copy(self.cards_list)
            for i in range(3):
                cards_lists[(self.place+i+1)%4]=cards_list_list[i]
            gamestate=GameState(cards_lists,self.scores,self.cards_on_table,self.place)
            if print_level>=1:
                log("gened scenario: %s"%(cards_lists))

            #mcts
            searcher=mcts(iterationLimit=200,rolloutPolicy=self.pv_policy,explorationConstant=200)
            searcher.search(initialState=gamestate,needNodeValue=False)
            d_legal_temp={}
            for action,node in searcher.root.children.items():
                d_legal[action]+=node.totalReward/node.numVisits
                d_legal_temp[action]=node.totalReward/node.numVisits
                #print(action,node)

            #save data for train
            if self.train_mode:
                target_p=torch.zeros(52)
                for k,v in d_legal_temp.items():
                    target_p[ORDER_DICT[k]]=math.exp(self.BETA*v)
                target_p/=target_p.sum()
                target_v=torch.tensor(max([v for k,v in d_legal_temp.items()]))
                #log("get target: %s, %s"%(target_p,target_v))
                netin=MrZeroTree.prepare_ohs(cards_lists,self.cards_on_table,self.scores,self.place)
                netin=netin.tolist()
                target_p=target_p.tolist()
                target_v=target_v.tolist()
                self.train_datas.append([netin,target_p,target_v])

        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice

def prepare_train_data(pv_net,device_train_num,data_queue):
    N1=2
    #log("preparing train datas using %d games"%(N1))

    #log("device num: %d"%(device_train_num))
    device_train=torch.device("cuda:%d"%(device_train_num))
    pv_net.to(device_train)
    zt=[MrZeroTree(room=0,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_train,train_mode=True,BETA=0.05) for i in range(4)]
    interface=OfflineInterface([zt[0],zt[1],zt[2],zt[3]],print_flag=False)

    stats=[]
    for k in range(N1):
        cards=interface.shuffle()
        for i,j in itertools.product(range(13),range(4)):
            interface.step()
        stats.append(interface.clear())
        interface.prepare_new()
        #print("%s"%(stats[-1]),end=" ",flush=True)
    else:
        #print("")
        pass
    #s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    #log(" 0+2 - 1+3: %.2f %.2f"%(numpy.mean(s_temp), numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1)) ))
    datas=zt[0].train_datas+zt[1].train_datas+zt[2].train_datas+zt[3].train_datas
    data_queue.put(datas,block=False)
    #log("get %d datas"%len(datas))
    time.sleep(1)
    while not data_queue.empty():
        time.sleep(1)

def benchmark(save_name,epoch,device_bench_num=3):
    N1=48;N2=2
    log("start benchmark against MrGreed for %dx%d"%(N1,N2))
    device_bench=torch.device("cuda:%d"%(device_bench_num))
    pv_net=torch.load(save_name)
    pv_net.to(device_bench)
    zt=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_bench) for i in range(4)]
    g=[MrGreed(room=255,place=i,name='greed%d'%(i)) for i in range(4)]
    interface=OfflineInterface([zt[0],g[1],zt[2],g[3]],print_flag=False)
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
        #log("benchmark: %s"%(stats[-1]))
    s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    log("benchmark at epoch %d result: %.2f %.2f"%(epoch,numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1))))

def train(pv_net,device_train_nums=[0,1,2,1,2]):
    device_trains=[torch.device("cuda:%d"%(i)) for i in device_train_nums]
    device_main=torch.device("cuda:0")
    pv_net=pv_net.to(device_main)
    optimizer=optim.SGD(pv_net.parameters(),lr=0.05,momentum=0.8)
    #optimizer=optim.Adam(pnet.parameters(),lr=initial_lr_1,betas=(0.9,0.999),eps=1e-04,weight_decay=1e-4,amsgrad=False)
    for epoch in range(7200):
        if epoch%20==0:
            save_name='%s-%s-%s-%d.pkl'%(pv_net.__class__.__name__,pv_net.num_layers(),pv_net.num_paras(),epoch)
            torch.save(pv_net,save_name)
            log("saved net to %s"%(save_name))
            if epoch>0:
                log("waiting benchmark process to join")
                p_benchmark.join()
            p_benchmark=Process(target=benchmark,args=(save_name,epoch))
            p_benchmark.start()

        data_queue=Queue()
        data_processes=[]
        for i in device_train_nums:
            data_processes.append(Process(target=prepare_train_data,args=(copy.deepcopy(pv_net),i,data_queue)))
            data_processes[-1].start()

        train_datas=[]
        for i in device_train_nums:
            try:
                train_datas+=data_queue.get(block=True,timeout=120)
            except:
                log("get data failed, has got %d datas"%(len(train_datas)),l=3)
        train_datas=[[torch.tensor(i[0],device=device_main),torch.tensor(i[1],device=device_main),torch.tensor(i[2],device=device_main)] for i in train_datas]
        trainloader=torch.utils.data.DataLoader(train_datas,batch_size=len(train_datas))
        batch=trainloader.__iter__().__next__()
        assert len(batch[0])==len(train_datas)

        optimizer.zero_grad()
        p,v=pv_net(batch[0])
        log_p=F.log_softmax(p,dim=1)
        loss1=F.kl_div(log_p,batch[1],reduction="batchmean") #normally 2.5~3 at most
        loss2=F.l1_loss(v.view(-1),batch[2],reduction='mean') #normally 40-60
        loss=loss1+loss2*0.1
        loss.backward()
        optimizer.step()

        if epoch%5==0:
            log("%3d: %f %f %d"%(epoch,loss1.item(),loss2.item(),len(train_datas)))

def spy_paras():
    pv_net1=torch.load("PV_NET-11-2319413-20.pkl")
    pv_net2=torch.load("PV_NET-11-2319413-40.pkl")
    print(pv_net1.fc0.weight)
    print(pv_net2.fc0.weight)

if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    pv_net=PV_NET()
    log("init pv_net: %s"%(pv_net))
    train(pv_net)
    #spy_paras()