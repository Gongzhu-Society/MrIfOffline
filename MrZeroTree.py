#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,calc_score
from Util import ORDER_DICT,ORDER_DICT2,ORDER_DICT5,SCORE_DICT,INIT_CARDS
from MrRandom import MrRandom
from MrGreed import MrGreed
from ScenarioGen import ScenarioGen
from OfflineInterface import OfflineInterface

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from MCTS.mcts import mcts
from torch.multiprocessing import Process,Queue,Lock
torch.multiprocessing.set_sharing_strategy('file_system') #fuck pytorch
import copy,itertools,numpy,time,math

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


class PV_NET(nn.Module):
    """
        return 52 policy and 1 value
    """

    """def __init__(self):
        super(PV_NET,self).__init__()
        #cards in four player(52*4), two cards on table(52*3*2), scores in four players
        self.fc0=nn.Linear(52*4+(54*3+20*4)+16*4,1024)
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
        v=self.fcv(x)*VALUE_RENORMAL
        return p,v"""

    def __init__(self):
        super(PV_NET,self).__init__()
        #cards in four player(52*4), two cards on table(52*3*2), scores in four players
        #totally 514
        self.fc0=nn.Linear(52*4+(54*3+20*4)+16*4,2048)
        self.fc1=nn.Linear(2048,2048)
        self.fc2=nn.Linear(2048,512)

        self.sc0a=nn.Linear(512,512)
        self.sc0b=nn.Linear(512,512)
        self.sc1a=nn.Linear(512,512)
        self.sc1b=nn.Linear(512,512)
        self.sc2a=nn.Linear(512,512)
        self.sc2b=nn.Linear(512,512)
        self.sc3a=nn.Linear(512,512)
        self.sc3b=nn.Linear(512,512)
        self.sc4a=nn.Linear(512,512)
        self.sc4b=nn.Linear(512,512)
        self.sc5a=nn.Linear(512,512)
        self.sc5b=nn.Linear(512,512)

        self.fcp=nn.Linear(512,52)
        self.fcv=nn.Linear(512,1)

    def forward(self, x):
        x=F.relu(self.fc0(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.sc0b(F.relu(self.sc0a(x))))+x
        x=F.relu(self.sc1b(F.relu(self.sc1a(x))))+x
        x=F.relu(self.sc2b(F.relu(self.sc2a(x))))+x
        x=F.relu(self.sc3b(F.relu(self.sc3a(x))))+x
        x=F.relu(self.sc4b(F.relu(self.sc4a(x))))+x
        x=F.relu(self.sc5b(F.relu(self.sc5a(x))))+x
        p=self.fcp(x)
        v=self.fcv(x)*VALUE_RENORMAL
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
    def __init__(self,room=0,place=0,name="default",device=None,train_mode=False,
                 sample_b=None,sample_k=None,pv_net=None,mcts_b=None,mcts_k=None,pv_deep=0):
        MrRandom.__init__(self,room,place,name)
        self.pv_net=pv_net
        self.device=device
        self.sample_b=sample_b
        self.sample_k=sample_k
        self.mcts_b=mcts_b
        self.mcts_k=mcts_k
        self.pv_deep=pv_deep
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
        """oh=torch.zeros(52*3+26*4)#,dtype=torch.uint8)
        for i,c in enumerate(cards_on_table[:0:-1]):
            oh[52*i+ORDER_DICT[c]]=1
        oh[52*3+26*(len(cards_on_table)-1):52*3+26*len(cards_on_table)]=1"""
        """oh=torch.zeros(52*3)
        for i,c in enumerate(cards_on_table[:0:-1]):
            oh[52*i+ORDER_DICT[c]]=1"""
        oh=torch.zeros(54*3+20*4)#,dtype=torch.uint8)
        for i,c in enumerate(cards_on_table[:0:-1]):
            index=54*i+ORDER_DICT[c]
            oh[index-1:index+2]=1
        oh[54*3+20*len(cards_on_table)-13:54*3+20*len(cards_on_table)]=1
        return oh

    def prepare_ohs(cards_lists,cards_on_table,score_lists,place):
        """
            double the time of four_cards for it to focus
        """
        oh_card=MrZeroTree.cards_lists_oh(cards_lists,place)
        oh_score=MrZeroTree.score_lists_oh(score_lists,place)
        oh_table=MrZeroTree.four_cards_oh(cards_on_table,place)
        return torch.cat([oh_card,oh_score,oh_table])

    def pv_policy(self,state,deep):
        if state.isTerminal():
            return state.getReward()
        elif deep==0:
            netin=MrZeroTree.prepare_ohs(state.cards_lists,state.cards_on_table,state.score_lists,state.pnext)
            with torch.no_grad():
                _,v=self.pv_net(netin.to(self.device))
            return v.item()*state.getCurrentPlayer()+state.getReward()
        else:
            log("pv-deep feature has been abondoned",l=2)
            """netin=MrZeroTree.prepare_ohs(state.cards_lists,state.cards_on_table,state.score_lists,state.pnext)
            with torch.no_grad():
                p,_=self.pv_net(netin.to(self.device))
            legal_choice=state.getPossibleActions()
            p_legal=numpy.array([p[ORDER_DICT[c]].item() for c in legal_choice])
            p_legal=numpy.exp(p_legal-max(p_legal))
            p_legal/=p_legal.sum()
            action=numpy.random.choice(legal_choice,p=p_legal)"""
            action=numpy.random.choice(state.getPossibleActions())
            neostate=state.takeAction(action)
            return self.pv_policy(neostate,deep-1)

    """def minmax_notgood(action,node):
        if len(node.children)==0:
            return node.totalReward/node.numVisits
        else:
            l=[MrZeroTree.minmax(action,subnode) for action,subnode in node.children.items()]
            if node.state.getCurrentPlayer()==1:
                l.sort(reverse=True)
            else:
                l.sort()
            return l[0]"""

    def pick_a_card(self):
        #确认桌上牌的数量和自己坐的位置相符
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place
        #utility datas
        suit=self.decide_suit() #inherited from MrRandom
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        #如果别无选择
        if (not self.train_mode) and cards_dict.get(suit)!=None and len(cards_dict[suit])==1:
            choice=cards_dict[suit][0]
            if print_level>=1:
                log("I have no choice but %s"%(choice))
            return choice

        if print_level>=1:
            log("my turn: %s, %s, %s"%(self.cards_on_table,self.cards_list,self.scores))

        legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)
        d_legal={c:0 for c in legal_choice}
        sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,
                            number=self.sample_b+int(self.sample_k*len(self.cards_list)))
        for cards_list_list in sce_gen:
            #initialize gamestate
            cards_lists=[None,None,None,None]
            cards_lists[self.place]=copy.copy(self.cards_list)
            for i in range(3):
                cards_lists[(self.place+i+1)%4]=cards_list_list[i]
            gamestate=GameState(cards_lists,self.scores,self.cards_on_table,self.place)
            if print_level>=2:
                log("gened scenario: %s"%(cards_lists))

            #mcts
            if self.mcts_b>=0:
                searchnum=self.mcts_b+self.mcts_k*len(legal_choice)
                searcher=mcts(iterationLimit=searchnum,rolloutPolicy=self.pv_policy,
                              explorationConstant=MCTS_EXPL,pv_deep=0)
                searcher.search(initialState=gamestate,needNodeValue=False)
                if self.train_mode:
                    d_legal_temp={}
                    for action,node in searcher.root.children.items():
                        #node_value=MrZeroTree.minmax(action,node)
                        node_value=node.totalReward/node.numVisits
                        d_legal[action]+=node_value
                        d_legal_temp[action]=node_value
                else:
                    for action,node in searcher.root.children.items():
                        d_legal[action]+=node.totalReward/node.numVisits
                        #d_legal[action]+=MrZeroTree.minmax(action,node)
                #save data for train
                if self.train_mode:
                    value_max=max(d_legal_temp.values())
                    target_p=torch.zeros(52);legal_mask=torch.zeros(52)
                    for k,v in d_legal_temp.items():
                        target_p[ORDER_DICT[k]]=math.exp(BETA*(v-value_max))
                        legal_mask[ORDER_DICT[k]]=1
                    target_p/=target_p.sum()
                    target_v=torch.tensor(value_max-gamestate.getReward())
                    netin=MrZeroTree.prepare_ohs(cards_lists,self.cards_on_table,self.scores,self.place)
                    self.train_datas.append((netin,target_p,target_v,legal_mask))
            else:
                log("reserved",l=2)
                """netin=MrZeroTree.prepare_ohs(cards_lists,self.cards_on_table,self.scores,self.place)
                with torch.no_grad():
                    p,_=self.pv_net(netin.to(self.device))
                p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice]
                p_legal.sort(key=lambda x:x[1],reverse=True)
                d_legal[p_legal[0][0]]+=1"""

        if print_level>=1:
            log(d_legal)
        if self.train_mode:
            d_legal={k:v+numpy.random.normal(scale=self.N_SAMPLE*2) for k,v in d_legal.items()} #2*2*3=12
        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice

    def pick_a_card_complete_info(self):
        #确认桌上牌的数量和自己坐的位置相符
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place

        #initialize gamestate
        assert self.cards_list==self.cards_remain[self.place]
        gamestate=GameState(self.cards_remain,self.scores,self.cards_on_table,self.place)

        #mcts
        suit=self.decide_suit()
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        legal_choice=MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)
        searchnum=self.mcts_b+self.mcts_k*len(legal_choice)
        searcher=mcts(iterationLimit=searchnum,rolloutPolicy=self.pv_policy,
                        explorationConstant=MCTS_EXPL,pv_deep=0)
        searcher.search(initialState=gamestate,needNodeValue=False)
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
BENCH_SMP_K=1

def benchmark(save_name,epoch,device_num=3,print_process=False):
    """
        benchmark raw network against MrGreed
    """
    N1=512;N2=2;
    log("start benchmark against MrGreed for %dx%d"%(N1,N2))

    device_bench=torch.device("cuda:%d"%(device_num))
    pv_net=torch.load(save_name)
    pv_net.to(device_bench)

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
    log("benchmark at epoch %s's result: %.2f %.2f"%(epoch,numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1))))

def prepare_train_data(pv_net,device_num,data_rounds,train_b,train_k,pv_deep,data_queue):#,data_lock):
    """
        prepare train data by self-learning
    """
    try:
        device_train=torch.device("cuda:%d"%(device_num))
        pv_net.to(device_train)
        zt=[MrZeroTree(room=0,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_train,train_mode=True,mcts_b=train_b,mcts_k=train_k,pv_deep=pv_deep) for i in range(4)]
        interface=OfflineInterface([zt[0],zt[1],zt[2],zt[3]],print_flag=False)

        for k in range(data_rounds):
            cards=interface.shuffle()
            for i in range(52):
                interface.step()
            interface.clear()
            interface.prepare_new()

        datas=[]
        for i in range(4):
            datas+=zt[0].train_datas
        data_queue.put(datas,block=False)
    except Exception as e:
        print(e)
        log("",l=3)

def prepare_train_data_complete_info(pv_net,device_num,data_rounds,train_b,train_k,data_queue):
    device_train=torch.device("cuda:%d"%(device_num))
    pv_net.to(device_train)
    zt=[MrZeroTree(room=0,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_train,train_mode=True,
                   mcts_b=train_b,mcts_k=train_k) for i in range(4)]
    interface=OfflineInterface([zt[0],zt[1],zt[2],zt[3]],print_flag=False)
    for k in range(data_rounds):
        cards=interface.shuffle()
        for i in range(52):
            interface.step_complete_info()
        interface.clear()
        interface.prepare_new()

    datas=[]
    for i in range(4):
        datas+=zt[0].train_datas
    data_queue.put(datas,block=False)

print_level=0
VALUE_RENORMAL=10
BETA=0.2
MCTS_EXPL=30

def train(pv_net,device_train_nums=[0,1,2]):
    data_rounds=64
    data_timeout=40
    loss2_weight=0.03
    train_mcts_b=0
    train_mcts_k=2
    review_number=3
    age_in_epoch=3
    log("BETA: %.2f, VALUE_RENORMAL: %d, MCTS_EXPL: %d, BENCH_SMP_B: %d, BENCH_SMP_K: %d"\
        %(BETA,VALUE_RENORMAL,MCTS_EXPL,BENCH_SMP_B,BENCH_SMP_K))
    log("loss2_weight: %.2f, data_rounds: %dx%d, train_mcts_b: %d, train_mcts_k: %s, review_number: %d, age_in_epoch: %d"
        %(loss2_weight,len(device_train_nums),data_rounds,train_mcts_b,train_mcts_k,review_number,age_in_epoch))

    device_main=torch.device("cuda:0")
    pv_net=pv_net.to(device_main)
    optimizer=optim.Adam(pv_net.parameters(),lr=0.0001,betas=(0.9,0.999),eps=1e-07,weight_decay=1e-4,amsgrad=False)
    log("optimizer: %s"%(optimizer.__dict__['defaults'],))

    train_datas=[]
    p_benchmark=None
    rest_flag=False
    for epoch in range(650):
        if epoch%40==0:
            save_name='%s-%s-%s-%d.pkl'%(pv_net.__class__.__name__,pv_net.num_layers(),pv_net.num_paras(),epoch)
            torch.save(pv_net,save_name)
            if p_benchmark!=None:
                if p_benchmark.is_alive():
                    log("waiting benchmark threading to join")
                p_benchmark.join()
            p_benchmark=Process(target=benchmark,args=(save_name,epoch))
            p_benchmark.start()

        #start prepare data processes
        data_queue=Queue()
        if rest_flag:
            log("resting...");time.sleep(120);rest_flag=False
        for i in device_train_nums:
            p=Process(target=prepare_train_data_complete_info,
                      args=(copy.deepcopy(pv_net),i,data_rounds,train_mcts_b,train_mcts_k,data_queue))
            #p=Process(target=prepare_train_data,args=(copy.deepcopy(pv_net),i,data_rounds,train_b,train_k,pv_deep,data_queue))
            p.start()
        else:
            time.sleep(data_timeout//2)
        #collect data
        if epoch>=review_number:
            train_datas=train_datas[len(train_datas)//review_number:]
        for i in device_train_nums:
            try:
                queue_get=data_queue.get(block=True,timeout=data_timeout*2+30)
                train_datas+=queue_get
            except:
                log("get data failed AGAIN at epoch %d! Has got %d datas."%(epoch,len(train_datas)),l=2)
                rest_flag=True

        train_datas_gpu=[[i[0].to(device_main),i[1].to(device_main),i[2].to(device_main),i[3].to(device_main)] for i in train_datas]
        trainloader=torch.utils.data.DataLoader(train_datas_gpu,batch_size=128,drop_last=True,shuffle=True)

        
        if (epoch<=5) or (epoch<40 and epoch%5==0) or epoch%20==0:
            output_flag=True
        else:
            output_flag=False
        for age in range(age_in_epoch):
            running_loss1=[];running_loss2=[]
            for batch in trainloader:
                p,v=pv_net(batch[0])
                log_p=F.log_softmax(p*batch[3],dim=1)
                loss1=F.kl_div(log_p,batch[1],reduction="batchmean")
                loss2=F.mse_loss(v.view(-1),batch[2],reduction='mean').sqrt()
                optimizer.zero_grad()
                loss=loss1+loss2*loss2_weight
                loss.backward()
                optimizer.step()
                running_loss1.append(loss1.item())
                running_loss2.append(loss2.item())
            batchnum=len(running_loss1)
            running_loss1=numpy.mean(running_loss1)
            running_loss2=numpy.mean(running_loss2)
            
            if age==0:
                if epoch==0:
                    test_loss1=running_loss1
                    test_loss2=running_loss2
                elif epoch<review_number:
                    test_loss1=running_loss1*(epoch+1)-last_loss1*epoch
                    test_loss2=running_loss2*(epoch+1)-last_loss2*epoch
                else:
                    test_loss1=running_loss1*3-last_loss1*2
                    test_loss2=running_loss2*3-last_loss2*2
                if output_flag:
                    log("%d: %.3f %.2f %d %d"%(epoch,test_loss1,test_loss2,len(train_datas),batchnum))
            elif age==age_in_epoch-1:
                last_loss1=running_loss1
                last_loss2=running_loss2    
            
            if output_flag:
                log("        epoch %d age %d: %.3f %.2f"%(epoch,age,running_loss1,running_loss2))

    log(p_benchmark)
    log("waiting benchmark threading to join: %s"%(p_benchmark.is_alive()))
    p_benchmark.join()
    log("benchmark threading should have joined: %s"%(p_benchmark.is_alive()))

def main():
    #pv_net=PV_NET();log("init pv_net: %s"%(pv_net))
    start_from="./ZeroNets/from-zero-9a/PV_NET-17-9479221-560.pkl"
    pv_net=torch.load(start_from);log("start from: %s"%(start_from))
    try:
        train(pv_net)
    except:
        log("",l=3)

def manually_test(save_name):
    device_cpu=torch.device("cpu")
    pv_net=torch.load(save_name)
    pv_net.to(device_cpu)

    zt=MrZeroTree(room=255,place=3,name='zerotree3',pv_net=pv_net,device=device_cpu,train_mode=True)
    g=[MrGreed(room=255,place=i,name='greed%d'%(i)) for i in [0,1,2]]
    interface=OfflineInterface([g[0],g[1],g[2],zt],print_flag=True)

    interface.shuffle()
    for i,j in itertools.product(range(13),range(4)):
        interface.step()
        input()
    log(interface.clear())
    interface.prepare_new()


if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    #print(torch.multiprocessing.get_all_sharing_strategies())
    #log("sharing_strategy: %s"%(torch.multiprocessing.get_sharing_strategy()))

    main()
    #manually_test("./ZeroNets/start-from-one-2nd/PV_NET-11-2247733-80.pkl")
    #MrGreedData.prepare_train_data_greed(None)