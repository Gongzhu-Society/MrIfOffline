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
        assert sum([len(i) for i in self.score_lists])==16
        scores=[calc_score(self.score_lists[(self.play_for+i)%4]) for i in range(4)]
        return scores[0]+scores[2]-scores[1]-scores[3]

class PV_NET(nn.Module):
    """
        return 52 policy and 1 value
    """
    def __init__(self):
        super(PV_NET,self).__init__()
        #cards in four player(52*4), two cards on table(52*3*2), scores in four players
        #totally 688
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

    def __init__(self,room=0,place=0,name="default",pv_net=None,device=None,N_SAMPLE=5,mcts_searchnum=200,train_mode=False):
        MrRandom.__init__(self,room,place,name)
        self.pv_net=pv_net
        self.train_mode=train_mode
        self.N_SAMPLE=N_SAMPLE
        self.mcts_searchnum=mcts_searchnum
        self.device=device
        if self.train_mode:
            assert self.mcts_searchnum>0
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
            return v.item()*state.getCurrentPlayer()

    def pick_a_card(self):
        #确认桌上牌的数量和自己坐的位置相符
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place
        #utility datas
        suit=self.decide_suit() #inherited from MrRandom
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        #如果别无选择
        if not self.train_mode and cards_dict.get(suit)!=None and len(cards_dict[suit])==1:
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
            if self.mcts_searchnum>0:
                searcher=mcts(iterationLimit=self.mcts_searchnum,rolloutPolicy=self.pv_policy,explorationConstant=200)
                searcher.search(initialState=gamestate,needNodeValue=False)
                for action,node in searcher.root.children.items():
                    d_legal[action]+=node.totalReward/node.numVisits
                #save data for train
                if self.train_mode:
                    d_legal_temp={action:node.totalReward/node.numVisits for action,node in searcher.root.children.items()}
                    value_max=max(d_legal_temp.values())
                    target_p=torch.zeros(52);legal_mask=torch.zeros(52)
                    for k,v in d_legal_temp.items():
                        target_p[ORDER_DICT[k]]=math.exp(BETA*(v-value_max))
                        legal_mask[ORDER_DICT[k]]=1
                    target_p/=target_p.sum()
                    target_v=torch.tensor(value_max)
                    netin=MrZeroTree.prepare_ohs(cards_lists,self.cards_on_table,self.scores,self.place)
                    self.train_datas.append([netin,target_p,target_v,legal_mask])

                    """
                    log(d_legal_temp)
                    log("target: %s, %s"%(["%.4f"%(i) for i in target_p],target_v))
                    with torch.no_grad():
                        p,v=self.pv_net(netin.to(self.device))
                        log("raw  p: %s"%(["%.4f"%(i) for i in p]))
                        p=F.softmax(p*legal_mask,dim=0)
                    log("output: %s, %s"%(["%.4f"%(i) for i in p],v))
                    input()
                    """
                    """
                    if len(self.cards_list)<8:
                        log("d_legal_temp: %s"%(d_legal_temp),l=0)
                        log("get target_p: %s"%(target_p),l=0)
                        log("get target_v: %s"%(target_v),l=0)
                        input()
                        log("place: %d"%(self.place))
                        #log("cards_lists: %s\n%s\n%s\n%s\n%s"%(cards_lists,netin[0:52],netin[52:104],netin[104:156],netin[156:208]),l=0)
                        log("scores: %s\n%s\n%s\n%s\n%s"%(self.scores,netin[208:224],netin[224:240],netin[240:256],netin[256:272]),l=0)
                        log("cards_on_table: %s\n%s\n%s\n%s\n%s"%(self.cards_on_table,netin[272:324],netin[324:376],netin[376:428],netin[428:480]),l=0)
                        input()
                    """
            elif self.mcts_searchnum==-1:
                searcher=mcts(iterationLimit=len(legal_choice),rolloutPolicy=self.pv_policy,explorationConstant=200)
                searcher.search(initialState=gamestate,needNodeValue=False)
                for action,node in searcher.root.children.items():
                    d_legal[action]+=node.totalReward/node.numVisits
                if print_level>=1:
                    log(d_legal)
            elif self.mcts_searchnum==-2:
                netin=MrZeroTree.prepare_ohs(cards_lists,self.cards_on_table,self.scores,self.place)
                with torch.no_grad():
                    p,_=self.pv_net(netin.to(self.device))
                p_legal=[(c,p[ORDER_DICT[c]]) for c in legal_choice]
                p_legal.sort(key=lambda x:x[1],reverse=True)
                d_legal[p_legal[0][0]]+=1

        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice

def benchmark(save_name,epoch,device_num=3,print_process=False):
    """benchmark raw network against MrGreed"""
    N1=512;N2=2;
    log("start benchmark against MrGreed for %dx%d"%(N1,N2))

    device_bench=torch.device("cuda:%d"%(device_num))
    pv_net=torch.load(save_name)
    pv_net.to(device_bench)

    zt=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_bench,mcts_searchnum=BENCHMARK_METHOD) for i in [0,2]]
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


class MrGreedData(MrGreed):
    def __init__(self,room=0,place=0,name="default"):
        MrGreed.__init__(self,room,place,name)
        self.train_datas=[]

    def pick_a_card(self):
        best_choice,d_legal_temp=MrGreed.pick_a_card(self,need_details=True)
        #log("%s %s"%(best_choice,d_legal_temp))
        if d_legal_temp!=None:
            value_max=max(d_legal_temp.values())
            target_p=torch.zeros(52);legal_mask=torch.zeros(52)
            for k,v in d_legal_temp.items():
                target_p[ORDER_DICT[k]]=math.exp(BETA*(v-value_max))
                legal_mask[ORDER_DICT[k]]=1
            target_p/=target_p.sum()
            target_v=torch.tensor(value_max)
            #prepare netin
            sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=1,METHOD1_PREFERENCE=100)
            cards_list_list=sce_gen.__iter__().__next__()
            cards_lists=[None,None,None,None]
            cards_lists[self.place]=copy.copy(self.cards_list)
            for i in range(3):
                cards_lists[(self.place+i+1)%4]=cards_list_list[i]
            netin=MrZeroTree.prepare_ohs(cards_lists,self.cards_on_table,self.scores,self.place)
            #append
            self.train_datas.append([netin,target_p,target_v,legal_mask])
        return best_choice

    def prepare_train_data_greed(data_queue):
        """
            prepare train data using MrGreed
        """
        gd=[MrGreedData(room=254,place=i,name='greedata%d'%(i)) for i in range(4)]
        interface=OfflineInterface([gd[0],gd[1],gd[2],gd[3]],print_flag=False)

        N1=10
        for k in range(N1):
            cards=interface.shuffle()
            for i in range(52):
                interface.step()
            interface.clear()
            interface.prepare_new()

        datas=[]
        for i in range(4):
            datas+=gd[0].train_datas
        data_queue.put(datas,block=False)

        while not data_queue.empty():
            time.sleep(2)
        time.sleep(4)

def prepare_train_data(pv_net,device_num,data_queue):#,data_lock):
    """
        prepare train data by self-learning
    """
    device_train=torch.device("cuda:%d"%(device_num))
    pv_net.to(device_train)
    zt=[MrZeroTree(room=0,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_train,train_mode=True) for i in range(4)]
    interface=OfflineInterface([zt[0],zt[1],zt[2],zt[3]],print_flag=False)

    N1=2;
    for k in range(N1):
        cards=interface.shuffle()
        for i in range(52):
            interface.step()
        interface.clear()
        interface.prepare_new()

    datas=[]
    for i in range(4):
        datas+=zt[0].train_datas
    data_queue.put(datas,block=False)

    #data_lock.acquire()
    #data_lock.release()
    while not data_queue.empty():
        time.sleep(2)
    time.sleep(4)


print_level=0

LOSS2_WEIGHT=0.05
BETA=0.2
VALUE_RENORMAL=100
BENCHMARK_METHOD=-1

def train(pv_net,device_train_nums=[0,1,2]):
    device_main=torch.device("cuda:0")
    pv_net=pv_net.to(device_main)
    #optimizer=optim.SGD(pv_net.parameters(),lr=0.05,momentum=0.8)
    #log("optimizer: %f %f"%(optimizer.__dict__['defaults']['lr'],optimizer.__dict__['defaults']['momentum']))
    optimizer=optim.Adam(pv_net.parameters(),lr=0.001,betas=(0.9,0.999),eps=1e-07,weight_decay=1e-4,amsgrad=False) #change beta from 0.999 to 0.99
    log("optimizer: %s"%(optimizer.__dict__['defaults'],))
    log("LOSS2_WEIGHT: %.4f, BETA: %.2f, VALUE_RENORMAL: %d, BENCHMARK_METHOD: %d"%(LOSS2_WEIGHT,BETA,VALUE_RENORMAL,BENCHMARK_METHOD))

    train_datas=[];p_benchmark=None
    for epoch in range(1000):
        output_flag=False
        if epoch%20==0:# and epoch!=0:
            save_name='%s-%s-%s-%d.pkl'%(pv_net.__class__.__name__,pv_net.num_layers(),pv_net.num_paras(),epoch)
            torch.save(pv_net,save_name)
            if p_benchmark!=None:
                if p_benchmark.is_alive():
                    log("waiting benchmark threading to join")
                p_benchmark.join()
            #benchmark(save_name,epoch)
            p_benchmark=Process(target=benchmark,args=(save_name,epoch))
            p_benchmark.start()

        data_queue=Queue()
        #data_lock=Lock();data_lock.acquire()
        for i in device_train_nums:
            p=Process(target=prepare_train_data,args=(copy.deepcopy(pv_net),i,data_queue))
            #p=Process(target=MrGreedData.prepare_train_data_greed,args=(data_queue,))
            p.start()

        train_datas=train_datas[len(train_datas)//2:len(train_datas)]
        for i in device_train_nums:
            try:
                queue_get=data_queue.get(block=True,timeout=300)
                train_datas+=[[i[0].to(device_main),i[1].to(device_main),i[2].to(device_main),i[3].to(device_main)] for i in queue_get]
            except:
                log("get data failed, has got %d datas"%(len(train_datas)),l=3)
        trainloader=torch.utils.data.DataLoader(train_datas,batch_size=len(train_datas))
        batch=trainloader.__iter__().__next__()
        assert len(batch[0])==len(train_datas)

        if (epoch<5) or (epoch<40 and epoch%5==0) or epoch%20==0:
            if epoch==0:
                log("#epoch: loss1 loss2 grad1/grad2 amp_probe #train_datas")
            output_flag=True

            p,v=pv_net(batch[0])

            optimizer.zero_grad()
            log_p=F.log_softmax(p*batch[3],dim=1)
            loss1_t=F.kl_div(log_p,batch[1],reduction="batchmean")
            loss1_t.backward(retain_graph=True)
            grad1=pv_net.fc0.bias.grad.abs().mean().item()

            optimizer.zero_grad()
            loss2_t=F.mse_loss(v.view(-1),batch[2],reduction='mean').sqrt()
            loss2_t.backward(retain_graph=True)
            grad2=pv_net.fc0.bias.grad.abs().mean().item()

            amp_probe=pv_net.fc0.bias.abs().mean().item()
            log("%d: %.2f %.2f %.4f %.4f %d"%(epoch,loss1_t.item(),loss2_t.item(),grad1/grad2,amp_probe,len(train_datas)))
            #log("\n%s\n%s\n%s"%(["%6.3f"%(i) for i in batch[1][0,:]],["%6.3f"%(i) for i in p[0,:]],["%6.3f"%(i) for i in log_p[0,:].exp()]))
            #input()

        for age in range(9+1):
            p,v=pv_net(batch[0])
            log_p=F.log_softmax(p*batch[3],dim=1)
            loss1=F.kl_div(log_p,batch[1],reduction="batchmean")
            loss2=F.mse_loss(v.view(-1),batch[2],reduction='mean').sqrt()

            optimizer.zero_grad()
            loss=loss1+loss2*LOSS2_WEIGHT
            loss.backward()
            optimizer.step()

            if output_flag and age%3==0:
                log("        epoch %d age %d: %.2f %.2f"%(epoch,age,loss1,loss2))

def main():
    #pv_net=PV_NET()
    #log("init pv_net: %s"%(pv_net))
    start_from="./ZeroNets/mimic-Greed-1st/PV_NET-11-2425909-300.pkl"
    pv_net=torch.load(start_from)
    log("start from: %s"%(start_from))
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
    #print(torch.multiprocessing.get_sharing_strategy())
    #torch.multiprocessing.set_sharing_strategy('file_system')

    main()
    #benchmark("./ZeroNets/mimic-Greed-1st/PV_NET-11-2425909-300.pkl","manual",print_process=True)
    #manually_test("./ZeroNets/mimic-Greed-1st/PV_NET-11-2425909-300.pkl")
