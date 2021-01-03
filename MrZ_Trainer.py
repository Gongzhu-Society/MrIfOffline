#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
from MrZeroTree import clean_worker,benchmark

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.multiprocessing import Process,Queue
torch.multiprocessing.set_sharing_strategy('file_system') #fuck pytorch

import copy,itertools,numpy,gc,time

class PV_NET_FATHER(nn.Module):
    """
        return 52 policy and 1 value
    """
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

VALUE_RENORMAL=10

class PV_NET(PV_NET_FATHER):
    def __init__(self):
        super(PV_NET,self).__init__()
        self.fc0=nn.Linear(52*4+(54*3+0*4)+16*4,2048)
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
        self.sc6a=nn.Linear(512,512)
        self.sc6b=nn.Linear(512,512)
        self.sc7a=nn.Linear(512,512)
        self.sc7b=nn.Linear(512,512)
        self.sc8a=nn.Linear(512,512)
        self.sc8b=nn.Linear(512,512)
        self.sc9a=nn.Linear(512,512)
        self.sc9b=nn.Linear(512,512)

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
        x=F.relu(self.sc2b(F.relu(self.sc6a(x))))+x
        x=F.relu(self.sc3b(F.relu(self.sc7a(x))))+x
        x=F.relu(self.sc4b(F.relu(self.sc8a(x))))+x
        x=F.relu(self.sc5b(F.relu(self.sc9a(x))))+x
        p=self.fcp(x)
        v=self.fcv(x)*VALUE_RENORMAL
        return p,v

def train(pv_net,dev_train_nums=[0,],dev_bench_num=1):
    import torch.optim as optim
    import gc
    data_rounds=64 #64
    data_timeout=30 #96
    data_timerest=10 #20
    loss2_weight=0.03
    train_mcts_b=0
    train_mcts_k=2
    review_number=3
    age_in_epoch=3
    log("loss2_weight: %.2f, data_rounds: %dx%d, train_mcts_b: %d, train_mcts_k: %.1f, review_number: %d, age_in_epoch: %d"
        %(loss2_weight,len(dev_train_nums),data_rounds,train_mcts_b,train_mcts_k,review_number,age_in_epoch))

    device_main=torch.device("cuda:0")
    pv_net=pv_net.to(device_main)
    optimizer=optim.Adam(pv_net.parameters(),lr=0.0004,betas=(0.9,0.999),eps=1e-07,weight_decay=1e-4,amsgrad=False)
    log("optimizer: %s"%(optimizer.__dict__['defaults'],))

    train_datas=[]
    p_benchmark=None
    data_queue=Queue()
    for epoch in range(4000):
        if epoch%90==0:
            save_name='%s-%s-%s-%d.pkl'%(pv_net.__class__.__name__,pv_net.num_layers(),pv_net.num_paras(),epoch)
            torch.save(pv_net,save_name)
            if p_benchmark!=None:
                if p_benchmark.is_alive():
                    log("waiting benchmark threading to join")
                p_benchmark.join()
            p_benchmark=Process(target=benchmark,args=(save_name,epoch,dev_bench_num))
            p_benchmark.start()

        if (epoch<=5) or (epoch<30 and epoch%5==0) or epoch%30==0:
            output_flag=True
            log("gc len at %d: %d"%(epoch,len(gc.get_objects())))
        else:
            output_flag=False

        #start prepare data processes
        for i in dev_train_nums:
            args=(copy.deepcopy(pv_net),i,data_rounds,train_mcts_b,train_mcts_k,data_queue)
            #p=Process(target=prepare_train_data_complete_info,args=args)
            p=Process(target=clean_worker,args=args)
            p.start()
        else:
            time.sleep(data_timerest)

        #collect data
        if epoch>=review_number:
            train_datas=train_datas[len(train_datas)//review_number:]
        for i in range(len(dev_train_nums)*4):
            try:
                if i==0:
                    queue_get=data_queue.get(block=True,timeout=data_timeout*2+data_timerest)
                else:
                    queue_get=data_queue.get(block=True,timeout=data_timerest)
                train_datas+=queue_get
            except:
                log("get data failed AGAIN at epoch %d! Has got %d datas."%(epoch,len(train_datas)),l=2)

        trainloader=torch.utils.data.DataLoader(train_datas,batch_size=128,drop_last=True,shuffle=True)
        for age in range(age_in_epoch):
            running_loss1=[];running_loss2=[]
            for batch in trainloader:
                p,v=pv_net(batch[0].to(device_main))
                log_p=F.log_softmax(p*batch[3].to(device_main),dim=1)
                loss1=F.kl_div(log_p,batch[1].to(device_main),reduction="batchmean")
                loss2=F.mse_loss(v.view(-1),batch[2].to(device_main),reduction='mean').sqrt()
                optimizer.zero_grad()
                loss=loss1+loss2*loss2_weight
                loss.backward()
                optimizer.step()
                running_loss1.append(loss1.item())
                running_loss2.append(loss2.item())
            batchnum=len(running_loss1)
            running_loss1=numpy.mean(running_loss1)
            running_loss2=numpy.mean(running_loss2)

            if output_flag and age==0:
                if epoch==0:
                    test_loss1=running_loss1
                    test_loss2=running_loss2
                elif epoch<review_number:
                    test_loss1=running_loss1*(epoch+1)-last_loss1*epoch
                    test_loss2=running_loss2*(epoch+1)-last_loss2*epoch
                else:
                    test_loss1=running_loss1*3-last_loss1*2
                    test_loss2=running_loss2*3-last_loss2*2
                log("%d: %.3f %.2f %d %d"%(epoch,test_loss1,test_loss2,len(train_datas),batchnum))

            if age==age_in_epoch-1:
                last_loss1=running_loss1
                last_loss2=running_loss2

            if output_flag:
                log("        epoch %d age %d: %.3f %.2f"%(epoch,age,running_loss1,running_loss2))

    log(p_benchmark)
    log("waiting benchmark threading to join: %s"%(p_benchmark.is_alive()))
    p_benchmark.join()
    log("benchmark threading should have joined: %s"%(p_benchmark.is_alive()))
    
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
    
def main():
    from MrZeroTree import BETA,MCTS_EXPL,BENCH_SMP_B,BENCH_SMP_K
    log("BETA: %.2f, VALUE_RENORMAL: %d, MCTS_EXPL: %d, BENCH_SMP_B: %d, BENCH_SMP_K: %.1f"\
        %(BETA,VALUE_RENORMAL,MCTS_EXPL,BENCH_SMP_B,BENCH_SMP_K))
    pv_net=PV_NET();log("init pv_net: %s"%(pv_net))
    #start_from="./ZeroNets/from-zero-14d/PV_NET-17-9479221-450.pkl"
    #pv_net=torch.load(start_from);log("start from: %s"%(start_from))
    train(pv_net)


if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')

    main()
    #manually_test("./ZeroNets/start-from-one-2nd/PV_NET-11-2247733-80.pkl")