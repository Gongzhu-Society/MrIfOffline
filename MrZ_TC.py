#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
from MrZeroTree import clean_worker,benchmark,prepare_data
#from MrZ_Trainer import PV_NET_FATHER,PV_NET
from MrZ_TB import PV_NET_B

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.multiprocessing import Process

import copy,itertools,numpy,gc,time

VALUE_RENORMAL=10

def train(pv_net,dev_train_num=2,dev_bench_num=0):
    import torch.optim as optim
    import gc
    data_rounds=64
    data_timeout=30
    data_timerest=10
    loss2_weight=0.03
    train_mcts_b=4
    train_mcts_k=4
    review_number=3
    age_in_epoch=3
    log("loss2_weight: %.2f, data_rounds: %d, train_mcts_b: %d, train_mcts_k: %.1f, review_number: %d, age_in_epoch: %d"
        %(loss2_weight,data_rounds,train_mcts_b,train_mcts_k,review_number,age_in_epoch))

    device_main=torch.device("cuda:%d"%(dev_train_num))
    pv_net=pv_net.to(device_main)
    optimizer=optim.Adam(pv_net.parameters(),lr=0.0001,betas=(0.3,0.999),eps=1e-07,weight_decay=1e-4,amsgrad=False)
    log("optimizer: %s"%(optimizer.__dict__['defaults'],))

    train_datas=[]
    p_benchmark=None
    for epoch in range(4000):
        if epoch%80==0:
            save_name='%s-C-%s-%s-%d.pkl'%(pv_net.__class__.__name__,pv_net.num_layers(),pv_net.num_paras(),epoch)
            torch.save(pv_net,save_name)
            if p_benchmark!=None:
                if p_benchmark.is_alive():
                    log("waiting benchmark threading to join")
                p_benchmark.join()
            p_benchmark=Process(target=benchmark,args=(save_name,epoch,dev_bench_num))
            p_benchmark.start()

        if (epoch<=5) or (epoch<30 and epoch%5==0) or epoch%20==0:
            output_flag=True
            log("gc len at %d: %d"%(epoch,len(gc.get_objects())))
        else:
            output_flag=False

        if epoch>=review_number:
            train_datas=train_datas[len(train_datas)//review_number:]
        train_datas+=prepare_data(copy.deepcopy(pv_net),dev_train_num,data_rounds,train_mcts_b,train_mcts_k)

        trainloader=torch.utils.data.DataLoader(train_datas,batch_size=64,drop_last=True,shuffle=True)
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

    p_benchmark.join()
    
def main():
    from MrZeroTree import BETA,MCTS_EXPL,BENCH_SMP_B,BENCH_SMP_K
    log("BETA: %.2f, VALUE_RENORMAL: %d, MCTS_EXPL: %d, BENCH_SMP_B: %d, BENCH_SMP_K: %.1f"\
        %(BETA,VALUE_RENORMAL,MCTS_EXPL,BENCH_SMP_B,BENCH_SMP_K))
    #pv_net=PV_NET_C();log("init pv_net: %s"%(pv_net))
    start_from="./ZeroNets/from-zero-18/PV_NET_B-17-9315381-2160.pkl"
    pv_net=torch.load(start_from);log("start from: %s"%(start_from))
    train(pv_net)


if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()