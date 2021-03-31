#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
from MrZeroTreeSimple import benchmark,prepare_data,prepare_inference_data

from torch import device
import torch.nn.functional as F
from torch.multiprocessing import Process
import torch.multiprocessing

import copy,itertools,numpy,time

def train(pv_net,dev_train_num=0,dev_bench_num=0,args={}):
    import torch.optim as optim
    import gc
    data_rounds=args['data_rounds']
    loss2_weight=0.03
    train_mcts_b=5
    train_mcts_k=4
    review_number=args['review_number']
    age_in_epoch=3
    log("loss2_weight: %.2f, data_rounds: %d, train_mcts_b: %d, train_mcts_k: %.1f, review_number: %d, age_in_epoch: %d"
        %(loss2_weight,data_rounds,train_mcts_b,train_mcts_k,review_number,age_in_epoch))

    device_main=device("cuda:%d"%(dev_train_num))
    pv_net.to(device_main)
    names = pv_net.__str__().split()
    if names[0] in {"PV_NET_3","PV_NET_4"}:
        optimizer = optim.Adam(pv_net.parameters(), lr=0.0001, eps=1e-07, weight_decay=1e-4,
                               amsgrad=False)
    else:
        optimizer=optim.Adam(pv_net.parameters(),lr=0.0001,betas=(0.3,0.999),eps=1e-07,weight_decay=1e-4,amsgrad=False)
    log("optimizer: %s"%(optimizer.__dict__['defaults'],))

    train_datas=[]
    p_benchmark=None
    for epoch in range(2401):
        if epoch%80==0:
            save_name='%s-%s-%s-%s-%d.pkl'%(pv_net.__class__.__name__,__file__[-5:-3],pv_net.num_layers(),pv_net.num_paras(),epoch)
            #torch.save(pv_net,save_name)
            torch.save(pv_net.state_dict(),save_name)
            '''if p_benchmark!=None:
                if p_benchmark.is_alive():
                    log("waiting benchmark threading to join")
                p_benchmark.join()
            p_benchmark=Process(target=benchmark,args=(save_name,epoch,dev_bench_num))
            p_benchmark.start()
            time.sleep(3600)
            '''
            print("benchmarking")
            benchmark(save_name,epoch, dev_bench_num, False,args=args)

        if (epoch<=5) or (epoch<30 and epoch%5==0) or epoch%20==0:
            output_flag=True
        else:
            output_flag=False

        if epoch>=review_number:
            train_datas=train_datas[len(train_datas)//review_number:]
        train_datas+=prepare_data(copy.deepcopy(pv_net),dev_train_num,data_rounds,train_mcts_b,train_mcts_k,args=args)
        trainloader=torch.utils.data.DataLoader(train_datas,batch_size=64,drop_last=True,shuffle=True)

        if output_flag:
            #show grad
            batch=trainloader.__iter__().__next__()
            p,v=pv_net(batch[0].to(device_main))
            optimizer.zero_grad()
            log_p=F.log_softmax(p*batch[3].to(device_main),dim=1)
            loss1_t=F.kl_div(log_p,batch[1].to(device_main),reduction="batchmean")
            loss1_t.backward(retain_graph=True)
            grad1=pv_net.fcp.weight.grad.abs().mean().item()
            optimizer.zero_grad()
            loss2_t=F.mse_loss(v.view(-1),batch[2].to(device_main),reduction='mean').sqrt()
            loss2_t.backward(retain_graph=True)
            grad2=pv_net.fcv.weight.grad.abs().mean().item()
            log("dloss at %d: %.4f %.4f %.4f"%(epoch,grad1,grad2,grad1/grad2))

        for age in range(age_in_epoch):
            running_loss1=[];running_loss2=[]
            for batch in trainloader:
                #print("batch size is", len(batch), len(batch[0]))
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
                    test_loss1=running_loss1*review_number-last_loss1*(review_number-1)
                    test_loss2=running_loss2*review_number-last_loss2*(review_number-1)
                log("%d: %.3f %.2f %d %d"%(epoch,test_loss1,test_loss2,len(train_datas),batchnum))

            if age==age_in_epoch-1:
                last_loss1=running_loss1
                last_loss2=running_loss2

            if output_flag:
                log("        epoch %d age %d: %.3f %.2f"%(epoch,age,running_loss1,running_loss2))

    #p_benchmark.join()
    log('C\'est fini!')

def train_guesser(pv_net, guesser_net,dev_train_num=0,args={},dev_bench_num=0):
    import torch.optim as optim
    import gc
    from inference import SimpleGuesser, Loss0, Buffer
    data_rounds=args['data_rounds']
    loss2_weight=0.03
    train_mcts_b=0
    train_mcts_k=2
    review_number=3
    age_in_epoch=args['age_in_epoch']
    log("loss2_weight: %.2f, data_rounds: %d, train_mcts_b: %d, train_mcts_k: %.1f, review_number: %d, age_in_epoch: %d"
        %(loss2_weight,data_rounds,train_mcts_b,train_mcts_k,review_number,age_in_epoch))

    device_main=device("cuda:%d"%(dev_train_num))
    guesser_net.to(device_main)
    pv_net.to(device_main)

    names = guesser_net.__str__().split()
    if names[0] in {"Guessing_net_1"}:
        optimizer = optim.Adam(guesser_net.parameters(), lr=0.0001, eps=1e-07, weight_decay=1e-4,
                               amsgrad=False)
    else:
        optimizer=optim.Adam(guesser_net.parameters(),lr=0.0001,betas=(0.3,0.999),eps=1e-07,weight_decay=1e-4,amsgrad=False)
    log("optimizer: %s"%(optimizer.__dict__['defaults'],))

    train_datas=[]
    p_benchmark=None
    Criterion = Loss0()
    for epoch in range(2401):
        if epoch%80==0:
            save_name='%s-%s-%s-%s-%d.pkl'%(guesser_net.__class__.__name__,__file__[-5:-3],guesser_net.num_layers(),guesser_net.num_paras(),epoch)
            #torch.save(pv_net,save_name)
            torch.save(guesser_net.state_dict(),save_name)
            #print("we should benchmark here, but I don't have time to implement")
            #benchmark(save_name,epoch, dev_bench_num, False)


        raw_train_datas = prepare_inference_data(pv_net,dev_train_num, data_rounds,args=args)
        #print(raw_train_datas[0])
        train_datas = [(SimpleGuesser.prepare_ohs(sample[3], sample[2], sample[0]), SimpleGuesser.prepare_target_ohs(sample[1], sample[0])) for sample in raw_train_datas]
        #print("len train data", len(train_datas), len(train_datas[0]))
        '''
        if epoch>=review_number:
            train_datas=train_datas[len(train_datas)//review_number:]
        train_datas+=prepare_data(copy.deepcopy(pv_net),dev_train_num,data_rounds,train_mcts_b,train_mcts_k)
        '''

        trainloader=torch.utils.data.DataLoader(Buffer(train_datas),batch_size=args['batch_size'],shuffle=True)

        if (epoch<=5) or (epoch<30 and epoch%5==0) or epoch%20==0:
            output_flag=True
        else:
            output_flag=False

        for age in range(age_in_epoch):
            running_loss1=[];running_loss2=[]
            for input, label in trainloader:
                #print("input size is", input.size())
                #print("label size is", label.size())
                #input = [SimpleGuesser.prepare_ohs(sample[3],sample[2],sample[0]) for sample in batch.to(device_main)]
                #print("input len", len(input))
                prediction=guesser_net(input.to(device_main))
                loss1 = Criterion(label.to(device_main),prediction)
                loss1.backward()
                optimizer.step()
                running_loss1.append(loss1.item())
                #running_loss2.append(loss2.item())
            batchnum=len(running_loss1)
            running_loss1=numpy.mean(running_loss1)
            #running_loss2=numpy.mean(running_loss2)

            if output_flag:
                log("%d in %d: %.3f %d %d"%(age, epoch,running_loss1,len(train_datas),batchnum))
            '''
            if age==age_in_epoch-1:
                last_loss1=running_loss1
                last_loss2=running_loss2

            if output_flag:
                log("        epoch %d age %d: %.3f %.2f"%(epoch,age,running_loss1,running_loss2))
            '''
    #p_benchmark.join()
    log('C\'est fini!')

def main(args):
    from MrZeroTreeSimple import BETA,MCTS_EXPL,BENCH_SMP_B,BENCH_SMP_K
    from MrZ_NETs import VALUE_RENORMAL
    log("BETA: %.2f, VALUE_RENORMAL: %d, MCTS_EXPL: %d, BENCH_SMP_B: %d, BENCH_SMP_K: %.1f"\
        %(BETA,VALUE_RENORMAL,MCTS_EXPL,BENCH_SMP_B,BENCH_SMP_K))

    from MrZ_NETs import PV_NET_2, PV_NET_3, PV_NET_4, PV_NET_5, RES_NET_18
    dev_train=0
    start_from=args['start_from'] # or a path to netpara file
    if args['pv_net'] in {'PV_NET_4'}:
        pv_net=PV_NET_4()#RES_NET_18()#PV_NET_2()
    elif args['pv_net'] in {'PV_NET_5'}:
        pv_net=PV_NET_5()
    else:
        pv_net = PV_NET_3()
    log("init pv_net: %s"%(pv_net))
    if start_from=='None':
        log("start from: zero")
    else:
        pv_net.load_state_dict(torch.load(start_from))
        log("start_from: %s"%(start_from))
    train(pv_net,dev_train_num=dev_train,dev_bench_num=0,args=args)

def main_for_guesser(args):
    from MrZeroTreeSimple import BETA,MCTS_EXPL,BENCH_SMP_B,BENCH_SMP_K
    from MrZ_NETs import VALUE_RENORMAL
    log("BETA: %.2f, VALUE_RENORMAL: %d, MCTS_EXPL: %d, BENCH_SMP_B: %d, BENCH_SMP_K: %.1f"\
        %(BETA,VALUE_RENORMAL,MCTS_EXPL,BENCH_SMP_B,BENCH_SMP_K))

    from MrZ_NETs import PV_NET_2, PV_NET_3, PV_NET_4, Guessing_net_1, RES_NET_18
    dev_train=0
    start_from=args['start_from'] # or a path to netpara file
    if args['pv_net'] in {'PV_NET_4'}:
        pv_net = PV_NET_4()  # RES_NET_18()#PV_NET_2()
    else:
        pv_net = PV_NET_3()
    log("init pv_net: %s" % (pv_net))
    if start_from == 'None':
        log("start from: zero")
    else:
        pv_net.load_state_dict(torch.load(start_from))
        log("start_from: %s" % (start_from))

    gs_net=Guessing_net_1()#RES_NET_18()#PV_NET_2()
    log("init gs_net: %s"%(gs_net))
    if start_from=='None':
        log("start from: zero")
    else:
        gs_net.load_state_dict(torch.load(start_from))
        log("start_from: %s"%(start_from))
    train_guesser(pv_net, gs_net, dev_train_num=dev_train, dev_bench_num=0, args=args)

if __name__=="__main__":
    torch.multiprocessing.set_start_method('spawn')

    f = open("setting.txt", 'r')
    args = eval(f.read())
    f.close()
    #main_for_guesser(args)
    main(args)
