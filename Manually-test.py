#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
from MrGreed import MrGreed
from MrZeroTree import MrZeroTree,PV_NET
from OfflineInterface import OfflineInterface

import torch,itertools,numpy

def benchmark(save_name,mcts_searchnum=200,device_num=3,print_process=True):
    """
        benchmark raw network against MrGreed
        METHOD=-1, N1=512, 7min
        METHOD=-2, N1=512, 3.5min
    """
    N1=1024;N2=2;
    log("start benchmark against MrGreed for %dx%d"%(N1,N2))
    log("benchmark method: %d, file: %s"%(mcts_searchnum,save_name))

    device_bench=torch.device("cuda:%d"%(device_num))
    pv_net=torch.load(save_name)
    pv_net.to(device_bench)

    zt=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_bench,mcts_searchnum=mcts_searchnum) for i in [0,2]]
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
            log("%s %4d"%(stats[-N2:],sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2))
    s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    log("benchmark result: %.2f %.2f"%(numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1))))

if __name__ == '__main__':
    try:
        benchmark("./ZeroNets/from-one-6f/PV_NET-11-2247733-600.pkl")
    except:
        log("",l=3)