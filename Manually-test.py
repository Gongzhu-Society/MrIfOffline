#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
from MrGreed import MrGreed
from MrZeroTree import MrZeroTree,PV_NET
from OfflineInterface import OfflineInterface
import torch

def benchmark(save_name,mcts_searchnum=None,pv_deep=None,print_process=True):
    """
        benchmark raw network against MrGreed
        METHOD=-1, N1=512, 7min
        METHOD=-2, N1=512, 3.5min
    """
    import itertools,numpy

    N1=128;N2=2;
    log("start benchmark against MrGreed for %dx%d, file: %s"%(N1,N2,save_name))
    log("benchmark method: %d, pv_deep: %d"%(mcts_searchnum,pv_deep))

    device_bench=torch.device("cuda:3")
    pv_net=torch.load(save_name)
    pv_net.to(device_bench)

    zt=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net,device=device_bench,mcts_searchnum=mcts_searchnum,pv_deep=pv_deep) for i in [0,2]]
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
        if l==N2-1:
            if print_process:
                log("%4d: %s"%(sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2,stats[-N2:]))
            else:
                print("%4d"%(sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2),end=" ",flush=True)
    s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    log("benchmark result: %.2f %.2f"%(numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1))))

def plot_log(fileperfix):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator,MultipleLocator
    import os,re

    lines=[]
    for i in "abcdefghijklmnopqrstuvwxyz":
        fname="./ZeroNets/%s%s.log"%(fileperfix,i)
        if not os.path.exists(fname):
            break
        log("reading %s..."%(fname))
        with open(fname,'r') as f:
            lines+=f.readlines()
    p_bench=re.compile("benchmark at epoch ([0-9]+)'s result: ([\\-\\.0-9]+) ([0-9\\.]+)")
    t_bench=[];v_bench=[];e_bench=[]
    bias_bench=[0,]
    for l in lines:
        s_bench=p_bench.search(l)
        if s_bench:
            epoch=int(s_bench.group(1))
            if epoch==0 and len(t_bench)!=0:
                bias_bench.append(t_bench[-1])
            t_bench.append(epoch+bias_bench[-1])
            v_bench.append(float(s_bench.group(2)))
            e_bench.append(float(s_bench.group(3)))
    p_loss=re.compile("\\[INFO,train:[0-9]+\\] ([0-9]+): ([0-9\\.]+) ([0-9\\.]+)")
    t_loss=[];v_loss=[]
    ax=-1
    for l in lines:
        s_loss=p_loss.search(l)
        if s_loss:
            epoch=int(s_loss.group(1))
            if epoch==0:
                ax+=1
            if epoch%20!=0 and epoch%50!=0:
                continue
            if len(bias_bench)>ax+1 and epoch>bias_bench[ax+1]-bias_bench[ax]:
                continue
            t_loss.append(epoch+bias_bench[ax])
            v_loss.append(float(s_loss.group(3)))
    log(bias_bench)
    log(t_bench)
    log(t_loss)
    fig=plt.figure()
    fig.set_size_inches(8,6)
    ax1=fig.subplots(1)
    ax2=ax1.twinx()

    ax2.errorbar(t_bench,v_bench,yerr=e_bench,fmt='o--',capsize=5,label="Raw Value Network")
    ax2.axhline(y=-80.3,dashes=(3,3),c='g',lw=3,label="Mr. If")
    ax1.plot(t_loss,v_loss,'y^-',label="Loss2")
    ax1.grid(True,which='both',axis='x')
    ax2.grid(True,which='both',axis='y')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.set_ylim((20,60))
    ax1.legend()#loc=
    ax2.legend()
    plt.title(fileperfix)
    plt.savefig(fileperfix+".png")

if __name__ == '__main__':
    #plot_log("from-one-11")
    try:
        benchmark("./ZeroNets/from-one-6g/PV_NET-11-2247733-600.pkl",mcts_searchnum=-1,pv_deep=6,print_process=False)
    except:
        log("",l=3)