#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log

def benchmark(print_process=True):
    """
        benchmark raw network against MrGreed
        METHOD=-1, N1=512, 7min
        METHOD=-2, N1=512, 3.5min
    """
    from MrGreed import MrGreed
    from MrZeroTree import MrZeroTree
    from MrZ_NETs import PV_NET
    from OfflineInterface import OfflineInterface
    import itertools,numpy,torch

    against_greed=True

    device_bench=torch.device("cuda:0")
    save_name_0="./ZeroNets/from-zero-29b/PV_NET-B-25-11416629-240.pkl"
    pv_net_0=torch.load(save_name_0,map_location=device_bench)

    if not against_greed:
        del save_name_0
        #save_name_1="./ZeroNets/from-zero-9a/PV_NET-17-9479221-560.pkl"
        pv_net_1=torch.load(save_name_1,map_location=device_bench)
        pv_net_1.to(device_bench)
        del save_name_1

    zt0=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net_0,device=device_bench,
                   mcts_b=10,mcts_k=2,sample_b=5,sample_k=0) for i in [0,2]]
    if against_greed:
        g=[MrGreed(room=255,place=i,name='greed%d'%(i)) for i in [1,3]]
        interface=OfflineInterface([zt0[0],g[0],zt0[1],g[1]],print_flag=False)
    else:
        zt1=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net_1,device=device_bench,mcts_searchnum=mcts_searchnum,pv_deep=pv_deep,N_SAMPLE=n_sample) for i in [1,3]]
        interface=OfflineInterface([zt[0],zt1[0],zt0[1],zt1[1]],print_flag=False)

    N1=256;N2=2;
    log("%s v.s. %s for %dx%d"%(interface.players[0].__class__.__name__,interface.players[1].__class__.__name__,N1,N2))
    stats=[]
    for k,l in itertools.product(range(N1),range(N2)):
        if l==0:
            cards=interface.shuffle()
        else:
            cards=cards[39:52]+cards[0:39]
            interface.shuffle(cards=cards)
        for i,j in itertools.product(range(13),range(4)):
            interface.step()
            #input("continue...")
        stats.append(interface.clear())
        interface.prepare_new()
        if l==N2-1:
            if print_process:
                log("%2d %4d: %s"%(k,sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2,stats[-N2:]))
            else:
                print("%4d"%(sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2),end=" ",flush=True)
    s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    s_temp=[sum(s_temp[i:i+N2])/N2 for i in range(0,len(s_temp),N2)]
    log("benchmark result: %.2f %.2f"%(numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1))))

def plot_log(fileperfix):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator,MultipleLocator
    import os,re

    lines=[]
    for i in ([""]+list("abcdefghijklmnopqrstuvwxyz")):
        fname="./ZeroNets/%s%s.log"%(fileperfix,i)
        if not os.path.exists(fname):
            continue
        log("reading %s..."%(fname),l=0)
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
    log(bias_bench,l=0)
    log(t_bench,l=0)
    log(t_loss,l=0)
    fig=plt.figure()
    fig.set_size_inches(8,6)
    ax1=fig.subplots(1)
    ax2=ax1.twinx()

    ax2.errorbar(t_bench,v_bench,yerr=e_bench,fmt='o--',capsize=5,label="Raw Value Network")
    ax2.axhline(y=-80.3,dashes=(2,2),c='limegreen',lw=2,label="Mr. If")
    ax2.axhline(y=0,dashes=(2,2),c='green',lw=2,label="Mr. Greed")


    ax1.plot(t_loss,v_loss,'^-',c='tomato',label="Loss2")

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss2 (Estimated)')
    ax1.grid(True,which='both',axis='x')
    ax2.set_ylabel('Benchmark Result (with Error Bar)')
    ax2.grid(True,which='both',axis='y')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax1.set_ylim((20,60))
    ax1.legend(loc=2)#loc=
    ax2.legend()
    plt.title(fileperfix)
    plt.savefig(fileperfix+".png")

if __name__ == '__main__':
    #plot_log("from-zero-33")
    benchmark()