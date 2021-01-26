#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log

#torch.save(pv_net_0.state_dict(),"Zero-29th-25-11416629-720.pt")

def log_source(s):
    s=s.split("\n")
    s2=[]
    for j,i in enumerate(s):
        if not i.strip().startswith("#") and len(i.strip())>0:
            if i.strip().startswith("if"):
                s2.append(s[j-1])
            s2.append(i)
    print("\n".join(s2))

def benchmark(print_process=False):
    from MrGreed import MrGreed
    from MrImpGreed import MrImpGreed
    from MrZeroTreeSimple import MrZeroTreeSimple
    from MrZeroTree import MrZeroTree
    from MrZ_NETs import PV_NET_2
    from OfflineInterface import OfflineInterface
    import itertools,torch,random,inspect

    mode=0
    complete_info=False
    log_source(inspect.getsource(MrZeroTree.decide_rect_necessity))
    log("complete info: %s, mode: %s"%(complete_info,mode))

    device_bench=torch.device("cuda:3")
    save_name_0="Zero-29th-25-11416629-720.pt"
    state_dict_0=torch.load(save_name_0,map_location=device_bench)
    pv_net_0=PV_NET_2()
    pv_net_0.load_state_dict(state_dict_0)
    pv_net_0.to(device_bench)

    if mode==0:
        log("Tree v.s. Greed mode")
        zt0=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net_0,device=device_bench,
                        mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-2) for i in [0,2]]
        g_aux=[None,None,None,None]
        g_aux[0]=MrGreed(room=255,place=0,name='gaux');g_aux[2]=MrGreed(room=255,place=2,name='gaux')
        zt0[0].g_aux=g_aux;zt0[1].g_aux=g_aux
        g=[MrGreed(room=255,place=i,name='greed%d'%(i)) for i in [1,3]]
        interface=OfflineInterface([zt0[0],g[0],zt0[1],g[1]],print_flag=False)
    elif mode==1:
        input("1 MrZeroTree v.s. 3 MrGreed mode")
    elif mode==2:
        log("Tree v.s. TreeSimple mode")
        zt=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net_0,device=device_bench,
                        mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-2) for i in [0,2]]
        zts=[MrZeroTreeSimple(room=255,place=i,name='ztsimple%d'%(i),pv_net=pv_net_0,device=device_bench,
                        mcts_b=10,mcts_k=2,sample_b=9,sample_k=0) for i in [1,3]]
        interface=OfflineInterface([zt[0],zts[0],zt[1],zts[1]],print_flag=False)

    N1=1024;N2=2;
    log("(%s+%s) v.s. (%s+%s) for %dx%d on %s"%(interface.players[0].family_name(),interface.players[2].family_name(),
                                          interface.players[1].family_name(),interface.players[3].family_name(),N1,N2,device_bench))
    if interface.players[0].family_name().startswith("MrZeroTree"):
        log("mcts_b/k: %d/%d, sample_b/k: %d/%d"%(interface.players[0].mcts_b,interface.players[0].mcts_k,
                                                  interface.players[0].sample_b,interface.players[0].sample_k))
    if interface.players[1].family_name().startswith("MrZeroTree"):
        log("mcts_b/k: %d/%d, sample_b/k: %d/%d"%(interface.players[1].mcts_b,interface.players[1].mcts_k,
                                                  interface.players[1].sample_b,interface.players[1].sample_k))
    stats=[]
    for k,l in itertools.product(range(N1),range(N2)):
        if l==0:
            cards=interface.shuffle()
        else:
            cards=cards[39:52]+cards[0:39]
            interface.shuffle(cards=cards)
        for i,j in itertools.product(range(13),range(4)):
            if complete_info and interface.players[interface.pnext].family_name().startswith("MrZeroTree"):
                interface.step_complete_info()
            else:
                interface.step()
            #input("continue...")
        stats.append(interface.clear())
        interface.prepare_new()
        if l==N2-1:
            if print_process:
                log("%2d %4d: %s"%(k,sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2,stats[-N2:]))
            else:
                print("%4d"%(sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2),end=" ",flush=True)
        if (k+1)%256==0 and l==N2-1:
            bench_stat(stats,N2,device_bench)
    bench_stat(stats,N2,device_bench)

def bench_stat(stats,N2,comments):
    import numpy
    print("")
    s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    s_temp=[sum(s_temp[i:i+N2])/N2 for i in range(0,len(s_temp),N2)]
    log("benchmark result: %.2f %.2f"%(numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1))))
    suc_ct=len([1 for i in s_temp if i>0])
    draw_ct=len([1 for i in s_temp if i==0])
    log("success rate: (%d+%d)/%d"%(suc_ct,draw_ct,len(s_temp)))
    log(comments)

def plot_log(fileperfixes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator,MultipleLocator
    import os,re

    lines=[]
    for fileperfix in fileperfixes:
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
    #ax2=ax1.twinx()

    ax1.errorbar(t_bench,v_bench,yerr=e_bench,fmt='o--',capsize=5,label="Raw Value Network")
    ax1.axhline(y=-275.2,dashes=(2,2),c='orange',lw=3,label="Mr. Random")
    ax1.axhline(y=-80.3,dashes=(2,2),c='limegreen',lw=3,label="Mr. If")
    ax1.axhline(y=0,dashes=(2,2),c='green',lw=3,label="Mr. Greed")


    #ax1.plot(t_loss,v_loss,'^-',c='tomato',label="Loss2")

    ax1.set_xlabel('Epoch')
    #ax1.set_ylabel('Loss2 (Estimated)')
    ax1.grid(True,which='both',axis='x')
    ax1.set_ylabel('Benchmark Results')
    ax1.grid(True,which='both',axis='y')
    ax1.xaxis.set_minor_locator(AutoMinorLocator(2))
    #ax1.set_ylim((20,60))
    ax1.legend(loc=2)#loc=
    #ax2.legend()
    plt.title(fileperfix)
    plt.savefig(fileperfix+".png")

def stat_rect_log(fname):
    import re,numpy
    p_num=re.compile("[0-1]\\.[0-9]{4}")
    with open(fname,"r") as f:
        lines=f.readlines()
    greed_stat=[]
    zero_stat=[]
    for l in lines:
        l=l.split(", ")
        if l[0].endswith("greed"):
            greed_stat.append(l[1:])
        elif l[0].endswith("zerotree"):
            zero_stat.append(l[1:])

    sublists=[]
    sublists.append([i for i in greed_stat if i[0][1] in "234567" and i[1]=="A"])
    sublists.append([i for i in greed_stat if i[0][1] not in "234567" and i[1]=="A"])
    sublists.append([i for i in greed_stat if i[0][1] in "234567" and i[1]!="A"])
    sublists.append([i for i in greed_stat if i[0][1] not in "234567" and i[1]!="A"])
    sublists.append([i for i in zero_stat if i[0][1] in "234567" and i[1]=="A"])
    sublists.append([i for i in zero_stat if i[0][1] not in "234567" and i[1]=="A"])
    sublists.append([i for i in zero_stat if i[0][1] in "234567" and i[1]!="A"])
    sublists.append([i for i in zero_stat if i[0][1] not in "234567" and i[1]!="A"])

    for i in sublists:
        log(i[0])
        l1=[1 for j in i if j[2]=="True"]
        l2=[float(j[3].strip()) for j in i]
        log("%d/%d %.4f %.4f"%(len(l1),len(i),numpy.mean(l2),numpy.var(l2)))

if __name__ == '__main__':
    #plot_log(["from-zero-26","from-zero-29"])
    benchmark()
    #stat_rect_log("stat_Q2.txt")