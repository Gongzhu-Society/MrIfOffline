#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log

def benchmark_transitivity(print_process=False):
    from MrRandom import MrRandom
    from MrIf import MrIf
    from MrGreed import MrGreed
    from MrRandTree import MrRandTree
    from MrZ_NETs import PV_NET_2
    from MrZeroTreeSimple import MrZeroTreeSimple
    from OfflineInterface import OfflineInterface
    import itertools,torch,random,inspect

    device_bench=torch.device("cuda:0")
    save_name_0="Zero-29th-25-11416629-720.pt"
    state_dict_0=torch.load(save_name_0,map_location=device_bench)
    pv_net_0=PV_NET_2()
    pv_net_0.load_state_dict(state_dict_0)
    pv_net_0.to(device_bench)
    team0=[MrZeroTreeSimple(room=255,place=i,name='zts%d'%(i),pv_net=pv_net_0,device=device_bench,mcts_b=10,mcts_k=2,sample_b=9,sample_k=0) for i in [0,2]]
    #team0=[MrRandTree(room=255,place=i,name='randtree%d'%(i)) for i in [0,2]]

    #team1=[MrRandTree(room=255,place=i,name='randtree%d'%(i)) for i in [1,3]]
    #team1=[MrGreed(room=255,place=i,name='greed%d'%(i)) for i in [1,3]]
    #team1=[MrRandom(room=255,place=i,name='random%d'%(i)) for i in [1,3]]
    team1=[MrIf(room=255,place=i,name='if%d'%(i)) for i in [1,3]]
    interface=OfflineInterface([team0[0],team1[0],team0[1],team1[1]],print_flag=False)

    N1=256;N2=2;
    log("(%s+%s) v.s. (%s+%s) for %dx%d"%(interface.players[0].family_name(),interface.players[2].family_name(),
                                            interface.players[1].family_name(),interface.players[3].family_name(),N1,N2))
    if interface.players[0].family_name().startswith("MrZeroTree"):
        log("mcts_b/k: %d/%d, sample_b/k: %d/%d"%(interface.players[0].mcts_b,interface.players[0].mcts_k,
                                                  interface.players[0].sample_b,interface.players[0].sample_k))
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
        if (k+1)%(N1//4)==0 and l==N2-1:
            bench_stat(stats,N2,None)
    bench_stat(stats,N2,None)
    
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
    log(t_bench)
    log(v_bench)
    log(e_bench)
    return

def add_dict(l,d):
    for i in l.split(","):
        i=i.strip().split(":")
        c=i[0];v=float(i[1])
        if c not in d:
            d[c]=[v]
        else:
            d[c].append(v)

def stat_r_log(fname):
    import re,numpy
    from Util import INIT_CARDS
    with open(fname,"r") as f:
        lines=f.readlines()
    dict_val={};dict_reg={}
    for l in lines:
        if "reg" in l:
            add_dict(l.split("reg")[1],dict_reg)
        elif "r/beta" in l:
            add_dict(l.split("r/beta")[1],dict_val)
    l_val=[(c,numpy.mean(dict_val[c]),numpy.sqrt(numpy.var(dict_val[c])),numpy.var(dict_val[c])/numpy.sqrt(len(dict_val[c])-1)) for c in INIT_CARDS]
    l_reg=[(c,numpy.mean(dict_reg[c]),numpy.sqrt(numpy.var(dict_reg[c])),numpy.var(dict_reg[c])/numpy.sqrt(len(dict_reg[c])-1)) for c in INIT_CARDS]

    """line_vals=[]
    for i in range(4):
        val=[v for c,v,s,e in l_val[i*13:i*13+13]]
        err=[e for c,v,s,e in l_val[i*13:i*13+13]]
        line_vals.append((l_val[i*13][0][0],val,err))"""
    line_val_vars=[]
    for i in range(4):
        var=[float("%.4f"%(s)) for c,v,s,e in l_val[i*13:i*13+13]]
        line_val_vars.append((l_val[i*13][0][0],var))
    line_regs=[]
    for i in range(4):
        val=[float("%.4f"%(v)) for c,v,s,e in l_reg[i*13:i*13+13]]
        err=[float("%.1e"%(e)) for c,v,s,e in l_reg[i*13:i*13+13]]
        line_regs.append((l_reg[i*13][0][0],val,err))
    """line_reg_vars=[]
    for i in range(4):
        var=[s for c,v,s,e in l_reg[i*13:i*13+13]]
        line_reg_vars.append((l_reg[i*13][0][0],var))"""
    log(line_regs)
    log(line_val_vars)

if __name__ == '__main__':
    #plot_log(["from-zero-26","from-zero-29"])
    #benchmark_transitivity()