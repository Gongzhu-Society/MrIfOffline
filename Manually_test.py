#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
import numpy,itertools

def log_source(s):
    s=s.split("\n")
    s2=[]
    for j,i in enumerate(s):
        if not i.strip().startswith("#") and len(i.strip())>0:
            #if i.strip().startswith("if"):
            #    s2.append(s[j-1])
            s2.append(i)
    log("\n".join(s2))

def benchmark(handsfile,print_process=False):
    from MrGreed import MrGreed
    #from MrZeroTree import MrZeroTree
    from MrZeroTreeSimple import MrZeroTreeSimple
    from OfflineInterface import OfflineInterface,read_std_hands,play_a_test
    import torch,inspect

    #log_source(inspect.getsource(MrZeroTree.decide_rect_necessity))
    #log_source(inspect.getsource(MrZeroTree.possi_rectify_pvnet))

    #zt0=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-2) for i in [0,2]]
    zt0=[MrZeroTreeSimple(room=255,place=i,name='zt%d'%(i),pv_net="Zero-29th-25-11416629-720.pt",tree_deep=2,sample_b=5,sample_k=0,device="cuda:0") for i in [0,2]]
    team1=[MrGreed(room=255,place=i,name='greed%d'%(i)) for i in [1,3]]
    interface=OfflineInterface([zt0[0],team1[0],zt0[1],team1[1]],print_flag=False)

    if interface.players[0].family_name().startswith("MrZeroTree"):
        p0=interface.players[0]
        #log("mcts_b/k: %d/%d, sample_b/k: %d/%d"%(p0.mcts_b,p0.mcts_k,p0.sample_b,p0.sample_k))
        log("sample_b/k: %d/%d, tree_deep: %d"%(p0.sample_b,p0.sample_k,p0.tree_deep))

    hands=read_std_hands(handsfile)
    N1=256;N2=2
    log("%s for %dx%d on %s"%(interface,N1,N2,zt0[0].device))
    stats=[]
    for k,hand in hands:
        stats.append(play_a_test(interface,hand,N2))
        print("%4d"%(stats[-1],),end=" ",flush=True)
        if (k+1)%(N1//4)==0:
            bench_stat(stats,N2)
    bench_stat(stats,N2)

def bench_stat(stats,N2,comments=None):
    print("")
    #s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    #s_temp=[sum(s_temp[i:i+N2])/N2 for i in range(0,len(s_temp),N2)]
    log("benchmark result: %.2f %.2f"%(numpy.mean(stats),numpy.sqrt(numpy.var(stats)/(len(stats)-1))))
    suc_ct=len([1 for i in stats if i>0])
    draw_ct=len([1 for i in stats if i==0])
    log("success rate: (%d+%d)/%d"%(suc_ct,draw_ct,len(stats)))
    #low_ct=len([1 for i in s_temp if i<-250])
    #high_ct=len([1 for i in s_temp if i>400])
    #log("low(<-250),high(>400): (%d,%d)/%d"%(low_ct,high_ct,len(s_temp)))
    if comments!=None:
        log(comments)

def benchmark_B(handsfile):
    from MrIf import MrIf
    from MrGreed import MrGreed
    from MrZeroTree import MrZeroTree
    from OfflineInterface import OfflineInterface,read_std_hands,play_a_test

    ifs=[MrIf(room=255,place=i,name='I%d'%(i)) for i in range(4)]
    gs=[MrGreed(room=255,place=i,name='G%d'%(i)) for i in range(4)]
    zs=[MrZeroTree(room=255,place=i,name='Z%d'%(i),mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-2) for i in [0,2]]
    I_GI=OfflineInterface([gs[0],ifs[1],gs[2],ifs[3]],print_flag=False)
    I_ZG=OfflineInterface([zs[0],gs[1],zs[1],gs[3]],print_flag=False)

    hands=read_std_hands(handsfile)
    stats=[]
    for k,hand in hands:
        stats.append(play_a_test(I_ZG,hand,2))
        print("%4d"%(stats[-1],),end=" ",flush=True)
    else:
        print("")
    log("benchmark result: %.2f %.2f"%(numpy.mean(stats),numpy.sqrt(numpy.var(stats)/(len(stats)-1))))

if __name__ == '__main__':
    benchmark("StdHands/random_0_1024.hands")
    #benchmark_B("StdHands/selectA_0_322.hands")