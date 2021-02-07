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
    log("\n".join(s2))
    
def benchmark(handsfile,print_process=False):
    from MrGreed import MrGreed
    #from MrZeroTreeSimple import MrZeroTreeSimple
    from MrZeroTree import MrZeroTree
    from MrZ_NETs import PV_NET_2
    from OfflineInterface import OfflineInterface,read_std_hands
    import itertools,torch,random,inspect

    #log_source(inspect.getsource(MrZeroTree.decide_rect_necessity))
    #log_source(inspect.getsource(MrZeroTree.possi_rectify_pvnet))
    
    mode=0
    
    device_bench=None
    """device_bench=torch.device("cuda:1")
    save_name_0="Zero-29th-25-11416629-720.pt"
    state_dict_0=torch.load(save_name_0,map_location=device_bench)
    pv_net_0=PV_NET_2()
    pv_net_0.load_state_dict(state_dict_0)
    pv_net_0.to(device_bench)"""

    if mode==0:
        #zt0=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net_0,device=device_bench,
                        #mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-2) for i in [0,2]]
                        #mcts_b=10,mcts_k=2,sample_b=9,sample_k=0) for i in [0,2]]
        zt0=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-2) for i in [0,2]]
        team1=[MrGreed(room=255,place=i,name='greed%d'%(i)) for i in [1,3]]
        interface=OfflineInterface([zt0[0],team1[0],zt0[1],team1[1]],print_flag=False)
        device_bench=zt0[0].device
    elif mode==2:
        log("Tree v.s. TreeSimple mode")
        zt=[MrZeroTree(room=255,place=i,name='zerotree%d'%(i),pv_net=pv_net_0,device=device_bench,
                        mcts_b=10,mcts_k=2,sample_b=-1,sample_k=-2) for i in [0,2]]
        zts=[MrZeroTreeSimple(room=255,place=i,name='ztsimple%d'%(i),pv_net=pv_net_0,device=device_bench,
                        mcts_b=10,mcts_k=2,sample_b=9,sample_k=0) for i in [1,3]]
        interface=OfflineInterface([zt[0],zts[0],zt[1],zts[1]],print_flag=False)

    
    if interface.players[0].family_name().startswith("MrZeroTree"):
        log("mcts_b/k: %d/%d, sample_b/k: %d/%d"%(interface.players[0].mcts_b,interface.players[0].mcts_k,
                                                  interface.players[0].sample_b,interface.players[0].sample_k))
    if interface.players[1].family_name().startswith("MrZeroTree"):
        log("mcts_b/k: %d/%d, sample_b/k: %d/%d"%(interface.players[1].mcts_b,interface.players[1].mcts_k,
                                                  interface.players[1].sample_b,interface.players[1].sample_k))
    
    hands=read_std_hands(handsfile)
    N1=len(hands);N2=2
    log("(%s+%s) v.s. (%s+%s) for %dx%d on %s"%(interface.players[0].family_name(),interface.players[2].family_name(),
                                                interface.players[1].family_name(),interface.players[3].family_name(),
                                                N1,N2,device_bench))
    stats=[]
    for k,l in itertools.product(range(N1),range(N2)):
        if l==0:
            #cards=interface.shuffle()
            cards=hands[k]
            interface.shuffle(cards=cards)
        else:
            cards=cards[39:52]+cards[0:39]
            interface.shuffle(cards=cards)
        #log("%s: %s"%(interface.players[interface.pnext].family_name(),interface.players[interface.pnext].cards_list))
        for i,j in itertools.product(range(13),range(4)):
            """if complete_info and interface.players[interface.pnext].family_name().startswith("MrZeroTree"):
                interface.step_complete_info()
            else:"""
            interface.step()    
        #input("continue...")
        stats.append(interface.clear())
        interface.prepare_new()
        if l==N2-1:
            if print_process:
                log("No.%4d %4d %s"%(k,sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2,stats[-N2:]))
            else:
                print("%4d"%(sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])/N2),end=" ",flush=True)
        if (k+1)%(N1//4)==0 and l==N2-1:
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

if __name__ == '__main__':
    benchmark("StdHands/random_0_1024.hands")