#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
import random,math,numpy

graph={0:(1,2,3,4),1:(0,2,3),2:(0,1),3:(0,1),4:(0,)}
possi=[0]*5
n_mc=100000

def f0(start):
    """simple random walk, has bias"""
    now=start
    for i in range(n_mc):
        possi[now]+=1
        now=random.choice(graph[now])
    return possi

def f1(start):
    """random walk with no go back, has bias"""
    global possi
    now=start
    last=-1
    for i in range(n_mc):
        possi[now]+=1
        now_neo=random.choice(graph[now])
        while now_neo==last:
            now_neo=random.choice(graph[now])
        last=now
        now=now_neo
    return possi

def f2(start):
    global possi
    w_dict={4:0.05,3:0.1,2:0.3,1:1}
    now=start
    for i in range(n_mc):
        possi[now]+=1
        weight=[w_dict[len(graph[j])] for j in graph[now]]
        weight_sum=sum(weight)
        weight=[j/weight_sum for j in weight]
        now=numpy.random.choice(graph[now],p=weight)
    return possi

if __name__=="__main__":
    possi=f2(0)

    possi=[i/n_mc for i in possi]
    error=[math.sqrt(i*(1-i)/n_mc) for i in possi]
    possi=["%.4f"%(i) for i in possi]
    error=["%.4f"%(i) for i in error]
    log(possi)
    log(error)

