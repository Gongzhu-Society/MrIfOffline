#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from MrIf import LOGFILE,log,cards_order,MrRandom,Human,MrIf
from OfflineInterface import OfflineInterface

if __name__=="__main__":
    #创建一系列机器人对象
    #机器人对象应当有如下方法
    #receive_shuffle(self,cards) 接收洗牌
    #pick_a_card(self,suit) 出哪张牌，但不更新数据结构，要等到offlineinterface调用机器人的pop_card才更新
    #pop_card(self,which) 确认手牌打出后会被调用，更新手牌的数据结构
    #但是offlineinterface还没有把历史信息传给机器人的功能，请自定义函数
    random0=MrRandom(0,0,"random0")
    random1=MrRandom(0,1,"random1")
    random2=MrRandom(0,2,"random2")
    random3=MrRandom(0,3,"random3")
    if0=MrIf(0,0,"if0")
    if1=MrIf(0,1,"if1")
    if2=MrIf(0,2,"if2")
    if3=MrIf(0,2,"if3")
    #使用四个机器人初始化OfflineInterface（机器人也可以是Human，这样人就可以加入）
    offlineinterface=OfflineInterface([if0,random1,if2,random3])
    #发牌
    offlineinterface.shuffle()
    #或者指定发什么牌然后发牌
    #offlineinterface.shuffle(cards=cards)
    #打54张牌
    for i in range(52):
        offlineinterface.step()
    #打印分数
    log(offlineinterface.clear())
    #准备重新开始
    offlineinterface.prepare_new()