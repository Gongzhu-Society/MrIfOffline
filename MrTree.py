#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from MrIf import LOGFILE,log,MrRandom,MrIf

class MrBasic():
    def __init__(self,cards_list,cards_of_others):
        self.cards_list=cards_list
        self.cards_dict={"S":[],"H":[],"D":[],"C":[]}
        for i in self.cards_list:
            self.cards_dict[i[0]].append(i)
        self.cards_of_others=cards_of_others #[str,str,...]

    def 
class Scenario():
    def __init__(self,list_basic,pnext,cards_on_table,scores,deep):
        self.list_basic=list_basic
        self.pnext=pnext
        self.cards_on_table=cards_on_table
        self.scores=scores
        self.deep=deep #number of cards unplayed
    
    def step():
        choice

