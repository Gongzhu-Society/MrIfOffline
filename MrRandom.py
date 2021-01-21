#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log
import random

class Robot():
    def __init__(self,room,place,name,create_room = False):
        #useful infos
        self.place = place
        self.cards_list = []
        self.history = []
        self.cards_on_table = []
        self.scores = [[],[],[],[]] #absolute order

        #useless but still be set infos
        self.room = room
        self.name = name
        self.game_mode = 4

        #useless and not be set infos
        self.players_information = [None, None, None, None]
        self.initial_cards = []

        #things should not realize by Robot class
        self.scores_num = [0,0,0,0]
        self.state = 'logout'
        self.creator = create_room
        self.master = 'MrComputer'

        #things I even do not know what it is
        self.res = [] #?

    def pick_a_card(self):
        pass

    def __str__(self):
        return 'name:{}, state:{}'.format(self.name,self.state)

    def shuffle(self):
        pass

    def update(self):
        pass

    def trickend(self):
        pass

    def gameend(self):
        # self.players_information looks like [['Sun', True, False], ['Miss.if0', True, True], ['Miss.if1', True, True], ['Miss.if2', True, True]] #?
        self.res.append(self.scores_num[self.place])
        should_record = True
        for i in range(self.place):
            if self.players_information[i][2]:
                should_record = False
                break
        if should_record:
            log("I, %s, should record."%(self.name))
            s = "\n".join([", ".join([str(i) for i in trick]) for trick in self.history])
            s += '\nresult: %s\n\n'%(", ".join([str(n) for n in self.scores_num]))
            fname = [pl[0] for pl in self.players_information]
            fname = "Records/" + "_".join(fname) + ".txt"
            log("writing to %s:\n%s"%(fname,s))
            with open(fname, 'a') as f:
                f.write(s)

class MrRandom(Robot):
    """
        ONLY for 4 players
        See OfflineInterface.step to get a feel of how to use this
    """
    def __init__(self,room,place,name,create_room=False):
        Robot.__init__(self,room,place,name,create_room=create_room)
        #self.place=place
        #self.cards_list=[]        #cards in hand
        #self.history=[]           #list of (int,str,str,str,str)
        #self.cards_on_table=[]    #[int,str,...]
        #self.scores=[[],[],[],[]]  #绝对坐次

        #useless but still be set infos
        #self.room=room
        #self.name=name

        #useless and not be set infos
        #self.initial_cards=[]
        #self.players_information=[None,None,None,None]

    def decide_suit(self):
        if len(self.cards_on_table)==1:
            suit="A"
        else:
            suit=self.cards_on_table[1][0]
        return suit

    def gen_cards_dict(self):
        """
            will be used in self.pick_a_card
        """
        cards_dict={"S":[],"H":[],"D":[],"C":[]}
        for i in self.cards_list:
            cards_dict[i[0]].append(i)
        return cards_dict

    def pick_a_card(self):
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place,"self.place and self.cards_on_table contrdict"
        suit=self.decide_suit()
        cards_dict=self.gen_cards_dict()
        if cards_dict.get(suit)==None or len(cards_dict[suit])==0:
            i=random.randint(0,len(self.cards_list)-1)
            choice=self.cards_list[i]
        else:
            i=random.randint(0,len(cards_dict[suit])-1)
            choice=cards_dict[suit][i]
        return choice

    @staticmethod
    def family_name():
        return 'MrRandom'

class Human(MrRandom):
    def pick_a_card(self,suit=None):
        suit=self.decide_suit()
        log("%s, %s, %s, %s"%(self.name,suit,self.cards_on_table,self.cards_list))
        while True:
            choice=input("your turn: ")
            if choice in self.cards_list:
                break
            else:
                log("%s is not your cards. "%(choice),end="")
        return choice

    @staticmethod
    def family_name():
        return 'Human'

if __name__=="__main__":
    log("",l=2)