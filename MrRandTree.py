#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from Util import log
from MrRandom import MrRandom
from MrGreed import MrGreed
from ScenarioGen import ScenarioGen
from MCTS.mcts import mcts
import copy


class GameState():
	def __init__(self,):
		pass

	def getCurrentPlayer(self):

	def getPossibleActions(self):

	def takeAction(self, action):

	def isTerminal(self):

	def getReward(self):

class MrRandTree(MrRandom):

    N_SAMPLE=5

    def pick_a_card(self):
        #确认桌上牌的数量和自己坐的位置相符
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place
        #utility datas
        suit=self.decide_suit() #inherited from MrRandom
        cards_dict=MrGreed.gen_cards_dict(self.cards_list)
        #如果别无选择
        if cards_dict.get(suit)!=None and len(cards_dict[suit])==1:
            choice=cards_dict[suit][0]
            return choice
        #fmt_score_list looks like [[0,0,False,False],[0,0,False,False],[0,0,False,False],[0,0,False,False]]
        #fmt_score_list is in absolute order
        fmt_score_list=MrGreed.gen_fmt_scores(self.scores)
        sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=MrRandTree.N_SAMPLE)
        for cards_list_list in sce_gen:
            gamestate=GameState(self.cards_list,cards_list_list,self.cards_on_table,fmt_score_list)

    @staticmethod
    def family_name():
        return 'MrRandTree'

if __name__=="__main__":
	log(mcts)