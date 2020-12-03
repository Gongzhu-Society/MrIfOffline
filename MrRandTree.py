#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

from Util import log
from Util import ORDER_DICT2,SCORE_DICT
from MrRandom import MrRandom
from MrGreed import MrGreed
from ScenarioGen import ScenarioGen
from MCTS.mcts import mcts
import copy,itertools,numpy,time

class GameState():
    def __init__(self,cards_lists,fmt_scores,cards_on_table,play_for):
        self.cards_lists=cards_lists
        self.cards_on_table=cards_on_table
        self.fmt_scores=fmt_scores
        self.play_for=play_for

        #decide cards_dicts, suit and pnext
        self.cards_dicts=[MrGreed.gen_cards_dict(i) for i in self.cards_lists]
        if len(self.cards_on_table)==1:
            self.suit="A"
        else:
            self.suit=self.cards_on_table[1][0]
        self.pnext=(self.cards_on_table[0]+len(self.cards_on_table)-1)%4
        self.remain_card_num=sum([len(i) for i in self.cards_lists])

    def getCurrentPlayer(self):
        if (self.pnext-self.play_for)%2==0:
            return 1
        else:
            return -1

    def getPossibleActions(self):
        return MrGreed.gen_legal_choice(self.suit,self.cards_dicts[self.pnext],self.cards_lists[self.pnext])

    def takeAction(self,action):
        neo_state=copy.deepcopy(self)
        neo_state.cards_lists[neo_state.pnext].remove(action)
        neo_state.cards_dicts[neo_state.pnext][action[0]].remove(action)
        neo_state.remain_card_num-=1
        neo_state.cards_on_table.append(action)
        if (neo_state.cards_on_table)!=5:
            neo_state.pnext=(neo_state.pnext+1)%4
        else:
            #decide pnext
            score_temp=-1024
            for i in range(4):
                if neo_state.cards_on_table[i+1][0]==neo_state.cards_on_table[1][0] and ORDER_DICT2[neo_state.cards_on_table[i+1][1]]>score_temp:
                    winner=i #in relative order
                    score_temp=ORDER_DICT2[neo_state.cards_on_table[i+1][1]]
            neo_state.pnext=(neo_state.pnext+winner)%4
            #clear scores
            for c in neo_state.cards_on_table[1:]:
                if c not in SCORE_DICT:
                    continue
                neo_state.fmt_scores[neo_state.pnext][0]+=SCORE_DICT[c]
                if c=='C10':
                    neo_state.fmt_scores[neo_state.pnext][2]=True
                else:
                    neo_state.fmt_scores[neo_state.pnext][3]=True
                    if c[0]=='H':
                        neo_state.fmt_scores[neo_state.pnext][1]+=1
            #clean table
            neo_state.cards_on_table=[neo_state.pnext,]
            neo_state.suit='A'
        return neo_state

    def isTerminal(self):
        if self.remain_card_num==0:
            return True
        else:
            return False

    def getReward(self):
        scores=[MrRandTree.clear_fmt_score(self.fmt_scores[(self.play_for+i)%4]) for i in range(4)]
        return scores[0]+scores[2]-scores[1]-scores[3]

class MrRandTree(MrRandom):

    N_SAMPLE=5

    def clear_fmt_score(fmt_score):
        """
            [0,0,False,False] stands for [score, #hearts, C10 flag, has score flag]
        """
        s=fmt_score[0]
        if fmt_score[1]==13:
            s+=400
        if fmt_score[2]:
            if fmt_score[3]:
                s*=2
            else:
                assert fmt_score[0]==0 and fmt_score[1]==0
                s=50
        return s

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

        fmt_scores=MrGreed.gen_fmt_scores(self.scores)
        """ fmt_scores looks like [[0,0,False,False],[0,0,False,False],[0,0,False,False],[0,0,False,False]]
            fmt_scores is in absolute order"""
        d_legal={c:0 for c in MrGreed.gen_legal_choice(suit,cards_dict,self.cards_list)} #dict of legal choice
        sce_gen=ScenarioGen(self.place,self.history,self.cards_on_table,self.cards_list,number=MrRandTree.N_SAMPLE)
        for cards_list_list in sce_gen:
            cards_lists=[None,None,None,None]
            cards_lists[self.place]=copy.copy(self.cards_list)
            for i in range(3):
                cards_lists[(self.place+i+1)%4]=cards_list_list[i]
            cards_on_table_copy=copy.copy(self.cards_on_table)
            gamestate=GameState(cards_lists,fmt_scores,cards_on_table_copy,self.place)
            searcher=mcts(iterationLimit=20)
            action,value=searcher.search(initialState=gamestate)
            d_legal[action]+=value
        best_choice=MrGreed.pick_best_from_dlegal(d_legal)
        return best_choice

    @staticmethod
    def family_name():
        return 'MrRandTree'

def benchmark():
    from MrRandom import MrRandom,Human
    from MrIf import MrIf
    from OfflineInterface import OfflineInterface
    g=[MrGreed(room=0,place=i,name='greed%d'%(i)) for i in range(4)]
    f=[MrIf(room=0,place=i,name="if%d"%(i)) for i in range(4)]
    r=[MrRandom(room=0,place=i,name="random%d"%(i)) for i in range(4)]
    rt=[MrRandTree(room=0,place=i,name='randtree%d'%(i)) for i in range(4)]

    offlineinterface=OfflineInterface([rt[0],r[1],rt[2],r[3]],print_flag=False)
    N1=256;N2=2;stats=[]
    log("%s vs. %s for %dx%d"%(offlineinterface.players[0].family_name(),offlineinterface.players[1].family_name(),N1,N2))
    tik=time.time()
    for k,l in itertools.product(range(N1),range(N2)):
        if l==0:
            cards=offlineinterface.shuffle()
        else:
            cards=cards[39:52]+cards[0:39]
            offlineinterface.shuffle(cards=cards)
        for i,j in itertools.product(range(13),range(4)):
            offlineinterface.step()
        stats.append(offlineinterface.clear())
        offlineinterface.prepare_new()
        if l==N2-1:
            print("%4d"%(sum([j[0]+j[2]-j[1]-j[3] for j in stats[-N2:]])),end=" ",flush=True)
    tok=time.time()
    log("time consume: %ds"%(tok-tik))
    s_temp=[j[0]+j[2]-j[1]-j[3] for j in stats]
    log("%.2f %.2f"%(numpy.mean(s_temp),numpy.sqrt(numpy.var(s_temp)/(len(s_temp)-1))))

if __name__=="__main__":
	benchmark()