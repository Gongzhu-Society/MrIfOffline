#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import ORDER_DICT2,INIT_CARDS,SCORE_DICT
from MrGreed import MrGreed
import copy,random,numpy

print_level=1

class MrTree(MrGreed):
    N_SAMPLE=5

    def pick_a_card(self):
        assert (self.cards_on_table[0]+len(self.cards_on_table)-1)%4==self.place,"self.place and self.cards_on_table contrdict"
        global print_level
        if print_level>=1:
            log("my turn %s %s"%(self.cards_on_table,self.cards_list))
        suit=self.decide_suit()
        cards_dict=MrTree.gen_cards_dict(self.cards_list)

        #如果别无选择
        if cards_dict.get(suit)!=None and len(cards_dict[suit])==1:
            choice=cards_dict[suit][0]
            if print_level>=1:
                log("I have no choice but %s"%(choice))
            return choice

        cards_remain=MrTree.calc_cards_remain(self.history,self.cards_on_table,self.cards_list)
        void_info=MrTree.gen_void_info(self.place,self.history,self.cards_on_table)
        fmt_score_l=MrTree.gen_fmt_scores(self.scores)
        if len(self.cards_on_table)==4:
            lens=[len(self.cards_list)-1,len(self.cards_list)-1,len(self.cards_list)-1]
        elif len(self.cards_on_table)==3:
            lens=[len(self.cards_list),len(self.cards_list)-1,len(self.cards_list)-1]
        elif len(self.cards_on_table)==2:
            lens=[len(self.cards_list),len(self.cards_list),len(self.cards_list)-1]
        elif len(self.cards_on_table)==1:
            lens=[len(self.cards_list),len(self.cards_list),len(self.cards_list)]
        d_legal={}
        for c in MrTree.gen_legal_choice(suit,cards_dict,self.cards_list):
            d_legal[c]=0
        if self.place%2==0:
            reverse_flag=False
        else:
            reverse_flag=True
        expire_date=0
        for ax in range(MrTree.N_SAMPLE):
            if expire_date==0:
                cards_list_list,exchange_info,bx=MrTree.gen_scenario(void_info,cards_remain,lens)
                expire_date=max(bx-5,0)
            else:
                exhausted_flag=MrTree.alter_scenario(cards_list_list,exchange_info,void_info)
                if exhausted_flag==1:
                    break
            cards_ll=[None,None,None,None]
            cards_ll[self.place]=self.cards_list
            for i in range(3):
                cards_list_list[i].sort(key=cards_order)
                cards_ll[(self.place+i+1)%4]=cards_list_list[i]
            #log("gen scenario: %s"%(cards_ll))
            """
            four_cards=['','','','']
            four_cards[0:len(self.cards_on_table)-1]=self.cards_on_table[1:]
            number=len(self.cards_on_table)-1
            #log("four_cards: %s"%(four_cards))
            #log("cards_ll(before): %s"%(cards_ll))
            Leaf.greed_play_a_trick(four_cards,cards_ll,self.place,number)
            #log("four_cards: %s"%(four_cards))
            d_legal[four_cards[number]]+=1
            """
            root=Leaf.produce_1(self.cards_on_table[1:],cards_ll,self.place,fmt_score_l)
            root.search(iters=100)
            if reverse_flag:
                for d in root.descendants:
                    #log("if I chose %s: %.2f"%(d.last_action,-1*d.mcts_w/d.mcts_n))
                    d_legal[d.last_action]-=d.mcts_w
            else:
                for d in root.descendants:
                    #log("if I chose %s: %.2f"%(d.last_action,d.mcts_w/d.mcts_n))
                    d_legal[d.last_action]+=d.mcts_w
        #log(d_legal)
        best_choice=MrTree.pick_best_from_dlegal(d_legal)
        return best_choice

    @staticmethod
    def family_name():
        return 'MrTree'

class Leaf():
    """蒙特卡洛树上的一个节点，储存了一个假想的牌局的信息，一些子节点，还有MCTS的统计信息"""
    VALUE_DICT={'SQ':-100,'DJ':100,'C10':-50,
            'H2':0,'H3':0,'H4':0,'H5':-10,'H6':-10,'H7':-10,'H8':-10,'H9':-10,'H10':-10,
            'HJ':-20,'HQ':-30,'HK':-40,'HA':-50,'JP':-60,'JG':-70}
    BURDEN_DICT={'SA':40,'SK':30,'SJ':7,'S10':6,'S9':5,'S8':4,'S7':3,'S6':2,'S5':1,'S4':1,
                 'DA':-10,'DK':-5,'DQ':-5,'D10':6,'D9':5,'D8':4,'D7':3,'D6':2,'D5':1,'D4':1,
                 'CA':20,'CK':15,'CQ':10,'CJ':10,'C9':5,'C8':4,'C7':3,'C6':2,'C5':1,'C4':1,
                 'H10':6,'H9':5,'H8':4,'H7':3,'H6':2,'H5':1,'H4':1}

    def __init__(self):
        """cards_on_table 就真的只是四张牌了
           cards_ll 是四个人的牌，按座次排好
           要最大化还是最小化看pnext就知道了
           last_action 是便于标记好返回"""
        self.cards_ll=None
        self.fmt_score_l=None

        self.cards_on_table=None
        self.pnext=0

        self.last_action=''
        self.descendants=[]
        self.mcts_w=0
        self.mcts_n=0
        self.end_flag=False

    def produce_1(cards_on_table,cards_ll,pnext,fmt_score_l):
        l=Leaf()
        l.cards_ll=cards_ll
        l.fmt_score_l=fmt_score_l
        l.cards_on_table=cards_on_table
        l.pnext=pnext
        l.last_action='created by product_1'
        return l

    def gen_legal_choice(cards_on_table,cards_list):
        l_tar=[]
        if len(cards_on_table)!=0:
            for c in cards_list:
                if c[0]==cards_on_table[0][0]:
                    l_tar.append(c)
            if len(l_tar)>0:
                return l_tar
            else:
                return cards_list
        else:
            return cards_list

    def judge_winner(four_cards):
        winner=0
        score_temp=ORDER_DICT2[four_cards[0][1]]
        if four_cards[1][0]==four_cards[0][0]\
        and ORDER_DICT2[four_cards[1][1]]>score_temp:
            winner=1
            score_temp=ORDER_DICT2[four_cards[1][1]]
        if four_cards[2][0]==four_cards[0][0]\
        and ORDER_DICT2[four_cards[2][1]]>score_temp:
            winner=2
            score_temp=ORDER_DICT2[four_cards[2][1]]
        if four_cards[3][0]==four_cards[0][0]\
        and ORDER_DICT2[four_cards[3][1]]>score_temp:
            winner=3
        return winner

    def clear_trick(four_cards):
        winner=Leaf.judge_winner(four_cards)
        delta_score=0
        for c in four_cards:
            delta_score+=Leaf.VALUE_DICT.get(c,0)
        if winner%2==0:
            delta_score*=-1
        return delta_score

    def greed_play_a_trick(four_cards,cards_ll,pnext,number):
        best_score=-65535
        four_cards_tmp=copy.copy(four_cards)
        for c in Leaf.gen_legal_choice(four_cards[0:number],cards_ll[pnext]):
            four_cards_tmp[number]=c
            if number<3:
                Leaf.greed_play_a_trick(four_cards_tmp,cards_ll,(pnext+1)%4,number+1)
            score_temp=Leaf.clear_trick(four_cards_tmp)
            if number%2==0:
                score_temp*=-1
            score_temp+=Leaf.BURDEN_DICT.get(c,0)
            if score_temp>best_score:
                four_cards[number:]=four_cards_tmp[number:]
                best_score=score_temp
        #log("I got %d by playing %s"%(best_score,four_cards[0]))
        assert len(four_cards)==4

    def calc_a_score(fmt_score):
        s=fmt_score[0]
        if fmt_score[1]==13:
            s+=400
        if fmt_score[2]:
            if fmt_score[3]:
                s*=2
            else:
                assert s==0
                s=50
        return s

    def greed_play(self):
        global print_level
        cards_ll_tmp=copy.deepcopy(self.cards_ll)
        fmt_score_l_tmp=copy.deepcopy(self.fmt_score_l)
        four_cards=['','','','']
        four_cards[0:len(self.cards_on_table)]=self.cards_on_table
        pnext_tmp=self.pnext
        number=len(self.cards_on_table)
        while len(cards_ll_tmp[self.pnext])>0:
            Leaf.greed_play_a_trick(four_cards,cards_ll_tmp,pnext_tmp,number)
            #log(four_cards)
            winner=Leaf.judge_winner(four_cards)
            winner_abs=(pnext_tmp+winner-number)%4
            for c in four_cards:
                fmt_score_l_tmp[winner_abs][0]+=SCORE_DICT.get(c,0)
                if c in SCORE_DICT:
                    if c=='C10':
                        fmt_score_l_tmp[winner_abs][2]=True
                    else:
                        fmt_score_l_tmp[winner_abs][3]=True
                        if c[0]=='H':
                            fmt_score_l_tmp[winner_abs][1]+=1
            #log("winner: %d, %d, %s"%(winner,winner_abs,fmt_score_l_tmp))
            for i,c in enumerate(four_cards[number:]):
                cards_ll_tmp[(i+pnext_tmp)%4].remove(c)
            four_cards=['','','','']
            pnext_tmp=winner_abs
            number=0
        s_l=[Leaf.calc_a_score(i) for i in fmt_score_l_tmp]
        #log(s_l)
        return s_l[0]+s_l[2]-s_l[1]-s_l[3]

    def update_fmt_score_l(cards_on_table,fmt_score_l,winner):
        for c in cards_on_table:
            fmt_score_l[winner][0]+=SCORE_DICT.get(c,0)
            if c in SCORE_DICT:
                if c=='C10':
                    fmt_score_l[winner][2]=True
                else:
                    fmt_score_l[winner][3]=True
                    if c[0]=='H':
                        fmt_score_l[winner][1]+=1

    def give_birth(self,choice_num):
        d_temp=Leaf()
        d_temp.cards_ll=copy.deepcopy(self.cards_ll)
        d_temp.fmt_score_l=copy.deepcopy(self.fmt_score_l)
        choice=d_temp.cards_ll[self.pnext].pop(choice_num)
        d_temp.last_action=choice
        self.descendants.append(d_temp)
        
        #下面初始化pnext和cards_on_table
        #如果这个儿子到结算的时候了
        if len(self.cards_on_table)==3:
            winner=Leaf.judge_winner(self.cards_on_table+[choice,])
            d_temp.pnext=(winner+1+self.pnext)%4
            Leaf.update_fmt_score_l(self.cards_on_table+[choice,],d_temp.fmt_score_l,d_temp.pnext)
            d_temp.cards_on_table=[]
        else:
            d_temp.pnext=(self.pnext+1)%4
            d_temp.cards_on_table=copy.copy(self.cards_on_table)
            d_temp.cards_on_table.append(choice)
        #log("gave birth to %s"%(d_temp.last_action))

    def breed(self):
        no_choice_flag=True
        if len(self.cards_on_table)!=0:
            for i,c in enumerate(self.cards_ll[self.pnext]):
                if c[0]==self.cards_on_table[0][0]:
                    no_choice_flag=False
                    self.give_birth(i)
            if no_choice_flag:
                for i in range(len(self.cards_ll[self.pnext])):
                    self.give_birth(i)
        else:
            for i in range(len(self.cards_ll[self.pnext])):
                self.give_birth(i)

    def uct(self,lnt,max_flag):
        if max_flag:
            return self.mcts_w+200*numpy.sqrt(lnt/self.mcts_n)
        else:
            return -1*self.mcts_w+200*numpy.sqrt(lnt/self.mcts_n)

    def get_best_child(self):
        lnt=numpy.log(self.mcts_n)
        if self.pnext%2==0:
            max_flag=True
        else:
            max_flag=False
        d_best=self.descendants[0]
        d_best_val=self.descendants[0].uct(lnt,max_flag)
        for d in self.descendants[1:]:
            temp_val=d.uct(lnt,max_flag)
            if temp_val>d_best_val:
                d_best=d
                d_best_val=temp_val
        #log("I chose %s"%(d_best.last_action))
        return d_best

    def step(self):
        if len(self.descendants)==0:
            if not self.end_flag:
                self.breed()
                if len(self.descendants)==0:
                    assert sum([len(i) for i in self.cards_ll])==0
                    self.end_flag=True
                    s_l=[Leaf.calc_a_score(i) for i in self.fmt_score_l]
                    self.mcts_w=s_l[0]+s_l[2]-s_l[1]-s_l[3]
                    self.mcts_n=1
                    #log("I play myself: %s, %d, %d"%(self.last_action,self.mcts_w,self.mcts_n))
                    return 1
                else:
                    dn=0
                    for d in self.descendants:
                        d.mcts_w=d.greed_play()
                        d.mcts_n=1
                        dn+=1
                        #log("greed play: %s, %d, %d"%(d.last_action,d.mcts_w,d.mcts_n))
                    if self.pnext%2==0:
                        self.mcts_w=max([d.mcts_w for d in self.descendants])
                    else:
                        self.mcts_w=min([d.mcts_w for d in self.descendants])
                    self.mcts_n+=dn
                    return dn
            else:
                return 1
        else:
            best_child=self.get_best_child()
            dn=best_child.step()
            if self.pnext%2==0:
                self.mcts_w=max([d.mcts_w for d in self.descendants])
            else:
                self.mcts_w=min([d.mcts_w for d in self.descendants])
            self.mcts_n+=dn
            return dn

    def search(self,iters=100):
        for i in range(iters):
            self.step()

    def __str__(self):
        return str(self.__dict__)

def test_breed():
    cards_ll=[['D2','C3'],['D4','SQ'],['D5','S2'],['HA','H5','C2']]
    for i in cards_ll:
        i.sort(key=cards_order)
    log(cards_ll)
    fmt_score_l=[[0,0,False,False],[0,0,False,False],[0,0,False,False],[0,0,False,False]]
    s=Leaf.produce_1(['H8','H9','H2'],cards_ll,3,fmt_score_l)
    for i in range(50):
        s.step()
        for d in s.descendants:
            log("%s, %d, %d"%(d.last_action,d.mcts_w,d.mcts_n))
        log("%s, %d, %d"%(s.last_action,s.mcts_w,s.mcts_n))
        input()

def test_greed_play():
    cards=copy.copy(INIT_CARDS)
    cards.remove('S2')
    cards.remove('S3')
    random.shuffle(cards)
    cards_ll=[cards[0:12],cards[12:24],cards[24:37],cards[37:50]]
    for i in cards_ll:
        i.sort(key=cards_order)
    log(cards_ll)
    fmt_score_l=[[0,0,False,False],[0,0,False,False],[0,0,False,False],[0,0,False,False]]
    s=Leaf.produce_1(['S2','S3'],cards_ll,2,fmt_score_l)
    log(s.greed_play())

if __name__=="__main__":
    test_breed()
    #test_greed_play()

"""
全展开情况笔记，从52开始，随机发牌
展开层数 1   2   3   4   5    8      9       12
情况数   13  53  159 466 4042 170689 1564813
"""