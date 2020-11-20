#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import INIT_CARDS
import random,copy

class NumTable():
    """
      A B C
    S 0 1 2
    H 3 4 5
    D 6 7 8
    C 9 x x
    """
    def __init__(self,vals,suit_ct,cnum_ct,re_list,depth=0):
        self.vals=vals
        self.suit_ct=suit_ct
        self.cnum_ct=cnum_ct
        self.re_list=re_list
        self.depth=depth
        #self.children=[] #for breadth first

    def breed(self):
        #breed condition
        if self.vals[self.depth]>=0:
            #log("shortcut 1: %d, %s, %s"%(self.depth,self.vals,self.maxs)); input()
            self.depth+=1
            self.breed()
        else:
            col_num=self.depth%3 #column number
            row_num=self.depth//3
            for i in range(0,min(self.cnum_ct[col_num],self.suit_ct[row_num])+1):
                neo_vals=copy.copy(self.vals)
                neo_vals[self.depth]=i
                sum_col=sum((neo_vals[3*k+col_num] for k in range(0,row_num+1)))
                if row_num==3:
                    if sum_col!=self.cnum_ct[col_num]:
                        continue
                else:
                    if sum_col>self.cnum_ct[col_num]:
                        continue
                sum_row=sum((neo_vals[3*row_num+j] for j in range(0,col_num+1)))
                if col_num==2:
                    if sum_row!=self.suit_ct[row_num]:
                        continue
                else:
                    if sum_row>self.suit_ct[row_num]:
                        continue
                #depth first
                if self.depth==11:
                    self.re_list.append(neo_vals)
                else:
                    child=NumTable(neo_vals,self.suit_ct,self.cnum_ct,self.re_list,depth=self.depth+1)
                    child.breed()


class ScenarioGen():
    def __init__(self,myseat,history,cards_on_table,cards_list,number=20,method=None):
        """
            myseat, history and cards_on_table will be fed to gen_void_info to get void_info
            history, cards_on_table and cards_list will be fed to gen_cards_remain to get cards_remain
            cards_list: cards in my hand
            number: the max number of sampling
            method: 0 for "shot and test", 1 for
        """

        #void_info and cards_remain are all in relative order!
        self.void_info=ScenarioGen.gen_void_info(myseat,history,cards_on_table)
        self.cards_remain=ScenarioGen.gen_cards_remain(history,cards_on_table,cards_list)
        #确认别人手里牌的数量和我手里的还有桌上牌的数量相符
        assert len(self.cards_remain)==3*len(cards_list)-(len(cards_on_table)-1)

        #lens is like [[13],[12],[12]], also in relative order like void_info and cards_remain
        if len(cards_on_table)==1: #If I am the first
            self.lens=[len(cards_list),2*len(cards_list),3*len(cards_list)]
        elif len(cards_on_table)==2: #I am the second one
            self.lens=[len(cards_list),2*len(cards_list),3*len(cards_list)-1]
        elif len(cards_on_table)==3: #I am the third one
            self.lens=[len(cards_list),2*len(cards_list)-1,3*len(cards_list)-2]
        elif len(cards_on_table)==4: #I am the last one
            self.lens=[len(cards_list)-1,2*len(cards_list)-2,3*len(cards_list)-3]
        assert self.lens[2]==len(self.cards_remain)

        self.void_num=sum([sum(i.values()) for i in self.void_info])
        if method==None:
            if (self.lens[-1]/(12-self.void_num))**(-self.void_num)>0.5:
                self.method=0
            else:
                print(self.lens[-1],self.void_num)
                self.gen_num_table_cases()
                self.method=1
        self.number=number
        self.suc_ct=0 #success counter
        self.tot_ct=0 #total try counter

    def __iter__(self):
        return self

    def __next__(self):
        if self.suc_ct>=self.number:
            #if self.suc_ct/self.tot_ct<0.2:
            #    print("%5.2f|%d|%2d"%(self.suc_ct*100/self.tot_ct,self.void_num,self.lens[-1]))
            raise StopIteration
        if self.method==0:
            return self.shot_and_test()
        elif self.method==1:
            pass

    def gen_cards_remain(history,cards_on_table,cards_list):
        cards_remain=set(INIT_CARDS)
        for h in history:
            for c in h[1:5]:
                cards_remain.remove(c)
        for c in cards_on_table[1:]:
            cards_remain.remove(c)
        for c in cards_list:
            cards_remain.remove(c)
        return list(cards_remain)

    def gen_void_info(myseat,history,cards_on_table):
        """
            generate void info in __relative order__, True means is empty
            will be called in pick_a_card and used as the input for check_void_info in gen_scenario
        """
        void_info=[{'S':False,'H':False,'D':False,'C':False},{'S':False,'H':False,'D':False,'C':False},{'S':False,'H':False,'D':False,'C':False}]
        for h in history:
            for i,c in enumerate(h[2:5]):
                seat=(h[0]+i-myseat)%4
                if seat!=3 and c[0]!=h[1][0]:
                    void_info[seat][h[1][0]]=True
        for i,c in enumerate(cards_on_table[2:]):
            seat=(cards_on_table[0]+i-myseat)%4
            if seat!=3 and c[0]!=cards_on_table[1][0]:
                void_info[seat][cards_on_table[1][0]]=True
        return void_info

    def check_void_legal(cards_list1,cards_list2,cards_list3,void_info):
        """
            check the senario generated agree with void info or not
            will be called by gen_scenario, asserted in alter_scenario
        """
        s_temp=''.join((i[0] for i in cards_list1))
        if void_info[0]['S'] and 'S' in s_temp:
            return False
        elif void_info[0]['H'] and 'H' in s_temp:
            return False
        elif void_info[0]['D'] and 'D' in s_temp:
            return False
        elif void_info[0]['C'] and 'C' in s_temp:
            return False
        s_temp=''.join((i[0] for i in cards_list2))
        if void_info[1]['S'] and 'S' in s_temp:
            return False
        elif void_info[1]['H'] and 'H' in s_temp:
            return False
        elif void_info[1]['D'] and 'D' in s_temp:
            return False
        elif void_info[1]['C'] and 'C' in s_temp:
            return False
        s_temp=''.join((i[0] for i in cards_list3))
        if void_info[2]['S'] and 'S' in s_temp:
            return False
        elif void_info[2]['H'] and 'H' in s_temp:
            return False
        elif void_info[2]['D'] and 'D' in s_temp:
            return False
        elif void_info[2]['C'] and 'C' in s_temp:
            return False
        return True

    def shot_and_test(self):
        while True:
            self.tot_ct+=1
            random.shuffle(self.cards_remain)
            cards_list_list=[self.cards_remain[0:self.lens[0]],
                             self.cards_remain[self.lens[0]:self.lens[1]],
                             self.cards_remain[self.lens[1]:self.lens[2]]]
            if ScenarioGen.check_void_legal(cards_list_list[0],cards_list_list[1],cards_list_list[2],self.void_info):
                break
        self.suc_ct+=1
        return cards_list_list

    def gen_num_tables(self):
        """
            generate all possible, i.e. agree with the restrictions, tables.
        """
        s_temp=''.join((i[0] for i in self.cards_remain))
        suit_ct=[s_temp.count(s) for s in "SHDC"] #suit remain number count
        cnum_ct=[self.lens[0],self.lens[1]-self.lens[0],self.lens[2]-self.lens[1]] #remained card number count
        assert sum(suit_ct)==len(self.cards_remain)
        assert sum(cnum_ct)==len(self.cards_remain)
        val_dict={True:0,False:-1}
        vals=[val_dict[self.void_info[0]['S']],val_dict[self.void_info[1]['S']],val_dict[self.void_info[2]['S']],
              val_dict[self.void_info[0]['H']],val_dict[self.void_info[1]['H']],val_dict[self.void_info[2]['H']],
              val_dict[self.void_info[0]['D']],val_dict[self.void_info[1]['D']],val_dict[self.void_info[2]['D']],
              val_dict[self.void_info[0]['C']],val_dict[self.void_info[1]['C']],val_dict[self.void_info[2]['C']],]
        log(suit_ct)
        log(cnum_ct)
        re_list=[]
        nt_root=NumTable(vals,suit_ct,cnum_ct,re_list)
        nt_root.breed()
        print(re_list)
        print(len(re_list))

def test_gen_num_tables():
    s=ScenarioGen(0,[[0,'S2','S3','S4','S5'],[0,'S6','S7','S8','S9'],[0,'S10','SJ','SQ','SK'],
                     [0,'H2','H3','H4','H5'],[0,'H6','H7','H8','H9'],[0,'H10','HJ','HQ','HK'],
                     [0,'C2','C3','C4','C5'],[0,'C6','C7','C8','C9'],[0,'C10','CJ','CQ','CK'],]
                    ,[0,'SA','HA','CA'],['D2','D3','D4','D5'])
    s.gen_num_tables()

if __name__=="__main__":
    test_gen_num_tables()
