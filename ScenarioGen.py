#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,cards_order
from Util import INIT_CARDS
import random,copy,numpy

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

    def check(self,col_num,row_num,neo_vals):
        """
           check this new (col,row) number
           return False for not pass
        """
        sum_col=sum((neo_vals[3*k+col_num] for k in range(0,row_num+1)))
        if row_num==3:
            if sum_col!=self.cnum_ct[col_num]:
                return False
        else:
            if sum_col>self.cnum_ct[col_num]:
                return False
        sum_row=sum((neo_vals[3*row_num+j] for j in range(0,col_num+1)))
        if col_num==2:
            if sum_row!=self.suit_ct[row_num]:
                return False
        else:
            if sum_row>self.suit_ct[row_num]:
                return False
        return True

    def breed(self):
        col_num=self.depth%3 #column number
        row_num=self.depth//3
        if self.vals[self.depth]>=0:
            if not self.check(col_num,row_num,self.vals):
                return
            if self.depth==11:
                self.re_list.append(self.vals)
            else:
                self.depth+=1
                self.breed()
        else:
            for i in range(0,min(self.cnum_ct[col_num],self.suit_ct[row_num])+1):
                neo_vals=copy.copy(self.vals)
                neo_vals[self.depth]=i
                if not self.check(col_num,row_num,neo_vals):
                    continue
                #depth first
                if self.depth==11:
                    self.re_list.append(neo_vals)
                else:
                    child=NumTable(neo_vals,self.suit_ct,self.cnum_ct,self.re_list,depth=self.depth+1)
                    child.breed()

class ScenarioGen():

    FACTORIAL={0:1,1:1,2:2,3:6,4:24,5:120,6:720,7:5040,8:40320,9:362880,10:3628800,11:39916800,12:479001600,13:6227020800}

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
        #log(self.void_info)

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

        s_temp=''.join((i[0] for i in self.cards_remain))
        self.suit_ct=[s_temp.count(s) for s in "SHDC"] #suit remain number count
        self.reinforce_void_info()
        #log(self.void_info)

        self.number=number
        #predict success rate
        d1={True:0,False:1}
        s_cell_num=sum([d1[self.void_info[j]['S']] for j in range(3)])
        h_cell_num=sum([d1[self.void_info[j]['H']] for j in range(3)])
        d_cell_num=sum([d1[self.void_info[j]['D']] for j in range(3)])
        c_cell_num=sum([d1[self.void_info[j]['C']] for j in range(3)])
        self.suc_rate_predict=(s_cell_num/3)**(self.suit_ct[0])
        self.suc_rate_predict*=(h_cell_num/3)**(self.suit_ct[1])
        self.suc_rate_predict*=(d_cell_num/3)**(self.suit_ct[2])
        self.suc_rate_predict*=(c_cell_num/3)**(self.suit_ct[3])
        """def C(m,n):
            if n==3:
                return (m+2)*(m+1)/2
            elif n==2:
                return m+1
            else:
                return 1
        self.tables_num_predict=C(self.suit_ct[0],s_cell_num)
        self.tables_num_predict*=C(self.suit_ct[1],h_cell_num)
        self.tables_num_predict*=C(self.suit_ct[2],d_cell_num)
        self.tables_num_predict*=C(self.suit_ct[3],c_cell_num)"""
        if method==None:
            if self.suc_rate_predict<0.05:
                self.gen_num_tables()
                self.method=1
            else:
                self.tot_ct=0 #total try counter
                self.method=0
        self.suc_ct=0 #success counter

    def __iter__(self):
        return self

    def __next__(self):
        if self.suc_ct>=self.number:
            """if self.method==0 and self.suc_ct/self.tot_ct<0.15:
                print("suc rate: %5.2f(%d), %5.2f"%(self.suc_ct*100/self.tot_ct,self.number,self.suc_rate_predict*100))
                self.gen_num_tables()
                print("tables_num: %d, %d"%(len(self.num_tables),self.tables_num_predict))
                input()"""
            raise StopIteration
        if self.method==0:
            result=self.shot_and_test()
            self.suc_ct+=1
            return result
        elif self.method==1:
            result=self.suit_by_suit()
            self.suc_ct+=1
            return result

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

    def reinforce_void_info(self):
        """use the information in self.suit_ct to reinforce self.void_info"""
        if self.suit_ct[0]==0:
            self.void_info[0]['S']=True
            self.void_info[1]['S']=True
            self.void_info[2]['S']=True
        if self.suit_ct[1]==0:
            self.void_info[0]['H']=True
            self.void_info[1]['H']=True
            self.void_info[2]['H']=True
        if self.suit_ct[2]==0:
            self.void_info[0]['D']=True
            self.void_info[1]['D']=True
            self.void_info[2]['D']=True
        if self.suit_ct[3]==0:
            self.void_info[0]['C']=True
            self.void_info[1]['C']=True
            self.void_info[2]['C']=True

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
        return cards_list_list

    def calc_weight(self,vals):
        w=ScenarioGen.FACTORIAL[self.suit_ct[0]]//(ScenarioGen.FACTORIAL[vals[0]]*ScenarioGen.FACTORIAL[vals[1]]*ScenarioGen.FACTORIAL[vals[2]])
        w*=ScenarioGen.FACTORIAL[self.suit_ct[1]]//(ScenarioGen.FACTORIAL[vals[3]]*ScenarioGen.FACTORIAL[vals[4]]*ScenarioGen.FACTORIAL[vals[5]])
        w*=ScenarioGen.FACTORIAL[self.suit_ct[2]]//(ScenarioGen.FACTORIAL[vals[6]]*ScenarioGen.FACTORIAL[vals[7]]*ScenarioGen.FACTORIAL[vals[8]])
        w*=ScenarioGen.FACTORIAL[self.suit_ct[3]]//(ScenarioGen.FACTORIAL[vals[9]]*ScenarioGen.FACTORIAL[vals[10]]*ScenarioGen.FACTORIAL[vals[11]])
        return w

    def gen_num_tables(self):
        """
            generate all possible, i.e. agree with the restrictions, tables.
        """
        cnum_ct=[self.lens[0],self.lens[1]-self.lens[0],self.lens[2]-self.lens[1]] #remained card number count
        assert sum(cnum_ct)==len(self.cards_remain)
        val_dict={True:0,False:-1}
        vals=[val_dict[self.void_info[0]['S']],val_dict[self.void_info[1]['S']],val_dict[self.void_info[2]['S']],
              val_dict[self.void_info[0]['H']],val_dict[self.void_info[1]['H']],val_dict[self.void_info[2]['H']],
              val_dict[self.void_info[0]['D']],val_dict[self.void_info[1]['D']],val_dict[self.void_info[2]['D']],
              val_dict[self.void_info[0]['C']],val_dict[self.void_info[1]['C']],val_dict[self.void_info[2]['C']],]

        self.num_tables=[]
        nt_root=NumTable(vals,self.suit_ct,cnum_ct,self.num_tables)
        nt_root.breed()

        self.num_tables=[tuple(i) for i in self.num_tables]
        self.for_choice=list(range(len(self.num_tables)))
        self.num_table_weights=[self.calc_weight(i) for i in self.num_tables]
        self.num_table_count=sum(self.num_table_weights)
        self.num_table_weights=[i/self.num_table_count for i in self.num_table_weights]

    def suit_by_suit(self):
        vals=self.num_tables[numpy.random.choice(self.for_choice,p=self.num_table_weights)]
        numpy.random.shuffle(self.cards_remain)
        cards_list1=[];cards_list2=[];cards_list3=[]
        s_cards=[i for i in self.cards_remain if i[0]=='S']
        cards_list1+=s_cards[0:vals[0]]
        cards_list2+=s_cards[vals[0]:vals[0]+vals[1]]
        cards_list3+=s_cards[vals[0]+vals[1]:]
        h_cards=[i for i in self.cards_remain if i[0]=='H']
        cards_list1+=h_cards[0:vals[3]]
        cards_list2+=h_cards[vals[3]:vals[3]+vals[4]]
        cards_list3+=h_cards[vals[3]+vals[4]:]
        d_cards=[i for i in self.cards_remain if i[0]=='D']
        cards_list1+=d_cards[0:vals[6]]
        cards_list2+=d_cards[vals[6]:vals[6]+vals[7]]
        cards_list3+=d_cards[vals[6]+vals[7]:]
        c_cards=[i for i in self.cards_remain if i[0]=='C']
        cards_list1+=c_cards[0:vals[9]]
        cards_list2+=c_cards[vals[9]:vals[9]+vals[10]]
        cards_list3+=c_cards[vals[9]+vals[10]:]
        try:
            assert len(cards_list1)==self.lens[0]
            assert len(cards_list2)==self.lens[1]-self.lens[0]
            assert len(cards_list3)==self.lens[2]-self.lens[1]
            assert ScenarioGen.check_void_legal(cards_list1,cards_list2,cards_list3,self.void_info)
        except:
            print(cards_list1,cards_list2,cards_list3)
            print(self.lens)
            print(vals)
            print(self.cards_remain)
            print(self.suit_ct)
            input()
        return [cards_list1,cards_list2,cards_list3]

def test_gen_num_tables():
    s=ScenarioGen(0,[[0,'S2','S3','S4','S5'],[0,'S6','S7','S8','S9'],[0,'S10','SJ','SQ','SK'],
                     [0,'H2','H3','H4','H5'],[0,'H6','H7','H8','H9'],[0,'H10','HJ','HQ','HK'],
                     [0,'C2','C3','C4','C5'],[0,'C6','C7','C8','C9'],[0,'C10','CJ','CQ','D7'],]
                    ,[0,'SA','HA','D2'],['D3','D4','D5','D6'])
    s.gen_num_tables()
    #print("tables_num: %d, %d"%(len(s.num_tables),s.tables_num_predict))
    s.method=1;s.number=10
    for i in s:
        print(i)

if __name__=="__main__":
    test_gen_num_tables()
