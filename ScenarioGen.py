#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,INIT_CARDS
import copy,numpy,time,itertools

class NumTable():
    """
        The number table used to contruct legal scenarios, it looks like
                A | B | C
               --- --- ---
            S | 0 | 1 | 2
            -- --- --- ---
            H | 3 | 4 | 5
            -- --- --- ---
            D | 6 | 7 | 8
            -- --- --- ---
            C | 9 | x | x
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

    def C(m,n):
        """mCn, m is the larger one"""
        if n==3:
            return m*(m-1)*(m-2)//6
        elif n==2:
            return m*(m-1)//2
        elif n==1:
            return m
        else:
            return 1

    def __init__(self,myseat,history,cards_on_table,cards_list,number=20,method=None,METHOD1_PREFERENCE=0):
        """
            myseat, history and cards_on_table will be fed to gen_void_info to get void_info
            history, cards_on_table and cards_list will be fed to gen_cards_remain to get cards_remain
            cards_list: cards in my hand
            number: the max number of sampling
            method: 0 for "shot and test", 1 for
            METHOD1_PREFERENCE: will be used in decide_method
        """

        #void_info and cards_remain are all in relative order!
        self.void_info=ScenarioGen.gen_void_info(myseat,history,cards_on_table)
        self.cards_remain=ScenarioGen.gen_cards_remain(history,cards_on_table,cards_list)
        #log(self.void_info)
        #log(self.cards_remain)
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
        #log(self.lens)
        assert self.lens[2]==len(self.cards_remain)

        self.number=number
        self.METHOD1_PREFERENCE=METHOD1_PREFERENCE
        self.suc_ct=0 #success counter
        if method==None:
            self.decide_method()
        elif method==1:
            self.method=1
            s_temp=''.join((i[0] for i in self.cards_remain))
            self.suit_ct=[s_temp.count(s) for s in "SHDC"] #suit remain number count
            self.reinforce_void_info()
            self.gen_num_tables()
        else:
            self.method=0
            self.tot_ct=0 #total try counter

    def decide_method(self):
        s_temp=''.join((i[0] for i in self.cards_remain))
        self.suit_ct=[s_temp.count(s) for s in "SHDC"] #suit remain number count
        self.reinforce_void_info()
        #predict success rate
        cell_num=[sum((1 for j in range(3) if not self.void_info[j]['S'])),sum((1 for j in range(3) if not self.void_info[j]['H'])),
                  sum((1 for j in range(3) if not self.void_info[j]['D'])),sum((1 for j in range(3) if not self.void_info[j]['C']))]
        self.suc_rate_predict=1
        self.tables_num_predict=1
        for i in range(4):
            self.suc_rate_predict*=(cell_num[i]/3)**(self.suit_ct[i])
            self.tables_num_predict*=ScenarioGen.C(self.suit_ct[i]+cell_num[i]-1,cell_num[i]-1)
        col_res=1
        for j in range(3):
            p_suit=[self.suit_ct[i] for i in range(4) if not self.void_info[j]['SHDC'[i]]]
            if len(p_suit)>0:
                col_res*=(max(p_suit)+1)
        self.tables_num_predict/=col_res**(2/3)

        method0_time=(0.63*len(self.cards_remain)+1.01)/self.suc_rate_predict
        method1_time=self.tables_num_predict*80/self.number+25
        if method1_time-self.METHOD1_PREFERENCE<method0_time:
            self.method=1
            self.gen_num_tables()
        else:
            self.method=0
            self.tot_ct=0 #total try counter
        #log("choose method: %d"%(self.method))

    def __iter__(self):
        if self.method==2:
            return self.exhaustive.__iter__()
        else:
            return self

    def __next__(self):
        if self.suc_ct>=self.number:
            #for timing
            """if self.suc_ct/self.tot_ct<0.1:
                tik=time.time()
                self.gen_num_tables()
                tok=time.time()
                #log("%.2fus/table"%((tok-tik)*1e6/len(self.num_tables)))
                #input()
                tik=time.time()
                for i in range(10000):
                    self.construct_by_table()
                tok=time.time()
                log("%.2fus, %d"%((tok-tik)/10000*1e6,len(self.cards_remain)))
                input()"""
            """self.tot_ct=0
            tik=time.time()
            for i in range(100000):
                pass
                #self.shot_and_test()
            tok=time.time()
            log("%.2f, %.3f, %d"%(100000*100/self.tot_ct,(tok-tik)/self.tot_ct*1e6,len(self.cards_remain)))
            input()"""
            raise StopIteration
        if self.method==0:
            result=self.shot_and_test()
            self.suc_ct+=1
            return result
        elif self.method==1:
            result=self.construct_by_table()
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
        void_info=[{'S':False,'H':False,'D':False,'C':False},{'S':False,'H':False,'D':False,'C':False},
                   {'S':False,'H':False,'D':False,'C':False}]
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
            numpy.random.shuffle(self.cards_remain)
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

        self.s_cards=[i for i in self.cards_remain if i[0]=='S']
        self.h_cards=[i for i in self.cards_remain if i[0]=='H']
        self.d_cards=[i for i in self.cards_remain if i[0]=='D']
        self.c_cards=[i for i in self.cards_remain if i[0]=='C']

        if self.num_table_count<=self.number:
            self.method=2
            self.exhaust()

    def exhaust(self):
        self.exhaustive=[]
        for vals in self.num_tables:
            s_cases=[];h_cases=[];d_cases=[];c_cases=[]
            for player1 in itertools.combinations(self.s_cards,vals[0]):
                cards_left=[i for i in self.s_cards if i not in player1]
                for player2 in itertools.combinations(cards_left,vals[1]):
                    player3=[i for i in cards_left if i not in player2]
                    s_cases.append((player1,player2,player3))
            for player1 in itertools.combinations(self.h_cards,vals[3]):
                cards_left=[i for i in self.h_cards if i not in player1]
                for player2 in itertools.combinations(cards_left,vals[4]):
                    player3=[i for i in cards_left if i not in player2]
                    h_cases.append((player1,player2,player3))
            for player1 in itertools.combinations(self.d_cards,vals[6]):
                cards_left=[i for i in self.d_cards if i not in player1]
                for player2 in itertools.combinations(cards_left,vals[7]):
                    player3=[i for i in cards_left if i not in player2]
                    d_cases.append((player1,player2,player3))
            for player1 in itertools.combinations(self.c_cards,vals[9]):
                cards_left=[i for i in self.c_cards if i not in player1]
                for player2 in itertools.combinations(cards_left,vals[10]):
                    player3=[i for i in cards_left if i not in player2]
                    c_cases.append((player1,player2,player3))
            self.exhaustive+=[[list(s[0]+h[0]+d[0]+c[0]),list(s[1]+h[1]+d[1]+c[1]),s[2]+h[2]+d[2]+c[2]]\
                             for s,h,d,c in itertools.product(s_cases,h_cases,d_cases,c_cases)]
        assert len(self.exhaustive)==self.num_table_count
        #log("exhaust: %d cases, %s\n%s\n%s\n%s"%(self.num_table_count,self.cards_remain,self.void_info,self.num_tables,self.exhaustive))
        #input()

    def construct_by_table(self):
        vals=self.num_tables[numpy.random.choice(self.for_choice,p=self.num_table_weights)]
        numpy.random.shuffle(self.s_cards)
        numpy.random.shuffle(self.h_cards)
        numpy.random.shuffle(self.d_cards)
        numpy.random.shuffle(self.c_cards)
        cards_list1=self.s_cards[0:vals[0]]+self.h_cards[0:vals[3]]+self.d_cards[0:vals[6]]+self.c_cards[0:vals[9]]
        cards_list2=self.s_cards[vals[0]:vals[0]+vals[1]]+self.h_cards[vals[3]:vals[3]+vals[4]]+\
                    self.d_cards[vals[6]:vals[6]+vals[7]]+self.c_cards[vals[9]:vals[9]+vals[10]]
        cards_list3=self.s_cards[vals[0]+vals[1]:]+self.h_cards[vals[3]+vals[4]:]+\
                    self.d_cards[vals[6]+vals[7]:]+self.c_cards[vals[9]+vals[10]:]

        #assert len(cards_list1)==self.lens[0]
        #assert len(cards_list2)==self.lens[1]-self.lens[0]
        assert len(cards_list3)==self.lens[2]-self.lens[1]
        #assert ScenarioGen.check_void_legal(cards_list1,cards_list2,cards_list3,self.void_info)
        return [cards_list1,cards_list2,cards_list3]

def test_gen_num_tables():
    s=ScenarioGen(0,[[0,'S2','S3','S4','S5'],[0,'S6','S7','S8','S9'],[0,'S10','SJ','SQ','SK'],
                     [0,'H2','H3','H4','H5'],[0,'H6','H7','H8','H9'],[0,'H10','HJ','HQ','HK'],
                     [0,'C2','C3','C4','C5'],[0,'C6','C7','C8','C9'],[0,'C10','CJ','CQ','D7'],]
                    ,[0,'SA','HA','D2'],['D3','D4','D5','D6'],number=1000)#,method=1)
    for i in s:
        log(i);input()

def test_by_touchstone():
    h=[[0, 'CA', 'C10', 'CQ', 'C3'], [0, 'D3', 'DQ', 'D6', 'D9'], [1, 'S10', 'SA', 'SQ', 'SK'], [2, 'S4', 'SJ', 'S3', 'S6'], [3, 'H5', 'H8', 'H7', 'H9'], [2, 'H6', 'HJ', 'HQ', 'H10'],
       [0, 'S8', 'CK', 'CJ', 'S7'],
       [0, 'S9', 'D10', 'C6', 'S2'], [0, 'C9', 'C8', 'C4', 'D8'], [0, 'DK', 'D7', 'D4', 'DA'],
       [3, 'H4', 'HA', 'C7', 'HK']]
    s=ScenarioGen(2,h,[0, 'D5', 'C5'],['C2', 'H3'],number=10,method=1)
    log(s.void_info)

if __name__=="__main__":
    test_gen_num_tables()
    #test_by_touchstone()
