#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import time,sys,traceback,math,numpy
LOGLEVEL={0:"DEBUG",1:"INFO",2:"WARN",3:"ERR",4:"FATAL"}
LOGFILE=sys.argv[0].split(".")
LOGFILE[-1]="log"
LOGFILE=".".join(LOGFILE)
def log(msg,l=1,end="\n",logfile=None,fileonly=False):
    st=traceback.extract_stack()[-2]
    lstr=LOGLEVEL[l]
    now_str="%s %03d"%(time.strftime("%y/%m/%d %H:%M:%S",time.localtime()),math.modf(time.time())[0]*1000)
    if l<3:
        tempstr="%s [%s,%s:%d] %s%s"%(now_str,lstr,st.name,st.lineno,str(msg),end)
    else:
        tempstr="%s [%s,%s:%d] %s:\n%s%s"%(now_str,lstr,st.name,st.lineno,str(msg),traceback.format_exc(limit=5),end)
    if not fileonly:
        print(tempstr,end="")
    if l>=1 or fileonly:
        if logfile==None:
            logfile=LOGFILE
        with open(logfile,"a") as f:
            f.write(tempstr)

ORDER_DICT1={'S':-300,'H':-200,'D':-100,'C':0,'J':-200}
ORDER_DICT2={'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'1':10,'J':11,'Q':12,'K':13,'A':14,'P':15,'G':16}
ORDER_DICT={'S2': 0,'S3': 1,'S4': 2,'S5': 3,'S6': 4,'S7': 5,'S8': 6,'S9': 7,'S10': 8,'SJ': 9,'SQ':10,'SK':11,'SA':12, 
            'H2':13,'H3':14,'H4':15,'H5':16,'H6':17,'H7':18,'H8':19,'H9':20,'H10':21,'HJ':22,'HQ':23,'HK':24,'HA':25, 
            'D2':26,'D3':27,'D4':28,'D5':29,'D6':30,'D7':31,'D8':32,'D9':33,'D10':34,'DJ':35,'DQ':36,'DK':37,'DA':38, 
            'C2':39,'C3':40,'C4':41,'C5':42,'C6':43,'C7':44,'C8':45,'C9':46,'C10':47,'CJ':48,'CQ':49,'CK':50,'CA':51}
ORDER_DICT4={'S':0,'H':1,'D':2,'C':3}
ORDER_DICT5={'H2':0,'H3':1,'H4':2,'H5':3,'H6':4,'H7':5,'H8':6,'H9':7,'H10':8,'HJ':9,'HQ':10,'HK':11,'HA':12,
             'SQ':13,'DJ':14,'C10':15}
cards_order=lambda c:ORDER_DICT1[c[0]]+ORDER_DICT2[c[1]]

INIT_CARDS=['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'SJ', 'SQ', 'SK', 'SA', 
            'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'HJ', 'HQ', 'HK', 'HA', 
            'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'DJ', 'DQ', 'DK', 'DA', 
            'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'CJ', 'CQ', 'CK', 'CA']

SCORE_DICT={'SQ':-100,'DJ':100,'C10':0,
            'H2':0,'H3':0,'H4':0,'H5':-10,'H6':-10,'H7':-10,'H8':-10,'H9':-10,'H10':-10,
            'HJ':-20,'HQ':-30,'HK':-40,'HA':-50,'JP':-60,'JG':-70}

def calc_score(l):
    s=0
    has_score_flag=False
    c10_flag=False
    for i in l:
        if i=="C10":
            c10_flag=True
        else:
            s+=SCORE_DICT[i]
            has_score_flag=True
    if c10_flag==True:
        if has_score_flag==False: 
            s+=50
        else:
            s*=2
    return s

# git push https://github.com/Gongzhu-Society/MrIfOffline.git
# git pull https://github.com/Gongzhu-Society/MrIfOffline.git
# git add .
# git commit