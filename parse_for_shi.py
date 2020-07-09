#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import re

#pattern_shuffle 匹配 "shuffle: [['S7', 'S9', 'SK', 'H2'..."
pattern_shuffle=re.compile("shuffle: (.+)")
#pattern_play 匹配 "greed0 played S9"
pattern_play=re.compile("greed([0-3]) played ([SHDC][0-9JQKA]{1,2})")
#pattern_gamend 匹配 "game end: [-150, -300, 0, 100]"
pattern_gamend=re.compile("game end: (\\[.+?\\])")

def parse_for_shi(f):
    for line in f.readlines():
        s0=pattern_play.search(line)
        if s0:
            print("get played %s %s"%(s0.group(1),s0.group(2)))
            continue
        s1=pattern_shuffle.search(line)
        if s1:
            print("get shuffle: %s"%(s1.group(1)))
            continue
        s2=pattern_gamend.search(line)
        if s2:
            print("get game end: %s"%(s2.group(1)))
            continue
        print("cannot parse: %s"%(line))
        input()

if __name__=="__main__":
    try:
        f=open('for_shi_batch1.txt')
        parse_for_shi(f)
    except Exception as e:
        print(e)
    finally:
        f.close()