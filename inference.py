#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from Util import log,calc_score
from Util import ORDER_DICT,ORDER_DICT2,ORDER_DICT5,SCORE_DICT,INIT_CARDS
from MrRandom import MrRandom
from MrGreed import MrGreed
from ScenarioGenerator.ScenarioGen import ScenarioGen
from MCTS.mcts import mcts
from OfflineInterface import OfflineInterface

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy,math
from MrZeroTreeSimple import MrZeroTreeSimple

class SimpleGuesser():
    def __init__(self):
        1+1
    def prepare_ohs(cards_on_table, history, place, cards_remain_in_my_hand):
        oh_history = MrZeroTreeSimple.history_oh(history, place)
        offset = 4*len(history)
        for i in range(1,len(cards_on_table)):
            oh_history[offset, (cards_on_table[0]+i-1-place)%4]=1
            oh_history[offset, 4+ORDER_DICT[cards_on_table[i]]]=1
            offset+=1
        my_card = torch.zeros((1,56))
        for card in cards_remain_in_my_hand:
            my_card[0,4+ORDER_DICT[card]] = 1
        return torch.cat((oh_history,my_card),dim=0)#.unsqueeze(0)
    def prepare_target_ohs(cards_remain,place):
        oh = torch.zeros((3,52))
        for i in range(3):
            for card in cards_remain[(place+1+i)%4]:
                oh[i,ORDER_DICT[card]] = 1
        return oh

class Loss0(nn.Module):
    """
        return double cross entropy loss
    """
    def __init__(self):
        super(Loss0,self).__init__()

    def forward(self, label, output_feature):
        omat = output_feature.view(len(output_feature),3,-1)
        mask = 3*torch.mean(label,dim=1)
        mask = torch.stack([mask,mask,mask],dim=1)
        p_cardinplayer = F.softmax(omat*mask,dim=1)
        #p_playerhascard = F.softmax(omat*mask, dim=2)
        loss = -torch.sum(label * (torch.log(p_cardinplayer + 1e-10)))
        #loss = -torch.sum(label*(0.1*torch.log(p_playerhascard+1e-10)+torch.log(p_cardinplayer+1e-10)))
        return loss/len(output_feature)

class Buffer(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)