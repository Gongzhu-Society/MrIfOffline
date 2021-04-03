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

    def guessing_score(gs_net, possible_hands, cards_on_table, history, place, device):
        cards_remain = copy.deepcopy(possible_hands)
        #print(cards_remain)
        '''
        for his in history:
            for i in range(len(his) - 1):
                print("his, i=", i)
                print(his[(his[0] + i) % 4])
                print(his[1 + i])
                cards_remain[(his[0] + i) % 4].remove(his[1 + i])
        for i in range(len(cards_on_table) - 1):
            print("cot, i=",i)
            print(cards_on_table)
            print(cards_remain)
            print(possible_hands)
            print(cards_remain[(cards_on_table[0] + i) % 4])
            print(cards_on_table[1 + i])
            cards_remain[(cards_on_table[0] + i) % 4].remove(cards_on_table[1 + i])

        '''
        oh = SimpleGuesser.prepare_ohs(cards_on_table, history, place, cards_remain[place])
        output = gs_net(oh.unsqueeze(0).to(device))
        output = output.view(1,3,-1)

        representation = SimpleGuesser.prepare_target_ohs(cards_remain,place).unsqueeze(0).to(device)
        mask = 3 * torch.mean(representation, dim=1)
        mask = torch.stack([mask, mask, mask], dim=1).to(device)
        p_cardinplayer = F.softmax(output * mask, dim=1)
        score = torch.sum(p_cardinplayer*representation)
        return score


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