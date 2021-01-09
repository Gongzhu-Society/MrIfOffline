#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

VALUE_RENORMAL=10

class PV_NET_FATHER(nn.Module):
    """
        return 52 policy and 1 value
    """
    def num_paras(self):
        return sum([p.numel() for p in self.parameters()])

    def num_layers(self):
        ax=0
        for name,child in self.named_children():
            ax+=1
        return ax

    def __str__(self):
        stru=[]
        for name,child in self.named_children():
            if 'weight' in child.state_dict():
                #stru.append(tuple(child.state_dict()['weight'].t().size()))
                stru.append(child.state_dict()['weight'].shape)
        return "%s %s %s"%(self.__class__.__name__,stru,self.num_paras())

class PV_NET_0(nn.Module):
    """
        return 52 policy and 1 value
    """

    def __init__(self):
        super(PV_NET,self).__init__()
        #cards in four player(52*4), two cards on table(52*3*2), scores in four players
        self.fc0=nn.Linear(52*4+(54*3+20*4)+16*4,1024)
        self.fc1=nn.Linear(1024,1024)
        self.fc2=nn.Linear(1024,256)
        self.fc3=nn.Linear(256,256)
        self.fc4=nn.Linear(256,256)
        self.fc5=nn.Linear(256,256)
        self.fc6=nn.Linear(256,256)
        self.fc7=nn.Linear(256,256)
        self.fc8=nn.Linear(256,256)
        self.fcp=nn.Linear(256,52)
        self.fcv=nn.Linear(256,1)

    def forward(self, x):
        x=F.relu(self.fc0(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc4(F.relu(self.fc3(x))))+x
        x=F.relu(self.fc6(F.relu(self.fc5(x))))+x
        x=F.relu(self.fc8(F.relu(self.fc7(x))))+x
        p=self.fcp(x)
        v=self.fcv(x)*VALUE_RENORMAL
        return p,v

class PV_NET_1(PV_NET_FATHER):
    def __init__(self):
        super(PV_NET,self).__init__()
        self.fc0=nn.Linear(52*4+(54*3+0*4)+16*4,2048)
        self.fc1=nn.Linear(2048,2048)
        self.fc2=nn.Linear(2048,512)

        self.sc0a=nn.Linear(512,512)
        self.sc0b=nn.Linear(512,512)
        self.sc1a=nn.Linear(512,512)
        self.sc1b=nn.Linear(512,512)
        self.sc2a=nn.Linear(512,512)
        self.sc2b=nn.Linear(512,512)
        self.sc3a=nn.Linear(512,512)
        self.sc3b=nn.Linear(512,512)
        self.sc4a=nn.Linear(512,512)
        self.sc4b=nn.Linear(512,512)
        self.sc5a=nn.Linear(512,512)
        self.sc5b=nn.Linear(512,512)

        self.fcp=nn.Linear(512,52)
        self.fcv=nn.Linear(512,1)

    def forward(self, x):
        x=F.relu(self.fc0(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.sc0b(F.relu(self.sc0a(x))))+x
        x=F.relu(self.sc1b(F.relu(self.sc1a(x))))+x
        x=F.relu(self.sc2b(F.relu(self.sc2a(x))))+x
        x=F.relu(self.sc3b(F.relu(self.sc3a(x))))+x
        x=F.relu(self.sc4b(F.relu(self.sc4a(x))))+x
        x=F.relu(self.sc5b(F.relu(self.sc5a(x))))+x
        p=self.fcp(x)
        v=self.fcv(x)*VALUE_RENORMAL
        return p,v

class PV_NET_2(PV_NET_FATHER):
    def __init__(self):
        super(PV_NET,self).__init__()
        self.fc0=nn.Linear(52*4+(54*3+0*4)+16*4,2048)
        self.fc1=nn.Linear(2048,2048)
        self.fc2=nn.Linear(2048,512)

        self.sc0a=nn.Linear(512,512)
        self.sc0b=nn.Linear(512,512)
        self.sc1a=nn.Linear(512,512)
        self.sc1b=nn.Linear(512,512)
        self.sc2a=nn.Linear(512,512)
        self.sc2b=nn.Linear(512,512)
        self.sc3a=nn.Linear(512,512)
        self.sc3b=nn.Linear(512,512)
        self.sc4a=nn.Linear(512,512)
        self.sc4b=nn.Linear(512,512)
        self.sc5a=nn.Linear(512,512)
        self.sc5b=nn.Linear(512,512)
        self.sc6a=nn.Linear(512,512)
        self.sc6b=nn.Linear(512,512)
        self.sc7a=nn.Linear(512,512)
        self.sc7b=nn.Linear(512,512)
        self.sc8a=nn.Linear(512,512)
        self.sc8b=nn.Linear(512,512)
        self.sc9a=nn.Linear(512,512)
        self.sc9b=nn.Linear(512,512)

        self.fcp=nn.Linear(512,52)
        self.fcv=nn.Linear(512,1)

    def forward(self, x):
        x=F.relu(self.fc0(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.sc0b(F.relu(self.sc0a(x))))+x
        x=F.relu(self.sc1b(F.relu(self.sc1a(x))))+x
        x=F.relu(self.sc2b(F.relu(self.sc2a(x))))+x
        x=F.relu(self.sc3b(F.relu(self.sc3a(x))))+x
        x=F.relu(self.sc4b(F.relu(self.sc4a(x))))+x
        x=F.relu(self.sc5b(F.relu(self.sc5a(x))))+x
        x=F.relu(self.sc6b(F.relu(self.sc6a(x))))+x
        x=F.relu(self.sc7b(F.relu(self.sc7a(x))))+x
        x=F.relu(self.sc8b(F.relu(self.sc8a(x))))+x
        x=F.relu(self.sc9b(F.relu(self.sc9a(x))))+x
        p=self.fcp(x)
        v=self.fcv(x)*VALUE_RENORMAL
        return p,v

class BasicBlock(nn.Module):
    expansion=1 #?

    def __init__(self,in_planes,planes,stride=1):
        super(BasicBlock, self).__init__()
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(planes)
        self.conv2=nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(planes)
        self.shortcut=nn.Sequential()
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride!=1 or in_planes!=self.expansion*planes:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out=F.relu(self.bn1(self.conv1(x)))
        out=self.bn2(self.conv2(out))
        out+=self.shortcut(x)
        out=F.relu(out)
        return out

class RES_NET_18(PV_NET_FATHER):
    def __init__(self,block=BasicBlock,num_blocks=[2,2,2,2]):#,num_classes=52):
        super(RES_NET_18,self).__init__()
        self.in_planes=64

        #self.fc0=nn.Linear(52*4+54*3+16*4,3*32*32)
        self.conv1=nn.Conv2d(3,64,kernel_size=5,stride=1,padding=1,bias=False)
        self.bn1=nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(block,64,num_blocks[0],stride=1)
        self.layer2 = self._make_layer(block,128,num_blocks[1],stride=2)
        self.layer3 = self._make_layer(block,256,num_blocks[2],stride=2)
        self.layer4 = self._make_layer(block,512,num_blocks[3],stride=2)
        self.fcp=nn.Linear(512*block.expansion,52)
        self.fcv=nn.Linear(512*block.expansion,1)

    def _make_layer(self,block,planes,num_blocks,stride):
        strides=[stride]+[1]*(num_blocks-1)
        layers=[]
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride))
            self.in_planes=planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        #out=F.relu(self.fc0(x)).view(-1,3,32,32)
        out=torch.cat((F.pad(x.repeat(1,2),(0,156)).view(-1,1,32,32),
            F.pad(x.repeat(1,2),(78,78)).view(-1,1,32,32),
            F.pad(x.repeat(1,2),(156,0)).view(-1,1,32,32)),1)
        out=F.relu(self.bn1(self.conv1(out)))
        out=self.layer1(out)
        out=self.layer2(out)
        out=self.layer3(out)
        out=self.layer4(out)
        out=F.avg_pool2d(out,4)
        out=out.view(-1,512)
        p=self.fcp(out)
        v=self.fcv(out)*VALUE_RENORMAL
        return p,v

    def __str__(self):
        stru=[]
        return "%s %s"%(self.__class__.__name__,self.num_paras())

