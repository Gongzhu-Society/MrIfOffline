class PV_NET(nn.Module):
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

class PV_NET(PV_NET_FATHER):
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