def loss_func_single(netout,target_index,legal_mask):
    o1=netout*legal_mask
    #o2=o1/(torch.sum(o1)+1e-7)
    #o2=F.softmax(o1,dim=0)
    #loss_1=torch.sum(torch.pow(o2-target,2))
    #cross_entropy里有一个softmax
    loss_1=F.cross_entropy(torch.stack((o1,)),torch.tensor([target_index,]))
    #loss_2=torch.sqrt(torch.sum(torch.pow(netout-o1,2)))
    return loss_1

def correct_or_not(netout,target_index,legal_mask):
    _,max_i=torch.max(netout*legal_mask,0)
    if max_i==target_index:
        return 1
    else:
        return 0 

def train_first():
    train_data_1=parse_train_data("./Greed_batch/Greed_batch1.txt",trick_num=512)
    train_data_2=parse_train_data("./Greed_batch/Greed_batch2.txt",trick_num=512)
    check_data=parse_train_data("./Greed_batch/Greed_batch3.txt",trick_num=64)
    train_data_list=train_data_1[0]+train_data_2[0]
    net=NN_First()
    optimizer=optim.SGD(net.parameters(),lr=0.0005,momentum=0.9)
    for epoch in range(1000):
        log("%dth epoch"%(epoch))
        optimizer.zero_grad()
        running_loss=0
        for i,lis in enumerate(train_data_list):
            inputs=torch.cat((lis[2],lis[3]))
            netout=net(inputs)
            loss=loss_func_single(netout,lis[5],lis[4])
            loss.backward()
            running_loss+=loss.item()
            if i%100==99:
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.step()
            optimizer.zero_grad()
        log("finish loss: %f"%(running_loss/(i+1)))
        check_loss=0
        check_corr=0
        for i,lis in enumerate(check_data[0]):
            inputs=torch.cat((lis[2],lis[3]))
            netout=net(inputs)
            loss=loss_func_single(netout,lis[5],lis[4])
            check_loss+=loss.item()
            corr=correct_or_not(netout,lis[5],lis[4])
            check_corr+=corr
        log("check: %f, %f"%(check_loss/(i+1),check_corr/(i+1)))
    torch.save(net.state_dict(),'nn_first.data')