import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(4,3)
        self.fc2 = nn.Linear(3,1)

    def forward(self,x):
        embed = self.fc1(x)
        #y = self.fc2(embed)
        x = self.fc2(embed)
        return embed, x

#    def predict(self,x):
#        x = self.fc1(x)
#        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(3,1)

    def forward(self,x):
        x = self.fc1(x)
        return x

def print_para(m):
    for para in m.parameters():
        print(para)

def init_para(m):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(1)
    numpy.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    for para in m.parameters():
        if para.dim() < 2:
            nn.init.constant_(para,0)
        else:
            nn.init.xavier_uniform_(para)

if __name__ == "__main__":
    # Define DNN system
    feat = torch.rand([3,4])
    spkid = Net1()
    tts = Net2()
    y_spkid = torch.Tensor([[1],[0],[1]])
    y_tts = torch.Tensor([[1],[1],[1]])


    criterion = nn.BCEWithLogitsLoss()
    # spkid
    print("Weight for the spkid module:")
    init_para(spkid)
    print_para(spkid)
    y_spkid_embed, y_spkid_hat = spkid(feat)
    loss_spkid = criterion(y_spkid_hat, y_spkid)

    # tts
    print("Weight for the tts module:")
    init_para(tts)
    print_para(tts)
    y_tts_hat = tts(y_spkid_embed)
    loss_tts = criterion(y_tts_hat, y_tts)

    # UPDATE
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    random.seed(1)
    numpy.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    loss = loss_spkid + loss_tts
    optimizer = torch.optim.Adam(list(spkid.parameters()) + list(tts.parameters()))
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    print("AFTER UPDATE")
    print("Weight for the spkid module:")
    print_para(spkid)
    print("Weight for the tts module:")
    print_para(tts)

