import numpy as np
import torch
import torch.nn as nn 

# class to test .pt model saved as weight only
# /!\ The structure of this model was made quite randomly, do not expect it to be good at anything
# (there is probably too many conv layer considering the input size anyways)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, 1, padding=1)
        self.conv_bn_1 = nn.BatchNorm2d(16)
        torch.nn.init.normal_(self.conv1.weight)
        torch.nn.init.zeros_(self.conv1.bias)
        
        self.conv2 = nn.Conv2d(16, 32, 3, 1, padding=1)
        self.conv_bn_2 = nn.BatchNorm2d(32)
        torch.nn.init.normal_(self.conv2.weight)
        torch.nn.init.zeros_(self.conv2.bias)
        
        self.conv3 = nn.Conv2d(32, 64 , 3, 1, padding=1)
        self.conv_bn_3 = nn.BatchNorm2d(64)
        torch.nn.init.normal_(self.conv3.weight)
        torch.nn.init.zeros_(self.conv3.bias)
        
        self.conv4 = nn.Conv2d(64, 128 , 3, 1, padding=1)
        self.conv_bn_4 = nn.BatchNorm2d(128)
        torch.nn.init.normal_(self.conv4.weight)
        torch.nn.init.zeros_(self.conv4.bias)
        
        self.pool  = nn.MaxPool2d(2,2)

        self.act   = nn.ReLU(inplace=False)
        self.drop = nn.Dropout2d(0.2)
        
        self.mlp = nn.Sequential(
            nn.Linear(4*4*128,64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(16, 2)  
            #nn.Softmax(dim=-1)    
        )
    
    def forward(self, x):
        x = self.conv_bn_1(self.conv1(x))
        x = self.pool(self.act(x))
        x = self.drop(x)

        x = self.conv_bn_2(self.conv2(x))
        x = self.pool(self.act(x))
        x = self.drop(x)
        
        x = self.conv_bn_3(self.conv3(x))
        x = self.pool(self.act(x))
        x = self.drop(x)
        
        x = self.conv_bn_4(self.conv4(x))
        x = self.pool(self.act(x))
        x = self.drop(x)
        
        bsz, nch, height, width = x.shape
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        y = self.mlp(x)

        return y

