import torch.nn as nn
import torch.nn.functional as F

class ConvModel(nn.Module):
    def __init__(self, input_shape, batch_size=16, num_cats=50):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size = 3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(128)
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn7 = nn.BatchNorm2d(256)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn8 = nn.BatchNorm2d(256)
        self.dense1 = nn.Linear(256*(((input_shape[1]//2)//2)//2)*(((input_shape[2]//2)//2)//2),500)
        self.dropout = nn.Dropout(0.5)
        self.dense2 = nn.Linear(500, num_cats)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = F.max_pool2d(x, kernel_size=2) 
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv5(x)
        x = F.relu(self.bn5(x))
        x = self.conv6(x)
        x = F.relu(self.bn6(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = self.conv7(x)
        x = F.relu(self.bn7(x))
        x = self.conv8(x)
        x = F.relu(self.bn8(x))
        x = x.view(x.size(0),-1)
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = self.dense2(x)
        return x



class PiczakModel(nn.Module):
    '''Source: https://www.karolpiczak.com/papers/Piczak2015-ESC-ConvNet.pdf '''
    def __init__(self, input_shape, batch_size=16, num_cats=50):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 80, kernel_size = (57, 6), stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(80)
        #self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=3, padding=1)
        self.conv2 = nn.Conv2d(80, 80, kernel_size = (1, 3), stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(80)  
        n1 = (input_shape[1]-57)+1        
        n3 = ((n1 -4) / 1) + 1
        
        n2 = (input_shape[2]-6)+1        
        n4 = int((n2 -3) / 3) + 1
        
        n5 = ((n3-1) / 1) + 1
        n7 = ((n5 - 1) / 1) + 1        
        
        n6 = ((n4 - 3) / 1) + 1
        n8 = int((n6 -3) / 3) + 1
        n = int(n8 * n7 * 80)
        self.dense1 = nn.Linear(n, 500)
        self.dense2 = nn.Linear(500, num_cats)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = F.max_pool2d(x, kernel_size=(4, 3), stride= (1,3))
        x = F.dropout(x, p=0.5)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = F.max_pool2d(x, kernel_size=(1, 3), stride= (1,3))  
        x = x.view(x.size(0),-1)
        x = self.dense1(x)
        x = F.dropout(x, p=0.5)
        x = self.dense2(x)
        x = F.dropout(x, p=0.5)
        return x

