#2021/10/29 17:36
from torch import nn
import torchvision
from torch.nn  import functional as F
from torch.utils.tensorboard import SummaryWriter

class CustomerNet(nn.Module):
    def __init__(self):
        super(CustomerNet,self).__init__()
        self.conv2d0=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=5,stride=1,padding=2,bias=True)
        self.conv2d1=nn.Conv2d(in_channels=32,out_channels=32,kernel_size=5,stride=1,padding=2,bias=True)
        self.conv2d2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=5,stride=1,padding=2,bias=True)
        self.maxpool0=nn.MaxPool2d(kernel_size=2,stride=2,padding=0)
        self.conv2d3=nn.Conv2d(in_channels=64,out_channels=32,kernel_size=5,stride=1,padding=2,bias=True)
        self.conv2d4=nn.Conv2d(in_channels=32,out_channels=3,kernel_size=5,stride=1,padding=0,bias=True)
        self.flatten1=nn.Flatten()
        self.relu=nn.ReLU()
        self.liner1=nn.Linear(in_features=2160,out_features=320)
        self.liner2=nn.Linear(in_features=320,out_features=80)
        self.liner3=nn.Linear(in_features=80,out_features=20)
        self.liner4=nn.Linear(in_features=20,out_features=1)

        self.liner11=nn.Linear(in_features=4096,out_features=1024)
        self.liner22=nn.Linear(in_features=1024,out_features=128)
        self.liner33=nn.Linear(in_features=128,out_features=1)


    def forward(self,x):
        x=self.relu(self.conv2d0(x))
        x=self.maxpool0(x)
        x=self.relu(self.conv2d1(x))
        x=self.maxpool0(x)
        x=self.relu(self.conv2d2(x))
        x=self.maxpool0(x)
        x=self.relu(self.flatten1(x))
        # print(x.shape)
        x=F.relu(self.liner11(x))
        x=self.relu(self.liner22(x))
        x=self.liner33(x)
        return x


