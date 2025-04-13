#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import os
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


# In[2]:


BATCH_SIZE = 64
LR = 0.001
EPOCH = 60
DEVICE = torch.device('cpu')


# In[ ]:


# path_train = 'face_images/resnet_train_set'
# path_vaild = 'face_images/resnet_vaild_set'

# transforms_train = transforms.Compose([
#     transforms.Grayscale(),#使用ImageFolder默认扩展为三通道，重新变回去就行
#     transforms.RandomHorizontalFlip(),#随机翻转
#     transforms.ColorJitter(brightness=0.5, contrast=0.5),#随机调整亮度和对比度
#     transforms.ToTensor()
# ])
# transforms_vaild = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.ToTensor()
# ])

# data_train = torchvision.datasets.ImageFolder(root=path_train,transform=transforms_train)
# data_vaild = torchvision.datasets.ImageFolder(root=path_vaild,transform=transforms_vaild)

# train_set = torch.utils.data.DataLoader(dataset=data_train,batch_size=BATCH_SIZE,shuffle=True)
# vaild_set = torch.utils.data.DataLoader(dataset=data_vaild,batch_size=BATCH_SIZE,shuffle=False)


# In[3]:


# 设置数据路径
train_dir = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/github_code/dataset/train'
val_dir = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/github_code/dataset/val'
test_dir = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/github_code/dataset/test'

# 定义数据预处理操作
train_transforms = transforms.Compose([
    transforms.Resize((48, 48)),  # 调整图像大小
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.RandomRotation(10),  # 随机旋转10度
    transforms.RandomResizedCrop(48, scale=(0.9, 1.0)),  # 随机裁剪
    transforms.ColorJitter(brightness=0.1, contrast=0.1),  # 随机调整亮度和对比度
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

val_transforms = transforms.Compose([
    transforms.Resize((48, 48)),  # 调整图像大小
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 加载数据集
data_train = ImageFolder(root=train_dir, transform=train_transforms)
data_vaild = ImageFolder(root=val_dir, transform=val_transforms)
data_test = ImageFolder(root=test_dir, transform=val_transforms)

# 打印数据集信息
print(f"Training set size: {len(data_train)}")
print(f"Validation set size: {len(data_vaild)}")
print(f"Test set size: {len(data_test)}")

# 定义数据加载器
BATCH_SIZE = 128  # 可以根据需要调整批量大小
train_set = DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)
vaild_set = DataLoader(dataset=data_vaild, batch_size=BATCH_SIZE, shuffle=False)
test_set = DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=False)

# 如果需要使用批量大小为64的数据加载器
train_loader_64 = DataLoader(dataset=data_train, batch_size=64, shuffle=True)
val_loader_64 = DataLoader(dataset=data_vaild, batch_size=64, shuffle=False)
test_loader_64 = DataLoader(dataset=data_test, batch_size=64, shuffle=False)


# In[4]:


# train_dir = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/dataset/train'
# val_dir = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/dataset/val'
# test_dir = '/Users/gyd/HKU_STUDY/COMP_7404/Group_project/dataset/test'


# In[5]:


class ResNet(nn.Module):
    def __init__(self, *args):
        super(ResNet, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0],-1)


class GlobalAvgPool2d(nn.Module):
    # 全局平均池化层可通过将池化窗口形状设置成输入的高和宽实现
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()
    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:])


# 残差神经网络
class Residual(nn.Module): 
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return F.relu(Y + X)

    
def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    if first_block:
        assert in_channels == out_channels # 第一个模块的通道数同输入通道数一致
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, stride=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)


# In[6]:


resnet = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7 , stride=2, padding=3),
    nn.BatchNorm2d(64), 
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
resnet.add_module("resnet_block1", resnet_block(64, 64, 3, first_block=True))
resnet.add_module("resnet_block2", resnet_block(64, 128, 4))
resnet.add_module("resnet_block3", resnet_block(128, 256, 6))
resnet.add_module("resnet_block4", resnet_block(256, 512, 3))
resnet.add_module("global_avg_pool", GlobalAvgPool2d()) # GlobalAvgPool2d的输出: (Batch, 512, 1, 1)
resnet.add_module("fc", nn.Sequential(ResNet(), nn.Linear(512, 7))) 

model = resnet
model.to(DEVICE)
# optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
optimizer = optim.Adam(model.parameters(),lr=LR)
criterion = nn.CrossEntropyLoss()


train_loss = []
train_ac = []
vaild_loss = []
vaild_ac = []
y_pred = []


# In[7]:


def train(model,device,dataset,optimizer,epoch):
    model.train()
    correct = 0
    for i,(x,y) in tqdm(enumerate(dataset)):
        x , y  = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x)
        pred = output.max(1,keepdim=True)[1]
        correct += pred.eq(y.view_as(pred)).sum().item()
        loss = criterion(output,y) 
        loss.backward()
        optimizer.step()   
        
    train_ac.append(correct/len(data_train))   
    train_loss.append(loss.item())
    print("Epoch {} Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(epoch,loss,correct,len(data_train),100*correct/len(data_train)))

def vaild(model,device,dataset):
    model.eval()
    correct = 0
    with torch.no_grad():
        for i,(x,y) in tqdm(enumerate(dataset)):
            x,y = x.to(device) ,y.to(device)
            output = model(x)
            loss = criterion(output,y)
            pred = output.max(1,keepdim=True)[1]
            global  y_pred 
            y_pred += pred.view(pred.size()[0]).cpu().numpy().tolist()
            correct += pred.eq(y.view_as(pred)).sum().item()
            
    vaild_ac.append(correct/len(data_vaild)) 
    vaild_loss.append(loss.item())
    print("Test Loss {:.4f} Accuracy {}/{} ({:.0f}%)".format(loss,correct,len(data_vaild),100.*correct/len(data_vaild)))


def RUN():
    for epoch in range(1,EPOCH+1):
        '''if epoch==15 :
            LR = 0.1
            optimizer=optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
        if(epoch>30 and epoch%15==0):
            LR*=0.1
            optimizer=optimizer = optim.SGD(model.parameters(),lr=LR,momentum=0.9)
        '''
        #尝试动态学习率
        train(model,device=DEVICE,dataset=train_set,optimizer=optimizer,epoch=epoch)
        vaild(model,device=DEVICE,dataset=vaild_set)
        torch.save(model,'model/model_resnet.pkl')


# In[8]:


if __name__ == '__main__':
    RUN()


# In[ ]:


# get_ipython().system(' jupyter nbconvert --to script model.ipynb')


# # In[ ]:


# import torch

# # 加载整个模型（包含结构和参数）
# model = torch.load('model/model_resnet.pkl')

# # 设置为评估模式
# model.eval()
# model.to(DEVICE)

