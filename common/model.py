#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 21:14:12
LastEditor: YangDing
LastEditTime: 2021-12-11
Discription: neural network models
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class MLP(nn.Module):
    def __init__(self, input_dim,output_dim,hidden_dim=128):
        """ 初始化q网络，为全连接网络
            input_dim: 输入的特征数即环境的状态数
            output_dim: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, output_dim) # 输出层
        
    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class CNN(nn.Module):
    """针对输入为图像的 Q 网络
        输入为 3 通道的图片 (3 * h * w) ，每一层分别是: 
            容器的 hm
            物体的由上往下 hm
            物体的由下往上 hm
        输入为一张图片中的每个格点的选取概率: 
            h * w 个点
    """    
    def __init__(self, state_dim, action_dim):
        # state_dim 默认为 (3, 32, 32)
        # action_dim 默认为 1024

        assert state_dim[0] == 3 and state_dim[1] == state_dim[2] and \
            pow(state_dim[1], 2) == action_dim, "状态维度和动作维度不匹配"

        # 初始化卷积神经网络
        super(CNN, self).__init__()

        self.network = nn.Sequential(
            # 卷积层1
            # 如果在计算Q值时不考虑容器中已存在的物体下面的空洞，
            # 则只需要3张height map就足以描述当前系统的完整状态
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 卷积层2
            nn.Conv2d(16, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # 展平成1维，此时有 48 * (state_dim[1]/4) * (state_dim[1]/4) 
            # = 3 * state_dim[1]^2 个元素
            nn.Flatten(),
            # 全连接层
            nn.Linear(3 * action_dim, 2 * action_dim),
            nn.ReLU(),
            nn.Linear(2 * action_dim, action_dim),
        )

        
    def forward(self, x):
        result = self.network(x)
        return result


class Critic(nn.Module):
    def __init__(self, n_obs, output_dim, hidden_size, init_w=3e-3):
        super(Critic, self).__init__()
        
        self.linear1 = nn.Linear(n_obs + output_dim, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class Actor(nn.Module):
    def __init__(self, n_obs, output_dim, hidden_size, init_w=3e-3):
        super(Actor, self).__init__()  
        self.linear1 = nn.Linear(n_obs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_dim)
        
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class ActorCritic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ActorCritic, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value