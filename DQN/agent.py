#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: Ding Yang
LastEditTime: 2021-12-10
@Discription: Reinforce learning agent
'''
'''off-policy
'''

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import numpy as np
from common.memory import ReplayBuffer
from common.model import CNN, MLP

class DQN:
    """深度Q网络

    Attributes:
        state_dim: 状态空间维度
        action_dim: 动作空间维度
        device: 运行的设备类型
        gamma: 衰减系数
        frame_idx: 已经执行了多少个动作
        epsilon: 贪心策略的分界值
        batch_size: 选取多少动作经验计算一次梯度
        policy_net: 策略网络
        target_net: 目标网络（延后于策略网络更新）
        optimizer: 优化器
        memory: 做记忆回放的缓存区
    """
    
    def __init__(self, state_dim, action_dim, cfg):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = cfg.device
        self.gamma = cfg.gamma

        # e-greedy 策略相关参数
        # 记录当前已经进行了多少轮动作了
        self.frame_idx = 0
        # lambda 函数写法，计算 epsilon 的值
        self.epsilon = lambda frame_idx: cfg.epsilon_end + \
            (cfg.epsilon_start - cfg.epsilon_end) * \
            math.exp(-1. * frame_idx / cfg.epsilon_decay)

        self.batch_size = cfg.batch_size

        self.policy_net = CNN(state_dim, action_dim).to(self.device)
        self.target_net = CNN(state_dim, action_dim).to(self.device)

        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            # 复制参数到目标网路targe_net
            # 保证两个网络的参数在初始时是一致的
            target_param.data.copy_(param.data)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.memory_capacity)

    def choose_action(self, state):
        """按照 epsilon-greedy 原则选择动作
        Returns:
            int: 采取的动作的序号
        """        
        # esilon 值递减
        self.frame_idx += 1
        if random.random() > self.epsilon(self.frame_idx):
            with torch.no_grad():
                state = torch.tensor([state], device=self.device, dtype=torch.float32)
                # 获取当前状态对应的各个动作的价值函数
                q_values = self.policy_net(state)
                # 选择Q值最大的动作
                action = q_values.max(1)[1].item() 
        else:
            # 从动作域中随机选取一个动作
            action = random.randrange(self.action_dim)
        return action

    def predict(self, state):
        """直接选择 Q 值最大的动作
        Returns:
            int: 采取的动作的序号
        """        
        with torch.no_grad():
            state = torch.tensor([state], device=self.device, dtype=torch.float32)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()
        return action

    def update(self):
        # 当memory中不满足一个批量时，不更新策略
        if len(self.memory) < self.batch_size: 
            return
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.memory.sample(self.batch_size)
        # 转为张量
        state_batch = torch.tensor(
            state_batch, device=self.device, dtype=torch.float)

        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  

        reward_batch = torch.tensor(
            reward_batch, device=self.device, dtype=torch.float)  

        next_state_batch = torch.tensor(
            next_state_batch, device=self.device, dtype=torch.float)

        done_batch = torch.tensor(np.float32(
            done_batch), device=self.device)

        # 计算当前状态(s_t,a)对应的Q(s_t, a)
        q_values = self.policy_net(state_batch).gather(dim=1, index=action_batch) 
        # 计算下一时刻的状态(s_t_,a)对应的Q值
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach() 

        # 计算期望的Q值，对于终止状态，此时done_batch[0]=1, 对应的expected_q_value等于reward
        expected_q_values = reward_batch + self.gamma * next_q_values * (1-done_batch)
        loss = nn.MSELoss()(q_values, expected_q_values.unsqueeze(1))  # 计算均方根损失
        
        # 优化更新模型
        self.optimizer.zero_grad()  
        loss.backward()
        for param in self.policy_net.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step() 

    def save(self, path):
        torch.save(self.target_net.state_dict(), path+'dqn_checkpoint.pth')

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path+'dqn_checkpoint.pth'))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
