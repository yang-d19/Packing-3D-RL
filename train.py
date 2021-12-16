import torch
import datetime

from common.utils import save_results, make_dir
from common.plot import plot_rewards,plot_rewards_cn
from DQN.agent import DQN
from env import PackEnv

curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class DQNConfig:
    def __init__(self):
        # 算法名称
        self.algo = "DQN" 
        # 保存结果的路径
        self.result_path = "/outputs/" + self.env + \
            '/'+curr_time+'/results/'  
        # 保存模型的路径
        self.model_path = "/outputs/" + self.env + \
            '/'+curr_time+'/models/'  
        # 训练的回合数
        self.train_eps = 200 
        # 测试的回合数
        self.eval_eps = 30 
        # 折扣因子
        self.gamma = 0.95 
        # e-greedy策略中初始epsilon
        self.epsilon_start = 0.90 
        # e-greedy策略中的终止epsilon
        self.epsilon_end = 0.01 
        # e-greedy策略中epsilon的衰减率
        self.epsilon_decay = 500 
        # 默认学习率
        self.lr = 0.0001
        # 经验回放的容量
        self.memory_capacity = 100000
        # 计算一次梯度的批量大小
        self.batch_size = 64
        # 目标网络的更新频率
        self.target_update = 4 
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")


def env_agent_config(cfg):
    # 获取装箱问题的用于RL训练的环境API
    env = PackEnv()
    # 指定 Q 网络的输入层和输出层的维度大小
    state_dim = env.state_dim
    state_dim = env.action_dim
    # 构建深度Q网络
    agent = DQN(state_dim, state_dim, cfg)

    return env, agent


def train(cfg, env: PackEnv, agent: DQN):
    print('开始训练!')
    print(f'算法: {cfg.algo}, 设备: {cfg.device}')

    rewards = [] # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励

    for i_ep in range(cfg.train_eps):
        
        state = env.reset()
        done = False
        ep_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            agent.update()
            if done:
                break
        # 每固定的回合数，将目标网络的参数替换为策略网络的参数
        if (i_ep + 1) % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        if (i_ep + 1) % 10 == 0:
            print('回合: {}/{}, 奖励: {}'.format(i_ep+1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)

        # save ma_rewards
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成训练!')
    return rewards, ma_rewards


def eval(cfg, env: PackEnv, agent: DQN):
    print('开始测试!')
    print(f'算法: {cfg.algo}, 设备: {cfg.device}')
    rewards = []  
    ma_rewards = [] # moving average rewards
    for i_ep in range(cfg.eval_eps):
        ep_reward = 0  # reward per episode
        state = env.reset()  
        while True:
            action = agent.predict(state) 
            next_state, reward, done = env.step(action)  
            state = next_state  
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(ma_rewards[-1]*0.9+ep_reward*0.1)
        else:
            ma_rewards.append(ep_reward)
        print(f"回合: {i_ep+1}/{cfg.eval_eps}, 奖励: {ep_reward:.1f}")
    print('完成测试!')
    return rewards,ma_rewards


if __name__ == "__main__":
    cfg = DQNConfig()
    # 训练
    env,agent = env_agent_config(cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards_cn(rewards, ma_rewards, tag="train", algo=cfg.algo, path=cfg.result_path)

    # 测试
    env,agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)
    rewards,ma_rewards = eval(cfg,env,agent)
    save_results(rewards,ma_rewards,tag='eval',path=cfg.result_path)
    plot_rewards_cn(rewards,ma_rewards,tag="eval",env=cfg.env, algo=cfg.algo, path=cfg.result_path)
