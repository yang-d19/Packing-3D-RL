import random

from pack import *

class PackEnv:
    """一个可以供强化学习智能体训练装箱算法使用的环境

    主要由 PackingProblem 类包装而成

    Attributes:
        item_pool: 所有不大于 item_size 的长方体
        problem: PackingProblem 类的实例
        cnt: 已经取出的物体的个数
        batch_size: 一次取出多少物体
        state_space: 状态空间
        action_space: 动作空间 

    """    
    
    def __init__(self, item_size=10, box_size=32):
        
        self.item_pool = self.generate_all_items(item_size)
        self.problem = PackingProblem(box_size, self.item_pool)
        # 已经取出了多少物体
        self.cnt = 0
        self.batch_size = 10
        # 状态空间
        self.state_space = np.zeros((3, box_size, box_size))
        # 动作空间
        self.action_space = np.zeros(pow(box_size, 2))

        self.state_dim = (3, box_size, box_size)
        self.action_dim = pow(box_size, 2)


    def generate_all_items(self, max_size):
        itemPool = []
        # 生成所有不大于 max_size^3 的长方体
        for length in range(1, max_size):
            for width in range(1, max_size):
                for height in range(1, max_size):
                    item = Item(np.ones((length, width, height)))
                    itemPool.append(item)
        # 随机打乱顺序
        return random.shuffle(itemPool)


    # 获取一个 batchsize 大小的物体数据
    def get_onebatch_items(self):        
        batchItems = self.item_pool[self.cnt: self.cnt + self.batch_size]
        self.cnt += self.batch_size
        return batchItems
    
    
    def hm_padding(self, hm1, hm2, target_size):
        """将item的高度图填充为和容器的高度图一样的大小

        暂定为将原先的高度图放在最靠近原点的位置

        Args:
            hm1 (np.ndarray): 自顶向下看的高度图
            hm2 (np.ndarray): 自底向上看的高度图
            target_size (int): 填充后的目标大小

        Returns:
            np.ndarray: 自顶向下高度图的扩充
            np.ndarray: 自底向上高度图的扩充
        """ 

        assert hm1.shape == hm2.shape, "两张高度图的大小不一致"

        hm1_padded = np.zeros((target_size, target_size))
        hm2_padded = np.zeros((target_size, target_size))
        hm1_padded[:hm1.shape[0]][:hm1.shape[1]] = hm1
        hm2_padded[:hm2.shape[0]][:hm2.shape[1]] = hm2

        return hm1_padded, hm2_padded
    

    def get_curr_state(self):
        # 取得容器俯视的高度图以及物体俯视和仰视的高度图
        hm_container = self.problem.container.heightmap
        hm_item_topdown = self.problem.items[self.problem.count].heightmap_topdown
        hm_item_bottomup = self.problem.items[self.problem.count].heightmap_bottomup
        # 填充物体的高度图
        hm_item_td_padded, hm_item_bu_padded = \
            self.hm_padding(hm_item_topdown, hm_item_bottomup, hm_container.shape[0])
        # 组装成一个三通道的状态矩阵
        state = np.array([hm_container, hm_item_td_padded, hm_item_bu_padded])
        return state
    

    def calc_reward(self):
        """根据当前容器的状态计算回报值

        Returns:
            float: 回报值，0 ~ 10
        """        

        max_reward = 10

        # 容器中最高的物体的高度
        height_max = self.problem.container.geometry.cube.max()
        # 容器面积
        area = self.problem.container.geometry.x_size * self.problem.container.geometry.y_size
        # 总占据的容积
        total_volume = height_max * area
        # 有效容积
        occupy_volume = self.problem.container.geometry.cube.sum()

        # 已放置物体的占用率（越紧密占用率越高）
        occupancy = (total_volume - occupy_volume) / total_volume

        return occupancy * max_reward


    def reset(self):
        # 清空容器
        self.problem.container.clear()
        # 重新从物体池中得到一批物体
        self.problem.load_new_items(self.get_onebatch_items())
        # 返回当前的状态数组
        return self.get_curr_state()


    # next_state, reward, done, _
    def step(self, action):
        # action 是 [0, box_size-1] 的整数
        # 实际上代表的是放置在 x, y 的哪个位置
        x = action / self.box_size
        y = action - x

        # 实际可能放不进去，此时状态应该保留原状，但是返回负的 Reward
        result = self.problem.pack(x, y)

        next_state = self.get_curr_state()

        # 如果当前位置能放置，则根据摆放的情况返回正的 reward 值
        if result == True:
            reward = self.calc_reward()
        # 否则返回负值（此时 next_state == state）
        else:
            reward = -5
        
        done = (self.problem.count == len(self.problem.items))
        
        return next_state, reward, done
