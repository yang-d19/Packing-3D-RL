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
        """装箱模拟环境

        Args:
            item_size (int, optional): 物体的最大边长. Defaults to 10.
            box_size (int, optional): 容器的大小. Defaults to 32.
        """       

        # 容器的高度，目前设置成比较大的值
        box_height = 100

        self.box_size = box_size
        self.item_pool = self.generate_all_items(item_size)
        self.problem = PackingProblem((box_height, box_size, box_size), self.item_pool)
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
        """生成所有边长小于一定值的长方体

        Args:
            max_size (int): 最大边长

        Returns:
            item_pool (list(Item)): 所有满足条件的长方体的列表
        """        
        item_pool = []

        # 生成所有不大于 max_size^3 的长方体
        for length in range(1, max_size + 1):
            for width in range(1, max_size + 1):
                for height in range(1, max_size + 1):
                    item = Item(np.ones((length, width, height)))
                    item_pool.append(item)

        return item_pool


    # 随机获取一个 batchsize 大小的物体数据
    def get_onebatch_items(self):
        return random.sample(self.item_pool, self.batch_size)
    
    
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
        # 暂定为放在靠近原点的位置
        hm1_padded[:hm1.shape[0], :hm1.shape[1]] = hm1
        hm2_padded[:hm2.shape[0], :hm2.shape[1]] = hm2

        return hm1_padded, hm2_padded
    

    def get_curr_state(self):
        # 取得容器俯视的高度图以及物体俯视和仰视的高度图
        hm_container = self.problem.container.heightmap

        # print(f"in env.PackEnv.get_curr_state(), self.problem.count = {self.problem.count}")
        
        # 在放入最后一个物体之后, count == len(items), 此时返回空的 state
        if self.problem.count >= len(self.problem.items):
            return None

        curr_item: Item = self.problem.items[self.problem.count]
        curr_item.calc_heightmap()

        hm_item_topdown = curr_item.heightmap_topdown
        hm_item_bottomup = curr_item.heightmap_bottomup

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

        max_reward = 10.0

        # 容器中最高的物体的高度
        height_max = self.problem.container.geometry.cube.max()
        # 容器面积
        area = self.problem.container.geometry.x_size * self.problem.container.geometry.y_size
        # 总占据的容积
        total_volume = height_max * area
        # 有效容积
        occupy_volume = self.problem.container.geometry.cube.sum()
        # 已放置物体的占用率（越紧密占用率越高）
        occupancy: float = (total_volume - occupy_volume) / total_volume

        return occupancy * max_reward


    def reset(self):
        """重置环境

        Returns:
            new_state (ndarray): 重置环境后新的状态
        """        
        # 清空容器
        self.problem.container.clear()
        # 重新从物体池中得到一批物体
        self.problem.load_new_items(self.get_onebatch_items())
        # 返回当前的状态数组
        return self.get_curr_state()


    def step(self, action):
        """环境接收到动作之后做出回应

        Args:
            action (int): 0 ~ box_size^2，表示物体放置的位置

        Returns:
            next_state (ndarray): 环境的下一个状态
            reward (float): 这一步动作得到的回报值
            done (bool): 是否所有物体已经摆放完毕
        """        
        # action 是 [0, box_size^2 - 1] 的整数
        # 实际上代表的是放置在 x, y 的哪个位置
        x = action // self.box_size
        y = action % self.box_size

        # 实际可能放不进去，此时状态应该保留原状，但是返回负的 Reward
        result = self.problem.pack(x, y)

        next_state = self.get_curr_state()

        # 如果当前位置能放置，则根据摆放的情况返回正的 reward 值
        if result == True:
            reward = self.calc_reward()
        # 否则返回负值（此时 next_state == state）
        else:
            reward = -5.0
        
        # FIXME: 如果前几个物体摆放得不好，最后几个物体可能放不进去，导致 done 一直为 false
        # 目前的解决方法为将容器的高度设置得很高
        done = (self.problem.count == len(self.problem.items))

        # print("in env.PackEnv.step(), "
        #     f"count = {self.problem.count}, len = {len(self.problem.items)}, done = {done}")
        
        return next_state, reward, done
