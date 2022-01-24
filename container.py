from copy import deepcopy
from queue import PriorityQueue
import math
import time

from utils import *
from object import *

class Container(object):

    def __init__(self, box_size):
        """提供放置物体操作的容器类

        Args:
            box_size (tuple): 箱子的大小 (z, x, y)
        """        
        self.boxSize = box_size
        # 创建空的几何体
        self.geometry = Geometry(np.zeros(tuple(box_size)))
        # 计算高度表
        self.heightmap = self.geometry.heightmap_topdown()
        # 当前是放进来的第几个物体（为了用不同的颜色区分）
        self.number = 0


    # 清空容器
    def clear(self):
        self.geometry = Geometry(np.zeros(tuple(self.boxSize)))
        self.heightmap = self.geometry.heightmap_topdown()
        self.number = 0

    
    # 由给定的启发函数计算对应的分数
    def hueristic_score(self, item: Item, centroid, method):
        # 启发函数计算时使用到的小常数
        c = 0.5
        # item.position 只是物体框架的最靠近原点的坐标
        # 还要计算物体质心相对于物体原点的坐标
        # 然后两者相加，得到物体在容器中的实际质心
        true_centroid = item.position + centroid
        x = true_centroid.x
        y = true_centroid.y
        z = true_centroid.z

        if method == "DBLF":
            return z + c * (x + y)

        elif method == "HM":
            # 拷贝一份旧的高度图
            new_heightmap = deepcopy(self.heightmap)
            # 计算添加该物体后的容器的高度图
            for i in range(item.curr_geometry.x_size):
                for j in range(item.curr_geometry.y_size):
                    if item.heightmap_topdown[i][j] > 0:
                        new_heightmap[x + i][y + j] = z + item.heightmap_topdown[i][j]
            # 由高度图计算启发函数定义的分数
            score = c * (x + y)
            for i in range(self.geometry.x_size):
                for j in range(self.geometry.y_size):
                    score += new_heightmap[i][j]
            return score

        else:
            return -1


    def add_item(self, item: Item):
        self.number += 1
        # 计算新的点云模型
        self.geometry.add(item.curr_geometry, item.position, self.number)
        # 重新计算高度图
        self.heightmap = self.geometry.heightmap_topdown()


    def add_item_topdown(self, item: Item, x, y):
        """从上往下放置物体

        Args:
            item (Item): 当前要放入的物体
            x (int): 放入的x坐标
            y (int): 放入的y坐标

        Returns:
            bool: True表示能够放入容器中，False表示不能
        """        

        assert type(x) == int and type(y) == int \
            and x >= 0 and y >= 0, "x, y 必须为正整数"

        # 如果物体在平面维度不能放入容器中
        if x + item.curr_geometry.x_size > self.geometry.x_size \
            or y + item.curr_geometry.y_size > self.geometry.y_size:
            return False

        # 计算在这个位置放置该物体的上表面的 z 坐标
        item_upper_z = 0
        for i in range(item.curr_geometry.x_size):
            for j in range(item.curr_geometry.y_size):
                item_upper_z = max(item_upper_z, 
                                self.heightmap[x + i][y + j] + item.heightmap_bottomup[i][j])

        # 如果上表面超出了容器的上界
        if item_upper_z > self.geometry.z_size:
            return False

        # 物体的 z 坐标（特指物体所在三维体的原点的 z 坐标）
        z = round(item_upper_z - item.curr_geometry.z_size)

        # 确定物体的坐标（引用传递，直接作用到传入的参数上）
        item.position = Position(x, y, z)

        return True


    def search_possible_position(self, item: Item, grid_num=10, step_width=45):
        
        # 存放所有可能的变换矩阵，
        # stable_transforms_score = PriorityQueue(TransformScore(score=10000))
        stable_transforms_score = PriorityQueue()

        # 将容器划分为 grid_num * grid_num 个网格
        # 对于每个网格，尝试放下物体
        grid_coords = []
        for i in range(grid_num):
            for j in range(grid_num):
                x = math.floor(self.geometry.x_size * i / grid_num)
                y = math.floor(self.geometry.y_size * j / grid_num)
                grid_coords.append([x, y])

        t1 = time.time()

        # step_width 为遍历 roll, pitch, yaw 的步长
        # 预处理：提前找到一些比较稳定的 roll, pitch
        # yaw 不影响物体放在平面上的稳定性
        stable_attitudes = item.planar_stable_attitude(step_width)

        t2 = time.time()
        print("find state attitudes: ", t2 - t1)

        # for att in stable_attitudes:
        #     print(att)

        attCnt = 0
        # 遍历比较稳定的姿态（不包括 yaw）
        for part_attitude in stable_attitudes:

            t3 = time.time()

            # 对每一组 roll, pitch 遍历 yaw
            for yaw in range(0, 360, step_width):
                # 生成完整的姿态（包括 yaw）
                curr_attitude = Attitude(part_attitude.roll, part_attitude.pitch, yaw)
                # 生成旋转后的物体
                item.rotate(curr_attitude)
                # 获取该物体自顶向下和自底向上的高度图
                item.calc_heightmap()
                # 计算当前物体的质心
                centroid = item.curr_geometry.centroid()

                # --------- DEBUG BEGIN -----------
                # print(centroid)
                # print(curr_attitude)
                # display.show3d(item.curr_geometry)
                # input()
                # ---------- DEBUG END ------------

                # 遍历网格交点
                for [x, y] in grid_coords:

                    # 尝试在 (x, y) 位置放置物体
                    if self.add_item_topdown(item, x, y):
                        # 如果当前位置能放入容器中
                        # 计算当前位置的分数
                        score = self.hueristic_score(item, centroid, "DBLF")

                        curr_position = item.position
                        curr_transform = Transform(curr_position, curr_attitude)
                        # 组合起来，为排序做准备
                        tf_score = TransformScore(curr_transform, score)
                        stable_transforms_score.put(tf_score)

                    # 否则直接跳过
                    else:
                        continue

            t4 = time.time()
            print("try number {} yaw in this attitude: {}".format(attCnt, t4 - t3))
            attCnt += 1

        # 去掉 score ，只保留 transform
        # 取前10个
        stable_transforms = []
        cnt = 0
        while not stable_transforms_score.empty() and cnt < 10:
            cnt += 1
            transform_score = stable_transforms_score.get()

            stable_transforms.append(transform_score.transform)
        
        return stable_transforms
    

