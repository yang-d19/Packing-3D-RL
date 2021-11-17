from copy import deepcopy
import math
from queue import PriorityQueue

from numpy.lib.function_base import disp
from utils import *
import numpy as np
from show import Display
import matplotlib.pyplot as plt
import time

# display = Display()

# 还没写的东西：
# 判断放置稳定性和可抓取性
# fallback 情形


class Geometry(object):

    def __init__(self, cube):
        self.z_size = len(cube)
        self.x_size = len(cube[0])
        self.y_size = len(cube[0][0])      
        self.cube = cube
    
    def centroid(self):
        #计算物体的质心
        centroid = Position()
        counter = 0
        for z in range(self.z_size):
            for x in range(self.x_size):
                for y in range(self.y_size):
                    if self.cube[z][x][y] == 1:
                        counter += 1
                        centroid += Position(x, y, z)

        centroid /= counter
        return centroid
    

    def stability(self):
        # 初版：只考虑长方体的物体

        # 计算与底面接触的面积
        contact_area = 0
        # 只要与底面距离小于 margin，即认为接触 
        margin = 1
        for x in range(self.x_size):
            for y in range(self.y_size):
                for z in range(margin):
                    if self.cube[z][x][y] > 0:
                        contact_area += 1
                        break
        
        

        return contact_area

    def heightmap_topdown(self):
        heightmap = np.zeros((self.x_size, self.y_size))
        for x in range(self.x_size):
            for y in range(self.y_size):
                # 从顶向下找到第一个非零元素的位置
                max_z = self.z_size - 1
                while max_z >= 0 and self.cube[max_z][x][y] == 0:
                    max_z -= 1
                heightmap[x][y] = max_z + 1
        return heightmap
    
    def heightmap_bottomup(self):
        heightmap = np.zeros((self.x_size, self.y_size))
        for x in range(self.x_size):
            for y in range(self.y_size):
                # 从底向上找到第一个非零元素的位置
                min_z = 0
                while min_z < self.z_size and self.cube[min_z][x][y] == 0:
                    min_z += 1
                heightmap[x][y] = self.z_size - min_z
        return heightmap
    
    #-------------TESTED--------------#
    # 旋转几何体
    def rotate(self, attitude: Attitude):

        t_rotateStart = time.time()

        # roll 是绕 x 轴旋转，pitch 是绕 y 轴旋转，yaw 是绕 z 轴旋转
        # 将角度值转换成弧度制
        alpha = attitude.roll * math.pi / 180
        beta = attitude.pitch * math.pi / 180
        theta = attitude.yaw * math.pi / 180

        # 围绕原点做任意旋转时，所有点都在以如下值的半径的球中
        radius = math.sqrt(pow(self.x_size, 2) 
                        + pow(self.y_size, 2) 
                        + pow(self.z_size, 2))
        
        T_roll = np.mat([[1,                 0,                0], 
                         [0,   math.cos(alpha), -math.sin(alpha)],
                         [0,   math.sin(alpha),  math.cos(alpha)]])

        T_pitch = np.mat([[ math.cos(beta),  0,   math.sin(beta)],
                          [              0,  1,                0],
                          [-math.sin(beta),  0,   math.cos(beta)]])

        T_yaw = np.mat([[math.cos(theta), -math.sin(theta),   0],
                        [math.sin(theta),  math.cos(theta),   0],
                        [              0,                0,   1]])

        # 加上偏移量保证所有的点的坐标都大于零
        offset = np.mat([radius, radius, radius]).T

        # 给定旋转的执行顺序，依次是 roll, pitch, yaw
        T_rotate = T_yaw * T_pitch * T_roll

        # 存储变换后的点
        new_points = []

        # 改进后：逆映射
        # 因为单个旋转矩阵是可逆的，而可逆矩阵的乘积是可逆的
        # 所以可以找到变化的逆映射，遍历映射后空间，查找映射前的值
        upper_boxsize = math.ceil(2 * radius)
        T_rotate_inv = T_rotate.I

        # 逐个点做变换太慢，将所有点组合为一个矩阵，一起乘以旋转的逆映射
        # 组合矩阵的大小为 3 * n^3
        currPointAssemble = np.zeros((3, pow(upper_boxsize, 3)))

        t_beforeAssemble = time.time()

        pointCnt = 0
        for z in range(upper_boxsize):
            for x in range(upper_boxsize):
                for y in range(upper_boxsize):
                    currPoint = np.mat([x, y, z]).T
                    currPointAssemble[:, pointCnt: pointCnt + 1] = currPoint - offset
                    pointCnt += 1
        
        t_afterAssemble = time.time()
        # print("Point Assemble Time: ", t_afterAssemble - t_beforeAssemble, "\n")

        t_beforeMatMul = time.time()
        
        # 很大的矩阵乘法
        originPointAssemble = T_rotate_inv * currPointAssemble

        t_afterMatMul = time.time()
        # print("Matrix Multiply Time: ", t_afterMatMul - t_beforeMatMul, "\n")


        t_beforeCreate = time.time()

        # 遍历逆变换后点集的所有列
        for idx in range(originPointAssemble.shape[1]):
            [ox, oy, oz] = [round(originPointAssemble[i, idx]) for i in range(3)]
            # 原坐标不在物体框架中，直接跳过
            if ox < 0 or ox >= self.x_size \
                or oy < 0 or oy >= self.y_size \
                or oz < 0 or oz >= self.z_size:
                continue
            if self.cube[oz][ox][oy] > 0:
                new_points.append(currPointAssemble[:, idx: idx + 1] + offset)


        min_x = min_y = min_z = math.ceil(radius)
        max_x = max_y = max_z = 0

        for point in new_points:
            min_x = min(min_x, point[0, 0])
            min_y = min(min_y, point[1, 0])
            min_z = min(min_z, point[2, 0])

            max_x = max(max_x, point[0, 0])
            max_y = max(max_y, point[1, 0])
            max_z = max(max_z, point[2, 0])
        
        # 使物体的框架紧贴着坐标系的 “墙角”
        for point in new_points:
            point -= np.mat([min_x, min_y, min_z]).T
        
        # 旋转变换后的物体的框架的大小
        self.x_size = round(max_x - min_x + 1)
        self.y_size = round(max_y - min_y + 1)
        self.z_size = round(max_z - min_z + 1)

        # 新建空的 cube
        self.cube = np.zeros((self.z_size, self.x_size, self.y_size))
        # 填充 cube 中的有值部分
        for point in new_points:
            [x, y, z] = [int(point[i, 0]) for i in range(3)]
            self.cube[z][x][y] = 1
        
        t_afterCreate = time.time()
        # print("Create New Geometry Time: ", t_afterCreate - t_beforeCreate, "\n")

        t_rotateEnd = time.time()
        # print("Rotat Run Time: ", t_rotateEnd - t_rotateStart, "\n\n")


    def add(self, geom, position: Position, coef=1):
        # x, y, z 为添加的小物体中每个点的坐标
        for z in range(geom.z_size):
            for x in range(geom.x_size):
                for y in range(geom.y_size):
                    # nx, ny, nz 为大物体中的对应坐标
                    nz = z + position.z
                    nx = x + position.x
                    ny = y + position.y
                    # 如果超出了范围
                    if nz >= self.z_size \
                        or nx >= self.x_size \
                        or ny >= self.y_size:
                        continue
                    # 大几何体中添加小几何体的有值部分，其余部分保留原值
                    if geom.cube[z][x][y] > 0:
                        self.cube[nz][nx][ny] = geom.cube[z][x][y] * coef



class Item(object):

    def __init__(self, cube, position: Position = Position(), 
                    attitude: Attitude = Attitude()):
        self.init_geometry = Geometry(cube)
        self.curr_geometry = Geometry(cube)

        self.position = position
        self.attitude = attitude
        self.heightmap_top = None
        self.heightmap_bottom = None
    
    # 计算两个高度表
    def calc_heightmap(self):
        self.heightmap_top = self.curr_geometry.heightmap_topdown()
        self.heightmap_bottom = self.curr_geometry.heightmap_bottomup()

    # 旋转 init_geometry 一定角度得到 curr_geometry
    def rotate(self, attitude: Attitude):
        self.curr_geometry = Geometry(self.init_geometry.cube)
        self.curr_geometry.rotate(attitude)
        self.attitude = attitude

    # 包括旋转和平移的变换
    def transform(self, transform: Transform):
        self.rotate(transform.attitude)
        self.position = transform.position

    # 获取具有平面稳定性的物体姿态
    def planar_stable_attitude(self, step_width):

        # 取稳定性最高的前 6 个姿态
        stable_attitudes_score = PriorityQueue()

        # 遍历所有的翻滚角 roll 和俯仰角 pitch
        for roll in range(0, 360, step_width):
            for pitch in range(0, 360, step_width):
                # 当前的姿态参数
                curr_attitude = Attitude(roll, pitch, 0)
                self.curr_geometry = Geometry(self.init_geometry.cube)
                self.curr_geometry.rotate(curr_attitude)
                # 计算当前姿态对应的稳定性
                stabilty = self.curr_geometry.stability()

                # ------DEBUG BEGIN------
                # print("roll: ", roll, "    pitch: ", pitch)
                # print("stability: ", stabilty)
                # display = Display([15, 15, 15])
                # display.show(self.curr_geometry)
                # input()
                # -------DEBUG END-------

                # 加入优先队列中排序
                stable_attitudes_score.put(AttitudeStability(curr_attitude, stabilty))

        # 去掉稳定性数值，只保留姿态 
        # 取稳定性最高的前 6 个姿态
        stable_attitudes = []
        cnt = 0
        while not stable_attitudes_score.empty() and cnt < 6:
            cnt += 1
            attitude_score = stable_attitudes_score.get()
            # print(attitude_score)
            stable_attitudes.append(attitude_score.attitude)
        
        return stable_attitudes
    

class Container(object):

    def __init__(self, box_size):
        
        # 创建空的几何体
        self.geometry = Geometry(np.zeros(tuple(box_size)))
        # 计算高度表
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
                    if item.heightmap_top[i][j] > 0:
                        new_heightmap[x + i][y + j] = z + item.heightmap_top[i][j]
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



    def search_possible_position(self, item: Item, grid_num=10, step_width=45):
        
        # print("x_size={}  y_size={}  z_size={}\n"\
        #     .format(item.curr_geometry.x_size, item.curr_geometry.y_size, item.curr_geometry.z_size))

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
                    # 如果物体在平面维度不能放入容器中，直接跳过
                    if x + item.curr_geometry.x_size > self.geometry.x_size \
                        or y + item.curr_geometry.y_size > self.geometry.y_size:
                        continue
                    # 计算在这个位置放置该物体的上表面的 z 坐标
                    item_upper_z = 0
                    for i in range(item.curr_geometry.x_size):
                        for j in range(item.curr_geometry.y_size):
                            item_upper_z = max(item_upper_z, 
                                            self.heightmap[x + i][y + j] + item.heightmap_bottom[i][j])
                    # 如果上表面超出了容器的上界，跳过
                    if item_upper_z > self.geometry.z_size:
                        continue
                    # 物体的 z 坐标（特指物体所在三维体的原点的 z 坐标）
                    z = round(item_upper_z - item.curr_geometry.z_size)

                    # 确定物体的坐标
                    item.position = Position(x, y, z)
                    # 计算当前位置的分数
                    score = self.hueristic_score(item, centroid, "DBLF")

                    curr_position = Position(x, y, z)
                    curr_transform = Transform(curr_position, curr_attitude)

                    tf_score = TransformScore(curr_transform, score)

                    # print("temp node: \n", tf_score)

                    stable_transforms_score.put(tf_score)

            t4 = time.time()
            print("try every yaw in this attitude: ", t4 - t3, '\n')

                # print("\nAfter each XY in priority queue: \n")
                # cnt = 0
                # for ele in stable_transforms_score:
                #     cnt += 1
                #     if cnt > 10:
                #         break
                #     print(ele)
                # print("\n")
            
            # cnt = 0
            # print("\nAfter each Attitude in priority queue: \n")
            # for ele in stable_transforms_score:
            #     cnt += 1
            #     if cnt > 10:
            #         break
            #     print(ele)
            # print("\n")
                    
        
        # print("\n After All priority queue: \n\n\n")

        # 去掉 score ，只保留 transform
        # 取前10个
        stable_transforms = []
        cnt = 0
        while not stable_transforms_score.empty() and cnt < 10:
            cnt += 1
            transform_score = stable_transforms_score.get()
            # print(transform_score)
            stable_transforms.append(transform_score.transform)

        # print("")
        
        return stable_transforms


class PackingProblem(object):

    def __init__(self, box_size, items):
        # box_size 的 3 个元素依次为 z, x, y
        self.container = Container(box_size)
        self.items = items
        self.number = len(items)
        self.sequence = list(range(self.number))
        self.transforms = list()
    
    def pack_one_item(self, item_idx):

        print("itme index: ", item_idx, '\n')

        t1 = time.time()

        curr_item: Item = self.items[item_idx]
        transforms = self.container.search_possible_position(curr_item)

        t2 = time.time()
        print("search possible positions: ", t2 - t1, '\n')

        # 如果找不到可以放置的位置
        assert len(transforms) > 0, "未找到可以放置的位置和角度"

        # 暂时不考虑放置物体后物体堆的稳定性
        # 直接按照排名第一的变换矩阵放置物体
        curr_item.transform(transforms[0])

        t3 = time.time()
        print("rotate to a position: ", t3 - t2, '\n')

        self.container.add_item(curr_item)

        t4 = time.time()
        print("add item to container: ", t4 - t3, '\n\n')

    
    def pack_all_items(self):
        for idx in self.sequence:
            self.pack_one_item(idx)


if __name__ == "__main__": 
    pass 