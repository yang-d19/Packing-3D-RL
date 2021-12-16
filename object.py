from copy import deepcopy
import math
from queue import PriorityQueue
from typing import Sequence

from utils import *
import numpy as np

class Geometry(object):

    def __init__(self, cube: np.ndarray):
        self.z_size = len(cube)
        self.x_size = len(cube[0])
        self.y_size = len(cube[0][0])      
        self.cube = cube
        self.pointsMat = None
        
        self.get_points()
    
    def get_points(self):
        tmpPoints = []
        # 有多少个点
        pointCnt = 0
        for z in range(self.z_size):
            for x in range(self.x_size):
                for y in range(self.y_size):
                    if self.cube[z][x][y] == 1:
                        tmpPoints.append(np.mat([x, y, z]).T)
                        pointCnt += 1
        
        # 新建行数为3，列数为点总数的 numpy 矩阵，依次填入各点，用于加速旋转矩阵运算
        self.pointsMat = np.zeros((3, pointCnt))
        for idx in range(pointCnt):
            self.pointsMat[:, idx: idx + 1] = tmpPoints[idx]


    
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

    def get_rotate_matrix(self, attitude: Attitude):
        """获取给定姿态的旋转矩阵

        Args:
            attitude (Attitude): 姿态

        Returns:
            np.mat: 旋转矩阵(3×3)
        """
        # roll 是绕 x 轴旋转，pitch 是绕 y 轴旋转，yaw 是绕 z 轴旋转
        # 将角度值转换成弧度制
        alpha = attitude.roll * math.pi / 180
        beta = attitude.pitch * math.pi / 180
        theta = attitude.yaw * math.pi / 180

        # 围绕原点做任意旋转时，所有点都在以如下值的半径的球中
        # radius = math.sqrt(pow(self.x_size, 2) 
        #                 + pow(self.y_size, 2) 
        #                 + pow(self.z_size, 2))
        
        T_roll = np.mat([[1,                 0,                0], 
                         [0,   math.cos(alpha), -math.sin(alpha)],
                         [0,   math.sin(alpha),  math.cos(alpha)]])

        T_pitch = np.mat([[ math.cos(beta),  0,   math.sin(beta)],
                          [              0,  1,                0],
                          [-math.sin(beta),  0,   math.cos(beta)]])

        T_yaw = np.mat([[math.cos(theta), -math.sin(theta),   0],
                        [math.sin(theta),  math.cos(theta),   0],
                        [              0,                0,   1]])

        # 给定旋转的执行顺序，依次是 roll, pitch, yaw
        T_rotate = T_yaw * T_pitch * T_roll
        return T_rotate


    
    def rotate(self, attitude: Attitude):
        """旋转几何体（旧版）

        Args:
            attitude (Attitude): 旋转的目标位置
        """        

        # t_rotateStart = time.time()

        # 围绕原点做任意旋转时，所有点都在以如下值的半径的球中
        radius = math.sqrt(pow(self.x_size, 2) 
                        + pow(self.y_size, 2) 
                        + pow(self.z_size, 2))


        # 加上偏移量保证所有的点的坐标都大于零
        offset = np.mat([radius, radius, radius]).T

        # 给定旋转的执行顺序，依次是 roll, pitch, yaw
        # T_rotate = T_yaw * T_pitch * T_roll
        T_rotate = self.get_rotate_matrix(attitude)

        # 存储变换后的点
        new_points = []

        # 直接旋转变换后的物体有空洞, 因为离散点的映射可能会映射到相同的整数点内
        # 因此在映射时优化，一个点映射到多个目标点

        # 使用预先处理的 self.points 加速矩阵运算
        # 此时的 newPointMat 有正有负
        newPointMat = T_rotate * self.pointsMat

        distThsld = 0.84          # 0.8661

        for idx in range(newPointMat.shape[1]):
            
            # 加上偏置后得到的都是正坐标
            newPoint = newPointMat[:, idx: idx + 1] + offset
            # 得到一个点的坐标（小数）
            [nx, ny, nz] = [newPoint[i, 0] for i in range(3)]
            pxList = [math.floor(nx), math.ceil(nx)]
            pyList = [math.floor(ny), math.ceil(ny)]
            pzList = [math.floor(nz), math.ceil(nz)]
            
            for px in pxList:
                for py in pyList:
                    for pz in pzList:
                        # 计算变换后点到其周围整点的距离
                        ptDist = dist(nx, ny, nz, px, py, pz)
                        # 与 (nx, ny. nz) 距离小于阈值的整点都加入 new_points
                        if ptDist < distThsld:
                            new_points.append(np.mat([px, py, pz]).T)

        min_x = min_y = min_z = math.ceil(radius)
        max_x = max_y = max_z = 0

        # 找所有点各个轴方向的最大和最小值
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

        assert self.x_size > 0 and self.y_size > 0 and self.z_size > 0, \
            print("{} {} {}".format(self.x_size, self.y_size, self.z_size))
        #"物体框架大小不正确"

        # 新建空的 cube
        self.cube = np.zeros((self.z_size, self.x_size, self.y_size))
        # 填充 cube 中的有值部分
        for point in new_points:
            [x, y, z] = [int(point[i, 0]) for i in range(3)]
            self.cube[z][x][y] = 1
        
        # t_afterCreate = time.time()
        # print("Create New Geometry Time: ", t_afterCreate - t_beforeCreate, "\n")

        # t_rotateEnd = time.time()
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
        self.heightmap_topdown = None
        self.heightmap_bottomup = None
    
    # 计算两个高度表
    def calc_heightmap(self):
        self.heightmap_topdown = self.curr_geometry.heightmap_topdown()
        self.heightmap_bottomup = self.curr_geometry.heightmap_bottomup()

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
    