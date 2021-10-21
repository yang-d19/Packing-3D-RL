import math
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Display:
    
    def __init__(self, space_size=[10, 10, 10]):

        plt.ion() #开启interactive mode 成功的关键函数
        # 设置三维视图显示
        self.fig = plt.figure() 
             
        self.space_size = space_size
        
        # 每个物体使用不同的颜色
        self.colors = ['lightcoral', 'lightsalmon', 'gold', 'olive',
            'mediumaquamarine', 'deepskyblue', 'blueviolet', 'pink',
            'brown', 'darkorange', 'yellow', 'lawngreen', 'turquoise',
            'dodgerblue', 'darkorchid', 'hotpink', 'firepink', 'peru',
            'orange', 'darksage', 'cyan', 'purple', 'crimson']

        # 基础的小立方体
        self.origin_voxel = [[[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0]],
                             [[0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0]],
                             [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
                             [[0, 0, 1], [0, 0, 0], [0, 1, 0], [0, 1, 1]],
                             [[0, 1, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]],
                             [[0, 1, 1], [0, 0, 1], [1, 0, 1], [1, 1, 1]]]

    def set_size(self, space_size):
        self.space_size = space_size

    def set_ax3d(self):

        self.ax = self.fig.add_subplot(projection='3d')  
        self.ax.set_aspect('auto')
        self.ax.set_xlim([0, self.space_size[1]])
        self.ax.set_ylim([0, self.space_size[2]])
        self.ax.set_zlim([0, self.space_size[0]])

        # z 轴比例过小，导致高度看起来小了
        # 拉伸 z 坐标轴
        self.ax.get_proj = lambda: np.dot(Axes3D.get_proj(self.ax), np.diag([1, 1, 1.3, 1]))

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')
        self.ax.set_zlabel('z')

        self.ax.view_init(50, 45)
    
    def set_ax2d(self):

        self.ax = self.fig.add_subplot()  

        self.ax.set_xlabel('x')
        self.ax.set_ylabel('y')


    def one_voxel(self, offset):
        new_voxel = self.origin_voxel + np.array(offset)
        return new_voxel
    
    def show3d(self, geom):

        # 清空画布上的所有内容
        plt.clf() 
        # 重新设置画布上的图像信息
        self.set_ax3d()

        for x in range(geom.x_size):
            for y in range(geom.y_size):
                for z in range(geom.z_size):
                    if geom.cube[z][x][y] > 0:

                        color_idx = math.floor(geom.cube[z][x][y])
                        
                        pos = (x, y, z)
                        voxel = self.one_voxel(pos)

                        self.ax.add_collection3d(Poly3DCollection(verts=voxel, 
                                                    facecolors=self.colors[color_idx]))

        plt.draw()
    
    def show2d(self, mat):

        plt.clf()
        self.set_ax2d()
        bar = self.ax.imshow(mat)
        plt.colorbar(bar, ax=self.ax)
        plt.draw()
