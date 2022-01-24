from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import time

from show import Display
from utils import *
from object import *
from container import *


class PackingProblem(object):

    def __init__(self, box_size, items):
        """解决装箱问题的顶层类

        Args:
            box_size (tuple): 箱子的大小，依次为(z, x, y)
            items (list): 待装箱的物体列表
        """        
        # 容器
        self.container = Container(box_size)
        # 待装箱物体的列表
        self.items = items
        # 装箱顺序
        self.sequence = list(range(len(items)))
        # 各物体的旋转矩阵
        self.transforms = list()
        # 当前装了几个物体
        self.count = 0
    

    def load_new_items(self, items):
        """装载一组新的物体

        Args:
            items (list): 新的一组物体
        """        
        self.items = items
        self.sequence = list(range(len(items)))
        self.count = 0


    def pack(self, x, y):
        """在指定坐标放入当前的物体

        Args:
            x (int): x 坐标
            y (int): y 坐标

        Returns:
            bool : True表示可以且已经放入，False表示无法放入
        """

        # 取得当前要放入箱子中的物体
        item: Item = self.items[self.sequence[self.count]]

        # 平移到当前位置
        item.position = Position(x, y, 0)

        # 计算从上往下放物体，物体的坐标
        # 实际上只更改了 z 坐标的值，xy 坐标不变
        # 判断能否放入（受到容器体积的制约可能会失败）
        result = self.container.add_item_topdown(item, x, y)

        if result is True:
            # 真正把物体放入容器中
            self.container.add_item(item)
            self.count += 1
            return True
        
        return False

    
    def autopack_oneitem(self, item_idx):

        # print("itme index: ", item_idx)
        # t1 = time.time()
        curr_item: Item = self.items[item_idx]
        transforms = self.container.search_possible_position(curr_item)
        # t2 = time.time()
        # print("search possible positions: ", t2 - t1)

        # 如果找不到可以放置的位置
        assert len(transforms) > 0, "未找到可以放置的位置和角度"
        
        # 暂时不考虑放置物体后物体堆的稳定性
        # 直接按照排名第一的变换矩阵放置物体
        curr_item.transform(transforms[0])

        # t3 = time.time()
        # print("rotate to a position: ", t3 - t2)
        self.container.add_item(curr_item)

        # t4 = time.time()
        # print("add item to container: ", t4 - t3, '\n')

    
    def autopack_allitems(self):
        for idx in self.sequence:
            self.autopack_oneitem(idx)


if __name__ == "__main__": 
    pass 