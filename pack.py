from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import time

from show import Display
from utils import *
from object import *
from container import *

# 解决装箱问题的顶层类
# 可自定义元素包括一个容器、数个物体、装箱顺序
# 接口：
# pack_one_item() 装入一个物体
# pack_all_items() 装入所有物体
class PackingProblem(object):

    def __init__(self, box_size, items):
        # box_size 的 3 个元素依次为 z, x, y
        self.container = Container(box_size)
        self.items = items
        self.sequence = list(range(len(items)))
        self.transforms = list()
        self.count = 0
    
    # next_state, reward, done,
    def pack(self, x, y):
        # 取得当前要放入箱子中的物体
        item: Item = self.items[self.sequence[self.count]]
        # 平移到当前位置
        item.position = Position(x, y, 0)

        # 计算从上往下放物体，物体的坐标
        # 实际上只更改了 z 坐标的值，xy 坐标不变
        self.container.add_item_topdown()
        # 真正把物体放入容器中
        self.container.add_item(item)
        
        self.count += 1

    
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