import random
import numpy as np
from pack import *
from show import *
import time
import queue

def getSurfaceItem(xSize, ySize, zSize):

    cube = np.ones((xSize, ySize, zSize))
    # 将内部全部置为0，只保留表面
    cube[1: xSize-1, 1: ySize-1, 1: zSize-1] = 0

    return Item(cube)

def Task():
    box_size = (40, 40, 40)
  
    # 空心的，只保留表面，计算速度快
    items = [getSurfaceItem(5, 8, 12),
             getSurfaceItem(18, 6, 12),
             getSurfaceItem(10, 10, 9), 
             getSurfaceItem(12, 7, 8),
             getSurfaceItem(12, 8, 5),
             getSurfaceItem(8, 5, 4),
            ]

    # 实心的长方体
    # items = [Item(np.ones((5, 13, 15))),
    #         Item(np.ones((18, 6, 12))),
    #         Item(np.ones((10, 10, 9))), 
    #         Item(np.ones((16, 11, 13))),
    #         Item(np.ones((12, 8, 5))),
    #         Item(np.ones((8, 5, 4)))
    # ]

    problem = PackingProblem(box_size, items)

    # problem.pack_all_items()
    display = Display(box_size)
    
    for idx in range(len(items)):
        problem.autopack_oneitem(idx)   
        display.show3d(problem.container.geometry)
        input()
    

if __name__ == "__main__":

    Task()