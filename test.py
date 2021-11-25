import random
import numpy as np
from pack import *
from show import *
import time
import queue

# 生成随机大小的长方体
# def generate_random_item(size_range):
#     item_size = [random.randrange(size_range[0], size_range[1]) 
#                     for i in range(3)]
#     new_cube = np.ones(tuple(item_size))
#     return Item(new_cube)

# def generate_random_geom(size_range):
#     item_size = [random.randrange(size_range[0], size_range[1]) 
#                     for i in range(3)]
#     new_cube = np.ones(tuple(item_size))
#     return Geometry(new_cube)

# def generate_random_items(num, size_range):
#     items = []
#     for i in range(num):
#         items.append(generate_random_item(size_range))
#     return items

def getSurfaceItem(xSize, ySize, zSize):

    cube = np.ones((xSize, ySize, zSize))
    # 将内部全部置为0，只保留表面
    cube[1: xSize-1, 1: ySize-1, 1: zSize-1] = 0

    return Item(cube)


def T0():
    display = Display([20, 20, 20])
    geom = Geometry(np.ones((12, 8, 14)))
    display.show3d(geom)
    input()
    # plt.pause(3)
    # t1 = time.time()
    geom.rotate(Attitude(45, 0, 0))
    # t2 = time.time()
    # print(t2 - t1)
    display.show3d(geom)
    # t1 = time.time()
    # print(t1 - t2)
    input()

    geom.rotate(Attitude(0, 45, 0))
    display.show3d(geom)
    input()

    geom.rotate(Attitude(0, 0, 120))
    display.show3d(geom)
    input()


def T1():
    box_size = [10, 10, 10]
  
    # items = generate_random_items(5, [5, 10])
    items = [Item(np.ones((5, 3, 6))), 
             Item(np.ones((4, 2, 3))),
             Item(np.ones((2, 5, 3))),
             Item(np.ones((5, 4, 2))),
             Item(np.ones((4, 4, 4))),
             Item(np.ones((3, 6, 3))),
             Item(np.ones((5, 3, 2))),
             Item(np.ones((3, 7, 2))),
             Item(np.ones((3, 3, 2)))
            ]

    problem = PackingProblem(box_size, items)

    # problem.pack_all_items()
    display = Display(box_size)
    
    for idx in range(len(items)):
        problem.pack_one_item(idx)   
        display.show3d(problem.container.geometry)
        input()

def T2():
    geom = Geometry(np.ones((6, 4, 8)))
    geom.calc_centroid()
    print(geom.centroid.x, geom.centroid.y, geom.centroid.z)

def T3():
    # tf = Transform(Position(1, 2, 3), Attitude(30, 0, 45))
    # print(tf)
    x = AttitudeStability(Attitude(30, 25, 55), 4.3)
    print(x)

    y = TransformScore(Transform(Position(2, 5, 4), Attitude(42, 99, 3)), 8.3)
    print(y)

    z = Transform(Position(2, 5, 4), Attitude(42, 99, 3))
    print(z)

def T4():
    Q = queue.PriorityQueue()
    Q.put(TransformScore(Transform(Position(1, 2, 0), Attitude(30, 0, 60)), 3.2))
    Q.put(TransformScore(Transform(Position(3, 2, 1), Attitude(40, 50, 60)), 2.0))
    Q.put(TransformScore(Transform(Position(5, 2, 3), Attitude(0, 40, 25)), 4.9))
    Q.put(TransformScore(Transform(Position(1, 1, 1), Attitude(15, 90, 30)), 0.7))
    Q.put(TransformScore(Transform(Position(8, 5, 5), Attitude(90, 30, 0)), 5.4))
    while not Q.empty():
        print(Q.get())
    Q.get()

def T5():
    box_size = [40, 40, 40]
  
    # items = generate_random_items(5, [5, 10])
    items = [getSurfaceItem(5, 13, 15),
             getSurfaceItem(18, 6, 12),
             getSurfaceItem(10, 10, 9), 
             getSurfaceItem(16, 11, 13),
             getSurfaceItem(12, 8, 5),
             getSurfaceItem(8, 5, 4),
            ]
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
        problem.pack_one_item(idx)   
        display.show3d(problem.container.geometry)
        input()
    

if __name__ == "__main__":

    T5()
    

    # display.show3d(problem.container.geometry)
    # problem.container.search_possible_position(items[0])
    # display.show3d(problem.container.geometry)
    # input()





# item1 = Item(np.ones((2, 5, 3)))
# self.container.add_item(item1)
# self.display.update(self.container.geometry)
# # self.display.show()
# # plt.pause(2)

# item2 = Item(np.ones((3, 3, 3)), position=Position(0, 0, 3))
# self.container.add_item(item2)
# self.display.update(self.container.geometry)
# # self.display.show()
# # plt.pause(2)

# item3 = Item(np.ones((4, 2, 2)), position=Position(5, 2, 0))
# self.container.add_item(item3)
# self.display.update(self.container.geometry)
# self.display.show()
# plt.pause(1000)

