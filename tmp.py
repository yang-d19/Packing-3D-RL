import numpy as np
from show import *
import time
import pandas as pd
from utils import *





print(dist(0, 0, 0, 0.5, 0.5, 0.5))





# x = [[[[] for k in range(3)] for j in range(3)] for i in range(4)]
# print(x)

# x[1][2][0].append((1, 3, 2))

# x[2][1][1].append((0, 9, 4))

# # print(x)
# for l1 in x:
#     for l2 in l1:
#         for l3 in l2:
#             print(l3, end=" ")
#         print("")
#     print("\n")


# print(pow(2.5, 2))

# list = [[[(3, 2, 1), (5, 5, 5)], [(2, 3, 2)]], [[(3, 1, 2), (0, 9, 4)], [(1, 2)]], [[(1, 2, 3)], [(2, 3, 4)]]]

# print(list, "\n")

# test = pd.DataFrame(data=list)
# print(test, "\n")
# test.to_csv('test.csv', encoding='utf-8')

# data = pd.read_csv('test.csv')

# array = data.values[0::, 1::]

# print(array)
# print(array.shape)


# X = [[1, 2, 3],[3, 4, 5],[5, 6, 7]]  
# Y = [[4, 4, 3], [0, 1, 7], [3, 1, 6]]

# display = Display()
# display.show2d(X)
# plt.pause(5)

# display.show2d(Y)
# plt.pause(5)

# A = np.zeros((3, 5))
# print(A.shape)

# B = np.matrix([1, 2, 3]).T
# print(B.shape)

# A[:, 1:2] = B 
# print(A)

# t1 = time.time()

# upper_boxsize = 20
# currPointAssemble = np.zeros((3, pow(upper_boxsize, 3)))

# pointCnt = 0
# for z in range(upper_boxsize):
#     for x in range(upper_boxsize):
#         for y in range(upper_boxsize):
#             currPoint = np.mat([x, y, z]).T
#             currPointAssemble[:, pointCnt: pointCnt + 1] = currPoint
#             pointCnt += 1

# t2 = time.time()

# print(t2 - t1)



# x = [4, 3, 2]
# a = np.zeros(tuple(x))
# print(a)

# if a[1][0][0] == 0:
#     print("1")

# a = np.zeros((4, 2, 3))
# print(a)

# x = list(range(3))
# y = list(range(3, 0, -1))
# print(x)
# print(y)

# v = np.mat([1, 2, 3])
# print(v[0, 0])

# x = [[1, 2, 3], [4, 5, 6]]
# z = [1, 2, 3]
# y = [i*2 for i in z]
# print(y)

# from pack import *
# item1 = Item(np.ones((2, 5, 3)))
# print(len(item1.curr_geometry.cube), 
#     len(item1.curr_geometry.cube[0]), 
#     len(item1.curr_geometry.cube[0][0]))

# 

# import matplotlib.pyplot as plt
# import numpy as np
# import time
# from math import *

# plt.ion() #开启interactive mode 成功的关键函数
# plt.figure(1)
# t = [0]
# t_now = 0
# m = [sin(t_now)]

# for i in range(2000):
#     plt.clf() #清空画布上的所有内容
#     t_now = i*0.1
#     t.append(t_now)#模拟数据增量流入，保存历史数据
#     m.append(sin(t_now))#模拟数据增量流入，保存历史数据
#     plt.plot(t,m,'-r')
#     plt.draw()#注意此函数需要调用
#     plt.pause(0.01)

# pt = np.mat([3, 4, 5]).T

# print(pt[0, 0], pt[1, 0])

# from utils import *
# at1 = AttitudeStability(Attitude(1, 2, 3), 2)
# at2 = AttitudeStability(Attitude(3, 5, 3), 5)
# at3 = AttitudeStability(Attitude(4, 6, 1), 1)
# at4 = AttitudeStability(Attitude(2, 7, 6), 4)

# Q = PriorityQueue(at1)
# Q.push(at2)
# Q.push(at3)
# Q.push(at4)

# while not Q.empty():
#     at = Q.pop()
#     print(at.stability, at.attitude.roll, at.attitude.pitch, 
#             at.attitude.yaw)