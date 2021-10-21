import numpy as np
from show import *

X = [[1, 2, 3],[3, 4, 5],[5, 6, 7]]  
Y = [[4, 4, 3], [0, 1, 7], [3, 1, 6]]

display = Display()
display.show2d(X)
plt.pause(5)

display.show2d(Y)
plt.pause(5)







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