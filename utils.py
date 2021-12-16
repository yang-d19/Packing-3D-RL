from operator import pos
import queue
from copy import deepcopy
import math

# 使用排序数组实现的优先队列
# class PriorityQueue(object):

#     def __init__(self, node):
#         self._queue = sortedcontainers.SortedList([node])

#     def push(self, node):
#         self._queue.add(node)
#         # # 控制优先队列的长度
#         # if len(self._queue) > self.max_len:
#         #     self._queue.pop(index=len(self._queue)-1)

#     def pop(self):
#         return self._queue.pop(index=0)

#     def empty(self):
#         return len(self._queue) == 0

def dist(x1, y1, z1, x2, y2, z2):
    return math.sqrt((x1-x2) * (x1-x2) + (y1-y2) * (y1-y2) + (z1-z2) * (z1-z2))

class Position:

    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def set(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, rhs):
        return Position(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)

    def __sub__(self, rhs):
        return Position(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    
    def __truediv__(self, rhs):
        return Position(self.x / rhs, self.y / rhs, self.z / rhs)

    def __floordiv__(self, rhs):
        return Position(self.x // rhs, self.y // rhs, self.z // rhs)
    
    def __repr__(self):
        return "<Position: x={}  y={}  z={}>" \
                .format(self.x, self.y, self.z)


class Attitude:

    def __init__(self, roll=0, pitch=0, yaw=0):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
    
    def set(self, roll, pitch, yaw):
        self.roll = roll
        self.pitch = pitch
        self.yaw = yaw
    
    def __repr__(self):
        return "<Attitude: roll={}  pitch={}  yaw={}>" \
                .format(self.roll, self.pitch, self.yaw)


class Transform:

    def __init__(self, position = Position(), attitude = Attitude()):
        self.position = position
        self.attitude = attitude

    def __repr__(self):
        return "<Transform:\n  {}\n  {}\n>".format(self.position, self.attitude)


class AttitudeStability:

    def __init__(self, attitude = Attitude(), stability = 0):
        self.attitude = attitude
        self.stability = stability
    
    def __lt__(self, other):
        return self.stability > other.stability

    def __eq__(self, other):
        return self.stability == other.stability
    
    def __repr__(self):
        return "<AttitudeStability:\n  {}\n  stability={}\n>"\
                .format(self.attitude, self.stability)

class TransformScore:
    
    def __init__(self, transform = Transform(), score = 0):
        self.transform = transform
        self.score = score
    
    def __lt__(self, other):
        return self.score < other.score

    def __eq__(self, other):
        return self.score == other.score
    
    def __repr__(self):
        return "<TransformScore:\n  <Transform:\n    {}\n    {}\n  >\n  score={}\n>" \
                .format(self.transform.position, self.transform.attitude, self.score)