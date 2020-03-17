# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:00:30 2020

@author: 85726
"""

"""
思想：保证每一个圆在当前可行域内都取得最大的半径，可取值为圆心到四边的距离和圆心到各个圆边的距离，取其中最小的一个
就是当前最大半径

实现：在保证新取的圆心处于可行域的范围内，使用scipy的优化器，在邻域内不断优化圆心的位置，直至找到最大的半径

结果：
m=10, sum of r^2 =1.14
m=50, sum of r^2 =1.21
m=100, sum of r^2 =1.23
保存的图在unit3中
"""
import unittest
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class unit_test(unittest.TestCase):
    def 


#定义圆类
class circle:
    def __init__(self, radius = 0, x = 0, y = 0):
        self.radius = radius
        self.x = x
        self.y = y
        
    def print_circle(self):
        print('radius={}, coordinate=({},{})'.format(self.radius, self.x, self.y))
    
    #计算两圆心之间距离
    def distance(self, c2):       
        dis = ((self.x-c2.x)**2+(self.y-c2.y)**2)**0.5
        return dis
    
    #判断新圆与现存圆是否相交，相交为0，全不相交为1 
    def ifcross(self, c_list):
        for i in range (len(c_list)):
            c2 = c_list[i]
            r1 = self.radius
            r2 = c2.radius
            rr = r1+r2
            dis = self.distance(c2)
            if dis < rr:
                return 0
        return 1
        
    #判断圆是否越界，越界为0，不越界为1 
    def ifexcess(self):
        r = self.radius
        x = self.x
        y = self.y         
        if x + r > 1 or x - r < -1 or y + r > 1 or y - r < -1:
            return 0
        else:
            return 1

#找出可行的最大半径
def MaxR(c1, c_list):
    x = c1.x
    y = c1.y
    R_list = [1-x,1+x,1-y,1+y]
    for i in range (len(c_list)):
        c2 = c_list[i]
        dis = c1.distance(c2)
        R_list.append(dis-c2.radius)
    return min(R_list)

#需要优化的目标函数        
def func(c_list):
    return lambda x : 1 - MaxR(circle(x[0], x[1], x[2]), c_list)

#找出最优圆心
def opt_center(c, c_list):
    r = c.radius
    x = c.x
    y = c.y
    rxy = [r,x,y]
    bd_r = (0, 1)
    bd_x = (-1, 1)
    bd_y = (-1, 1)
    bds = (bd_r, bd_x, bd_y)       
    res = minimize(func(c_list), rxy, method='SLSQP', bounds=bds)
    c.x = res.x[1]
    c.y = res.x[2]
    c.radius = MaxR(c, c_list)
    return c

#找m个圆，使得每个圆在邻域内半径最大
def FindMaxCircuit(m):
    c_list = []
    for i in range (m):
        r = 0
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        c = circle(r, x, y)
        while not c.ifcross(c_list):           
            x = random.uniform(-1, 1)
            y = random.uniform(-1, 1)
            c = circle(r, x, y)
        c = opt_center(c, c_list)
        c_list.append(c)
    return c_list

def plot(c_list):
    plt.figure()
    plt.axes().set_aspect('equal')
    plt.xlim([-1,1])
    plt.ylim([-1,1])  
    theta = np.linspace(0,2*np.pi,50)
    for c in c_list:
        plt.plot(c.x+c.radius*np.cos(theta),c.y+c.radius*np.sin(theta),'b')       
    plt.show()
    
if __name__ == "__main__":
    m = 10
    c_list = FindMaxCircuit(m)   
    RR = 0
    for c in c_list:
        RR += c.radius**2
        c.print_circle()
    print('for {} circles, the maximize sum of r^2 = {}'.format(m, RR))
    
    plot(c_list)
    
        
        
    
        
            
    
    
    
    
    