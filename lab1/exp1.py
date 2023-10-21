import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np
import os
import imageio
from matplotlib.pyplot import cm

from math import pi, sqrt

# from sympy import *
# x = symbols('x')
# print(integrate(x, (x, 1, 2)))

# TODO: 1. Change N_Fourier to 2, 4, 8, 16, 32, 64, 128, get visualization results with differnet number of Fourier Series
N_Fourier = 128

# TODO: optional, implement visualization for semi-circle
# signal_name = "square"
signal_name = "semicircle"

# TODO: 2. Please implement the function that calculates the Nth fourier coefficient
# Note that n starts from 0
# For n = 0, return a0; n = 1, return b1; n = 2, return a1; n = 3, return b2; n = 4, return a2 ...
# n = 2 * m - 1(m >= 1), return bm; n = 2 * m(m >= 1), return am. 

# 修改函数定义
def fourier_coefficient(n):
    if signal_name == "square":
        return square_fourier_coefficient(n)
    elif signal_name == "semicircle":
        return semi_circle_fourier_coefficient(n)
    else:
        raise Exception("Unknown Signal")

# semicircle中的积分计算
from scipy import integrate
def f(x, n):
    return sqrt(1-(x-1)*(x-1))*np.cos(n * pi * x)

def jifen(n):
    v, err = integrate.quad(f, 0, 1, args =(n))
    return v

def semi_circle_fourier_coefficient(n):
    if n==0: #a0
        return (pi*pi)/4
    elif n%2==1: #bn
        return 0
    else: #an
        return 2 * pi * jifen(n/2)

def square_fourier_coefficient(n):
    if n==0: #a0
        return 1/2;
    elif n%2==0: #an
        return 0
    elif (n+1)%4==0: #bn，n为偶数
        return 0
    else: #bn，n为奇数
        return 2/(((n+1)/2) * pi)

# TODO: 3. implement the signal function
def square_wave(t):
    if t>=0 and t<pi:
        return 1
    else:
        return 0

# TODO: optional. implement the semi circle wave function
def semi_circle_wave(t):
    return sqrt(pi*pi-(t-pi)*(t-pi))

def function(t):
    if signal_name == "square":
        return square_wave(t)
    elif signal_name == "semicircle":
        return semi_circle_wave(t)
    else:
        raise Exception("Unknown Signal")


def visualize():
    if not os.path.exists(signal_name):
        os.makedirs(signal_name)

    frames = 100

    # x and y are for drawing the original function
    x = np.linspace(0, 2 * math.pi, 1000)
    y = np.zeros(1000, dtype = float)
    for i in range(1000):
        y[i] = function(x[i])

    for i in range(frames):
        figure, axes = plt.subplots()
        color=iter(cm.rainbow(np.linspace(0, 1, 2 * N_Fourier + 1)))

        time = 2 * math.pi * i / 100
        point_pos_array = np.zeros((2 * N_Fourier + 2, 2), dtype = float)
        radius_array = np.zeros((2 * N_Fourier + 1), dtype = float)

        point_pos_array[0, :] = [0, 0]
        radius_array[0] = fourier_coefficient(0)
        point_pos_array[1, :] = [0, radius_array[0]]

        circle = patches.Circle(point_pos_array[0], radius_array[0], fill = False, color = next(color))
        axes.add_artist(circle)

        f_t = function(time)
        for j in range(N_Fourier):
            # calculate circle for a_{n}
            radius_array[2 * j + 1] = fourier_coefficient(2 * j + 1)
            point_pos_array[2 * j + 2] = [point_pos_array[2 * j + 1][0] + radius_array[2 * j + 1] * math.cos((j + 1) * time),   # x axis
                                        point_pos_array[2 * j + 1][1] + radius_array[2 * j + 1] * math.sin((j + 1) * time)]     # y axis
            circle = patches.Circle(point_pos_array[2 * j + 1], radius_array[2 * j + 1], fill = False, color = next(color))
            axes.add_artist(circle)
            
            # calculate circle for b_{n}
            radius_array[2 * j + 2] = fourier_coefficient(2 * j + 2)
            point_pos_array[2 * j + 3] = [point_pos_array[2 * j + 2][0] + radius_array[2 * j + 2] * math.sin((j + 1) * time),   # x axis
                                        point_pos_array[2 * j + 2][1] + radius_array[2 * j + 2] * math.cos((j + 1) * time)]     # y axis
            circle = patches.Circle(point_pos_array[2 * j + 2], radius_array[2 * j + 2], fill = False, color = next(color))
            axes.add_artist(circle)
            
        plt.plot(point_pos_array[:, 0], point_pos_array[:, 1], 'o-')
        plt.plot(x, y, '-')
        plt.plot([time, point_pos_array[-1][0]], [f_t, point_pos_array[-1][1]], '-', color = 'r')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.savefig(os.path.join(signal_name, "{}.png".format(i)))
        plt.show()
        plt.close()
        
    images = []
    for i in range(frames):
        images.append(imageio.imread(os.path.join(signal_name, "{}.png".format(i))))
    imageio.mimsave('{}.mp4'.format(signal_name), images)


if __name__ == "__main__":
    visualize()