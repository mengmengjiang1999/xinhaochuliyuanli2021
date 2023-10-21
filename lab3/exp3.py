'''
信号处理原理 实验3
陈张萌 2017013678
'''
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape

import scipy.signal

# 时长为1秒
t = 1
# 采样率为60hz
fs = 60
t_split = np.arange(0, t * fs)

# 1hz与25hz叠加的正弦信号
x_1hz = t_split * 1 * np.pi * 2 / fs
x_25hz = t_split * 25 * np.pi * 2 / fs
signal_sin_1hz = np.sin(x_1hz)
signal_sin_25hz = np.sin(x_25hz)

signal_sin = signal_sin_1hz + 0.25 * signal_sin_25hz


# TODO: 补全这部分代码
# 通带边缘频率为10Hz，
# 阻带边缘频率为22Hz，
# 阻带衰减为44dB，窗内项数为17的汉宁窗函数
# 构建低通滤波器
# 函数需要返回滤波后的信号
def filter_fir(input):
    ft = 10
    fz = 22
    sj = 44
    N = 17

    # 理想低通滤波器的截止频率
    fc = ft + fz/2

    # 截止频率的数字频率
    omgc = 2 * np.pi * fc / fs
    # 单位冲激响应函数
    def h(n):
        if n!=0:
            return np.sin(n * omgc)/(n * np.pi)
        else:
            return omgc / np.pi

    # 窗函数表达式
    def w(n):
        return 0.5 + 0.5 * np.cos(2*np.pi*n/(N-1))

    # 滤波器脉冲响应
    def h2(n):
        return h(n-(N-1)/2)*w(n-(N-1)/2)

    print(h2(8))
    print(h(8))

    ht = np.array([h2(i) for i in range(N)])

    # 计算卷积
    result = []
    for i in range(len(input)):
        sum = 0
        for k in range(0,N):
            sum = sum + input[i-k]*ht[k]
        result.append(sum)

    return result


# TODO: 首先正向对信号滤波(此时输出信号有一定相移)
# 将输出信号反向，再次用该滤波器进行滤波
# 再将输出信号反向
# 函数需要返回零相位滤波后的信号
def filter_zero_phase(input):
    result = filter_fir(input)
    result = result[::-1]
    result = filter_fir(result)
    result = result[::-1]
    return result

if __name__ == "__main__":
    delay_filtered_signal = filter_fir(signal_sin)
    zerophase_filtered_signal = filter_zero_phase(signal_sin)

    plt.plot(t_split, signal_sin, label = 'origin')
    plt.plot(t_split, delay_filtered_signal, label = 'fir')
    plt.plot(t_split, zerophase_filtered_signal, label = 'zero phase')

    plt.legend()
    plt.show()
