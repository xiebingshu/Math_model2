# 出租车问题
import numpy as np
import math
from prettytable import PrettyTable
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings
import time
warnings.filterwarnings("ignore")   # 忽略警告信息输出

# mpl.style.use('ggplot')

# 为了画图中文可以正常显示
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei'] #指定默认字体
plt.rcParams['axes.unicode_minus'] = False  #解决保存图像时负号'-'显示为方块的问题

errors1 = []
errors2 = []

def sample(m, total, i):
    samples = np.random.randint(1, total, m)  # 采样
    table_samples.add_row(samples)  # 加入采样表格
    samples.sort()  # 采样结果排列
    result = []
    average = 0
    # 模型1
    for j in range(m):  # 数组求和
        average = average + samples[j]
    average /= m  # 平均数
    result.append(round(float(2 * average - 1), 1))

    # 模型2
    if m % 2 == 0:  # 中位数
        average = (samples[m // 2 - 1] + samples[m // 2]) / 2
    else:
        average = samples[(m + 1) // 2 - 1]
    result.append(round(float(2 * average - 1), 1))

    # 模型3
    result.append(round(samples[0] + samples[m - 1] - 1, 1))  # 两端间隔对称

    # 模型4
    result.append(round((1 + 1 / m) * samples[m - 1] - 1, 1))

    # 模型5
    result.append(round((1 + 1 / (2 * m - 1)) * (samples[m - 1] - 1 / 2 * m), 1))
    table_results.add_row(result)
    result_table.append(result)


def analyze(m, total, k):
    model_average = [0, 0, 0, 0, 0]
    model_error1 = [0, 0, 0, 0, 0]
    model_error2 = [0, 0, 0, 0, 0]
    for j in range(m):  # 读取每次的结果数据
        for i in range(5):
            temp = result_table[j][i]
            model_average[i] += temp
    # 估计均值
    for i in range(5):
        model_average[i] = round(model_average[i] / m, 1)
    table_analysis.add_row(model_average)

    # 估计误差
    min0 = {'num': 10000, 'id': 0}  # 最优模型
    for i in range(5):
        model_error1[i] = round(model_average[i] - total, 1)
        if abs(model_error1[i]) < min0['num']:
            min0['num'] = abs(model_error1[i])
            min0['id'] = i
    table_analysis.add_row(model_error1)

    # 估计标准差
    for j in range(m):  # 方差
        for i in range(5):
            temp = result_table[j][i]
            model_error2[i] += pow((temp - model_average[i]), 2)
    for i in range(5):
        model_error2[i] = model_average[i] / m - 1
        model_error2[i] = round(math.sqrt(model_average[i]), 1)
    errors1.append(model_error1)
    errors2.append(model_error2)
    return min0  # 返回最优模型


def run():
    t = int(input('请输入每轮采样组数:'))  # t为采样组数
    m = [(i + 1) * 10 for i in range(50)]
    model_ana = [0, 0, 0, 0, 0]  # 每个模型在所有实验中胜出次数
    print(
        "实验将采用五种模型:\n" + "模型1:平均值模型\n" + "模型2:中位数模型\n" + "模型3:两端间隔对称模型\n" + "模型4:平均间隔模型\n" + "模型5:区间均分模型")
    for k in range(50):
        temp1 = []
        for i in range(m[k]):
            temp1.append("Sample" + str(i + 1))
        global table_samples  # 将采样表格设为全局变量
        table_samples = PrettyTable(temp1)
        temp1.clear()
        for i in range(5):
            temp1.append("Model" + str(i + 1))
        global table_results  # 将结果表格设为全局变量
        table_results = PrettyTable(temp1)
        global table_analysis  # 将分析表格设为全局变量
        table_analysis = PrettyTable(temp1)
        global result_table
        print("\n接下来是第" + str(k + 1) + "轮实验\n")
        # 初始化三个表格
        table_samples.clear_rows()
        table_analysis.clear_rows()
        table_results.clear_rows()
        result_table = []
        total = 5000
        print("在第" + str(k + 1) + "轮实验中,出租车总数为" + str(total))
        for i in range(t):
            sample(m[k], total, t)  # 开始采样
        print("接下来是采样表格:")
        print(table_samples)
        print("接下来是各模型结果表格:")
        print(table_results)
        min0 = analyze(t, total, k)  # 误差分析
        print("接下来是误差分析表格:")
        print(table_analysis)
        print("\n该组实验中，最优模型为模型" + str(min0['id']+1) + "\n")
        model_ana[min0['id']] += 1
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(m, np.array(errors1).T[0], label="误差_模型1")
    plt.plot(m, np.array(errors1).T[1], label="误差_模型2")
    plt.plot(m, np.array(errors1).T[2], label="误差_模型3")
    plt.plot(m, np.array(errors1).T[3], label="误差_模型4")
    plt.plot(m, np.array(errors1).T[4], label="误差_模型5")
    for i in range(5):
        plt.scatter(m, np.array(errors1).T[i])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.xlabel("每组采样数", fontdict={'size': 16})
    plt.ylabel("数据", fontdict={'size': 16})
    plt.title("模型误差随采样数变化", fontdict={'size': 20})
    plt.savefig(r'./result_picture_trend/picture of error1.jpg', bbox_inches='tight')
    plt.show()
    plt.figure(figsize=(20, 10), dpi=100)
    plt.plot(m, np.array(errors2).T[0], label="标准差_模型1")
    plt.plot(m, np.array(errors2).T[1], label="标准差_模型2")
    plt.plot(m, np.array(errors2).T[2], label="标准差_模型3")
    plt.plot(m, np.array(errors2).T[3], label="标准差_模型4")
    plt.plot(m, np.array(errors2).T[4], label="标准差_模型5")
    for i in range(5):
        plt.scatter(m, np.array(errors2).T[i])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.xlabel("每组采样数", fontdict={'size': 16})
    plt.ylabel("数据", fontdict={'size': 16})
    plt.title("模型标准差随采样数变化", fontdict={'size': 20})
    plt.savefig(r'./result_picture_trend/picture of error2.jpg', bbox_inches='tight')
    plt.show()
if __name__ == '__main__':
    table_samples = PrettyTable()  # 采样表格
    table_results = PrettyTable()  # 结果表格
    table_analysis = PrettyTable()  # 分析表格
    result_table = []  # 结果表格 （上面的三个表格只用于输出，无法存储数据）
    run()
