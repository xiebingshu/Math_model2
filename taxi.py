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
    table_analysis.add_row(model_error2)
    labels = ['均值', '误差', '标准差']
    model_labels = ["模型" + str(i+1) for i in range(5)]
    bar_width = 0.28
    plt.figure(figsize=(10, 7))
    bar0 = plt.bar(x=np.arange(5), height=model_average, width=bar_width, label=labels[0], color='r', tick_label=model_labels)
    bar1 = plt.bar(x=np.arange(5) + bar_width, height=model_error1, width=bar_width , label=labels[1])
    bar2 = plt.bar(x=np.arange(5) + 2 * bar_width, height=model_error2, width=bar_width , label=labels[2])
    plt.bar_label(bar0)
    plt.bar_label(bar1)
    plt.bar_label(bar2)

    # 补充标题及标签
    plt.title('各模型对比')  # 图的标题
    plt.xlabel('模型', fontsize=15)  # 横轴标签5

    plt.ylabel('效果', fontsize=15)  # 纵轴标签
    plt.xticks(np.arange(5) + 0.17, model_labels, fontsize=12)  # 柱状图横轴坐标各类别标签
    plt.legend()  # 显示两组柱状图的标签
    plt.savefig(r'./result_picture_bar/picture of round' + str(k+1) + '.jpg', bbox_inches='tight')
    # 显示图像
    plt.show()
    return min0  # 返回最优模型


def run():
    n = int(input('请输入出租车实验轮数:'))  # n为实验轮数
    t = int(input('请输入每轮采样组数:'))  # t为采样组数
    m = int(input('请输入每组采样数:'))  # m为每组采样数
    model_ana = [0, 0, 0, 0, 0]  # 每个模型在所有实验中胜出次数
    print(
        "实验将采用五种模型:\n" + "模型1:平均值模型\n" + "模型2:中位数模型\n" + "模型3:两端间隔对称模型\n" + "模型4:平均间隔模型\n" + "模型5:区间均分模型")
    # 初始化采样表格和结果表格
    temp1 = []
    for i in range(m):
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

    for k in range(n):
        print("\n接下来是第" + str(k + 1) + "轮实验\n")
        # 初始化三个表格
        table_samples.clear_rows()
        table_analysis.clear_rows()
        table_results.clear_rows()
        result_table = []

        random_sum = bool(input('请输入该轮出租车总数是否随机(若是请输入1，否则输入0):'))  # 每轮出租车总数是否随机
        if not random_sum:
            total = int(input('请输入第' + str(k + 1) + '轮实验出租车总数'))
        else:  # 随机生成总数
            total = (np.random.randint(100, 10000, 1))[0]
        print("在第" + str(k + 1) + "轮实验中,出租车总数为" + str(total))
        for i in range(t):
            sample(m, total, t)  # 开始采样
        print("接下来是采样表格:")
        print(table_samples)
        print("接下来是各模型结果表格:")
        print(table_results)
        min0 = analyze(t, total, k)  # 误差分析
        print("接下来是误差分析表格:")
        print(table_analysis)
        print("\n该组实验中，最优模型为模型" + str(min0['id']+1) + "\n")
        model_ana[min0['id']] += 1
    print("\n所有实验结束，在所有实验中精确度最高的为模型" + str(model_ana.index(max(model_ana))+1) + "\n")


if __name__ == '__main__':
    table_samples = PrettyTable()  # 采样表格
    table_results = PrettyTable()  # 结果表格
    table_analysis = PrettyTable()  # 分析表格
    result_table = []  # 结果表格 （上面的三个表格只用于输出，无法存储数据）
    run()
