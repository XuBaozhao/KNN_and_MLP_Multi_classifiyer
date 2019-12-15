# coding=utf-8
import numpy as np
import operator
import argparse
import pandas as pd

'''
当k为100时，此时的准确率为：0.581292, 错误率为：0.418708.
当k为200时，此时的准确率为：0.581292, 错误率为：0.418708.
当k为300时，此时的准确率为：0.595768, 错误率为：0.404232.
当k为400时，此时的准确率为：0.576837, 错误率为：0.423163.
当k为500时，此时的准确率为：0.571269, 错误率为：0.428731.
当k为600时，此时的准确率为：0.563474, 错误率为：0.436526.
当k为700时，此时的准确率为：0.555679, 错误率为：0.444321.
当k为800时，此时的准确率为：0.547884, 错误率为：0.452116.
当k为900时，此时的准确率为：0.547884, 错误率为：0.452116.

当把不同的K结合进行投票选举时：
当k1为100,k2为100,k3为300,k4为400,k5为300时，此时的准确率为：0.600223, 错误率为：0.399777.
当k1为100,k2为900,k3为100,k4为200,k5为20时，此时的准确率为：0.599109, 错误率为：0.400891.
当k1为100,k2为100,k3为100,k4为100,k5为600时，此时的准确率为：0.581292, 错误率为：0.418708.
当k1为100,k2为100,k3为200,k4为700,k5为200时，此时的准确率为：0.579065, 错误率为：0.420935.
'''

# 属性配置
parser = argparse.ArgumentParser(description='KNN实现多分类')
parser.add_argument('--train_data', default='winequality-white-train.csv', type=str, help='train_data')
parser.add_argument('--test_data', default='winequality-white-test.csv', type=str, help='test_data')
parser.add_argument('--acc', default=0, type=int, help='accuracy_rate')
parser.add_argument('--error', default=0, type=int, help='error_rate')
parser.add_argument('--test_data_num', default=0, type=int, help='test_data_num')
parser.add_argument('--k', default=3, type=int, help='k_value')
args = parser.parse_args()

# 创建数据集和标签
def createDataSet(mode='train'):
    assert mode in ['train', 'test']
    if mode == 'train':
        data = pd.read_csv(args.train_data)
        print(data.describe())
    else:
        data = pd.read_csv(args.test_data)
    group = data.iloc[:,:-1].values
    labels = data.iloc[:,-1:].values
    print(len(group[0]))
    for i in range(len(group[0])):
        # 归一化
        group[:, i:i + 1] = (group[:, i:i + 1] - group[:, i:i + 1].mean()) / (group[:, i:i + 1].max() - group[:, i:i + 1].min())
        group[:, i:i + 1] = (group[:, i:i + 1] - group[:, i:i + 1].mean()) / group[:, i:i + 1].std()
    return group, labels

# KNN算法
def classify0(inX, inY, dataset, labels, k):

    dataSetSize = dataset.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataset
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5

    #按距离大小升序排序，得出按向量大小排序的索引地址
    sortedDistIndicies = distances.argsort()

    classCount = {}
    for i in range(k):
        #获得距离最短的向量值
        voteIlabel = labels[sortedDistIndicies[i]]
        voteIlabel = int(voteIlabel)
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount  = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)

    # 取第一个数据的第一个属性的值，即为出现次数最多的标签
    if int(sortedClassCount[0][0]) == int(inY):
        args.acc += 1
    else:
        args.error += 1

def mul_classify0(inX, inY, dataset, labels, k1, k2, k3, k4, k5):

    dataSetSize = dataset.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataset
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances ** 0.5

    #按距离大小升序排序，得出按向量大小排序的索引地址
    sortedDistIndicies = distances.argsort()

    classCount1 = {}
    for i in range(k1):
        #获得距离最短的向量值
        voteIlabel = labels[sortedDistIndicies[i]]
        voteIlabel = int(voteIlabel)
        classCount1[voteIlabel] = classCount1.get(voteIlabel,0) + 1
    sortedClassCount1  = sorted(classCount1.items(),key=operator.itemgetter(1),reverse=True)

    classCount2 = {}
    for i in range(k2):
        # 获得距离最短的向量值
        voteIlabel = labels[sortedDistIndicies[i]]
        voteIlabel = int(voteIlabel)
        classCount2[voteIlabel] = classCount2.get(voteIlabel, 0) + 1
    sortedClassCount2 = sorted(classCount2.items(), key=operator.itemgetter(1), reverse=True)

    classCount3 = {}
    for i in range(k3):
        # 获得距离最短的向量值
        voteIlabel = labels[sortedDistIndicies[i]]
        voteIlabel = int(voteIlabel)
        classCount3[voteIlabel] = classCount3.get(voteIlabel, 0) + 1
    sortedClassCount3 = sorted(classCount3.items(), key=operator.itemgetter(1), reverse=True)

    classCount4 = {}
    for i in range(k4):
        # 获得距离最短的向量值
        voteIlabel = labels[sortedDistIndicies[i]]
        voteIlabel = int(voteIlabel)
        classCount4[voteIlabel] = classCount4.get(voteIlabel, 0) + 1
    sortedClassCount4 = sorted(classCount4.items(), key=operator.itemgetter(1), reverse=True)

    classCount5 = {}
    for i in range(k5):
        # 获得距离最短的向量值
        voteIlabel = labels[sortedDistIndicies[i]]
        voteIlabel = int(voteIlabel)
        classCount5[voteIlabel] = classCount5.get(voteIlabel, 0) + 1
    sortedClassCount5 = sorted(classCount5.items(), key=operator.itemgetter(1), reverse=True)

    from collections import Counter

    sample = [sortedClassCount1[0][0],sortedClassCount2[0][0],sortedClassCount3[0][0],sortedClassCount4[0][0],sortedClassCount5[0][0]]
    data = Counter(sample)

    # 取第一个数据的第一个属性的值，即为出现次数最多的标签
    if int(data.most_common(1)[0][0]) == int(inY):
        args.acc += 1
    else:
        args.error += 1

train_group, train_labels = createDataSet(mode='train')
test_group, test_labels = createDataSet(mode='test')
args.test_data_num = len(test_group)

for k1 in (100, 200, 300, 400, 500, 600, 700, 800, 900):
    for k2 in (100, 200, 300, 400, 500, 600, 700, 800, 900):
        for k3 in (100, 200, 300, 400, 500, 600, 700, 800, 900):
            for k4 in (100, 200, 300, 400, 500, 600, 700, 800, 900):
                for k5 in (100, 200, 300, 400, 500, 600, 700, 800, 900):
                    for test_data, test_label in zip(test_group, test_labels):
                        mul_classify0(test_data, test_label, train_group, train_labels, k1, k2, k3, k4, k5)

                    print('当k1为%d,k2为%d,k3为%d,k4为%d,k5为%d时，此时的准确率为：%f, 错误率为：%f.' % (k1, k2, k3, k4, k5, args.acc / args.test_data_num, args.error / args.test_data_num))
                    args.acc = 0; args.error = 0