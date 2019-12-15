import pandas as pd
import argparse
import torch.nn as nn
import torch
import os

'''
网络结构过于复杂：
准确率为:0.540089, 错误率为:0.459911
epoch:6. lr:[0.01] 
准确率为:0.543430, 错误率为:0.456570
epoch:7. lr:[0.01] 
准确率为:0.548998, 错误率为:0.451002
epoch:8. lr:[0.01] 

简化网络结构：精度提升
准确率为:0.602450, 错误率为:0.397550
epoch:27. lr:[0.009] 
准确率为:0.603563, 错误率为:0.396437
epoch:28. lr:[0.009] 
准确率为:0.593541, 错误率为:0.406459
epoch:29. lr:[0.0008099999999999998] 
准确率为:0.603563, 错误率为:0.396437
epoch:30. lr:[0.0026999999999999997] 
'''

# 属性配置
parser = argparse.ArgumentParser(description='CNN实现多分类')
parser.add_argument('--train_data', default='winequality-white-train.csv', type=str, help='train_data')
parser.add_argument('--test_data', default='winequality-white-test.csv', type=str, help='test_data')
args = parser.parse_args()


gpu_ids = [0, 1, 2]
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2'
cuda = torch.cuda.is_available()

# 创建数据集和标签
def createDataSet(mode='train'):
    assert mode in ['train', 'test']
    if mode == 'train':
        data = pd.read_csv(args.train_data)
    else:
        data = pd.read_csv(args.test_data)
    group = data.iloc[:,:-1].values
    # print(type(group))
    labels = data.iloc[:,-1:].values
    labels = labels - 3
    for i in range(len(group[0])):
        # 归一化
        group[:, i:i + 1] = (group[:, i:i + 1] - group[:, i:i + 1].mean()) / (group[:, i:i + 1].max() - group[:, i:i + 1].min())
        group[:, i:i + 1] = (group[:, i:i + 1] - group[:, i:i + 1].mean()) / group[:, i:i + 1].std()
    return group, labels

class Net(nn.Module):
    def __init__(self, in_channel, Num_classes):
        super().__init__()
        self.hidden1 = nn.Sequential(
            nn.Linear(in_features=in_channel, out_features=30),
            nn.ReLU(),
            nn.Dropout(p=0.3))
        # self.hidden2 = nn.Sequential(
        #     nn.Linear(in_features=30, out_features=50),
        #     nn.ReLU(),
        #     nn.Dropout(p=0.5)
        # )
        self.hidden3 = nn.Sequential(
            nn.Linear(in_features=30, out_features=20),
            nn.ReLU(),
            nn.Dropout(p=0.5))
        self.out = nn.Linear(in_features=20, out_features=Num_classes)

    def forward(self, x):
        x = self.hidden1(x)
        #x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.out(x)
        return x

group, labels = createDataSet()
test_group, test_label = createDataSet(mode='test')
len_test = len(test_group)
# 898
group = torch.FloatTensor(group)
labels = torch.LongTensor(labels)

test_group = torch.FloatTensor(test_group)
test_label = torch.LongTensor(test_label)

net = Net(in_channel=11, Num_classes=7).cuda()
net = nn.DataParallel(net, gpu_ids)

optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.3, last_epoch=-1)

criterion = torch.nn.CrossEntropyLoss().cuda()
epochs = 100000000

last_acc = 0

def train(epoch):
    global group
    global labels
    for i in range(100):
        group, labels = group.cuda(), labels.cuda()
        net.train()
        optimizer.zero_grad()
        predict = net(group)

        labels = labels.squeeze()

        loss = criterion(predict, labels)
        loss.backward()
        optimizer.step()

    print('epoch:%d. lr:%s ' % (epoch, scheduler.get_lr()))

def test(epoch):
    global test_group
    global test_label
    acc = 0
    err = 0
    net.eval()
    predict = net(test_group)
    predict = torch.argmax(predict, dim=1)
    for i in range(len(predict)):
        if int(predict[i]) == int(test_label[i]):
            acc += 1
        else:
            err += 1

    print('准确率为:%f, 错误率为:%f' % (acc/len_test, err/len_test))
    if acc / len_test > 0.9:
        torch.save('hx_model/%f.pkl' % (acc/len_test))


for epoch in range(epochs):
    scheduler.step()
    train(epoch)
    test(epoch)