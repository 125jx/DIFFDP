# 从一个数据集中加载数据，并检查每个样本的输入（inputs）中的倒数第二个通道的最大值是否小于1。如果小于1，则输出样本的名称和该通道的最大值
from datasets_npz import make_datasetS
data = make_datasetS()

for a in data:
    if a['inputs'][:,-2,...].max()<1:
        print(a['name'],a['inputs'][:,-2,...].max())