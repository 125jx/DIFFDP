# 处理医学图像数据集的 PyTorch 数据加载器和数据集类
import os.path    # 导入用于处理文件路径的函数
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torch
import numpy as np
import SimpleITK as sit
from PIL import Image

# def is_image_file(filename):
#     return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class TrainDataset(Dataset):
    def __init__(self, dir):
        super(TrainDataset, self).__init__()
        self.dir = dir
        self.transform = torch.from_numpy    # 转换数组的函数：将 NumPy 数组转换为 PyTorch张量

        self.files = []
        self.names = []
        # 写入文件路径和文件名
        for root,_,fnames in sorted(os.walk(dir)):
            for fname in fnames:
                self.files.append(os.path.join(self.dir,fname))
                self.names.append(fname.split(".")[0])
        # self.names = self.names[-8:]
        # self.files = self.files[-8:]

        print("patients: ",len(self.names))  # 输出病人的数量


    # 数据集中的特定索引处加载文件，提取输入数据和标签
    def __getitem__(self, index):
        file = np.load(self.files[index])

        inputs = self.transform(file["arr_0"]).type(torch.FloatTensor) #torch.Size([1, 185, 6, 512, 512])

        rd = self.transform(file["arr_1"]).type(torch.FloatTensor) #torch.Size([1, 185, 1, 512, 512])
        c = len(rd)

        # print(rd.shape)
        #return {'original': orig,'rd':rd,'B':Bladder_numpy,'FHL':FemoralHeadL_numpy,'FHR':FemoralHeadR_numpy,'P':PCTV_numpy,'S':Smallintestine_numpy,'channel':c}
        # 返回一个包含相关信息的字典。这个字典中包含了输入数据、标签数据、通道数和数据的名称。
        return {'inputs': inputs*2-1, 'rd': rd*2-1, 'channel': c, "name": self.names[index]}

    # 返回数据集中样本的总数
    def __len__(self):
        return len(self.names)

   # def name(self):
     #   return str(self.kind)+'Dataset'


# 创建并返回用于训练数据的 PyTorch DataLoader
# (DataLoader 是 PyTorch 中一个用于加载数据的实用工具。它提供了对数据的并行加载和预处理的支持，使得在训练神经网络时更加方便。)
def make_datasetS():
    dir = r"F:\lab_dataset\rectum333_npz\train"
    batch_size = 1    # 批大小（Batch Size）是指在一次模型训练中同时处理的样本数量
    # Syn_train = TrainDataset(dir,"keshihuaForComparison_DVH")
    Syn_train = TrainDataset(dir)
    # 设置了批大小，对数据进行了洗牌，如果存在不完整的批次则丢弃最后一个，还指定了用于数据加载的工作线程数量。
    SynData_train = DataLoader(dataset=(Syn_train), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=1)

    return SynData_train

#  检查当前脚本是否是主程序直接运行的
if __name__ == "__main__":
    tra = make_datasetS()
    channels = []
    # batch = 2
    for ii, batch_sample in enumerate(tra):
        # target是标签
        inputs, target, c, name = batch_sample['inputs'], batch_sample['rd'], batch_sample['channel'], batch_sample['name']

        print(name)
        print(inputs.shape)
        channels.append(c)
    print("--------------")
    print(min(channels), max(channels))
