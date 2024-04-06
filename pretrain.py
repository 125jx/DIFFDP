import time
import torch
import numpy as np
from GAN import CDiff
from Visualize import Visualizer
from datasets_npz import make_datasetS
from torch.autograd import Variable
from collections import OrderedDict

# 这段代码主要是一个简单的模型训练脚本，包括模型初始化、数据加载、训练循环等
class Metric():
    def __init__(self):
        self.sum = 0
        self.cnt = 0
        self.cur = 0

    def add(self,x):
        self.sum +=x
        self.cnt +=1
        self.cur = x

    def get_sum(self):
        return self.sum

    def get_avg(self):
        if self.cnt>0:
            return self.sum/self.cnt
        else:
            return 0

    def clear(self):
        self.sum=0
        self.cnt=0
        self.cur=0


if __name__ == '__main__':
    param = OrderedDict()
    param['gpu_ids'] = [1]
    vis = Visualizer('diff')
    # fp_lossG = open('.\\result\\netG_losses.txt','w')
    # fp_lossD = open('.\\result\\netD_losses.txt', 'w')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    trainData, _ = make_datasetS()
    model = CDiff(gpu_ids=param['gpu_ids'], is_Train=True, continue_train=False)
    batch = 24  # dataloader的bs是1 但是由于数据是3D的 取出来时是1 1 185 6 512 512（其中第二个1是dataloder是人为误操作） 我们人工设置batch=2得到2 6 512 512（其中2是依次取到185）来达到2D输入网络
    metric_l1 = Metric()
    metric_loss = Metric()
    for epoch in range(0, 5):
        epoch_start_time = time.time()
        metric_l1.clear()
        metric_loss.clear()
        for ii, batch_sample in enumerate(trainData):
            inputs, target, channel, name = batch_sample['inputs'], batch_sample['rd'],\
                                              batch_sample['channel'], batch_sample['name']
            # torch.Size([1, 140, 512, 512])

            inputs = inputs.squeeze(0)  # inputs shape:[154,512, 512]
            # inputs = inputs.unsqueeze(1)  # torch.Size([185, 1, 512, 512])
            # print(target.shape)
            target = target.squeeze(0)
            target = target.unsqueeze(1)  # torch.Size([185, 1, 512, 512])
            # print(target.shape)
            for i in range(channel // batch):
                if batch * i + batch <= channel - 1:  # 这样会少最后一个batch 所以改为<=c
                    # print(batch*i + batch)
                    main_inputs = inputs[batch * i:batch * (i + 1), :, :, :]  # inputs shape:[2, 1, 512, 512]
                    main_target = target[batch * i:batch * (i + 1), :, :, :]

                else:
                    break

                main_inputs, main_target = Variable(main_inputs).cuda(1), Variable(main_target).cuda(1)
                # print(torch.max(targets))
                model.optimizer_parameters(main_inputs,main_target)
                dose = model.init_pre  # 2 1 512 512
                dose_show = dose[
                    0].detach().cpu().numpy()  # 如果去掉detach 报错：loss.cpu().numpy())#报错 RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
                target_show = model.init_GT[0].detach().cpu().numpy()  # 就是说要numpy不能存有grad的东西 所以detach阻断反向传播（即去掉grad属性）
                vis.img("dose", img_=dose_show*255)
                vis.img("target", img_=target_show*255)
                # vis.img("ptv", img_=model.mask_ptv[0].detach().cpu().numpy() * 255)
                metric_loss.add(model.loss)
                metric_l1.add(model.loss_l)

        vis.plot("loss",metric_loss.get_avg().item())
        vis.plot("l1",metric_l1.get_avg().item())
        print('lr=', model.lr, ',loss_G=', model.loss.item())
        epoch_end_time = time.time()
        print('time consuming:', (epoch_end_time - epoch_start_time) / 60)
        print("local time", time.strftime('%c'))
        # if epoch == 0 or epoch >= 30:
        print('save_model....')
        print('epoch: %d' % epoch)
        # model.save_model(epoch)
        model.save_model('latest')
        if (epoch+1)%200==0:
            model.update_learning_rate()
