import random
import time
import torch
from GAN import CDiff               # 从GAN文件中国导入CDiff模型
from Visualize import Visualizer    # 从Visualize中导入可视化工具
from datasets_npz import make_datasetS  # 加载数据集的dataloader，便于后期的预处理
from torch.autograd import Variable     # 一种自动计算梯度的机制，用于构建动态计算图。
from collections import OrderedDict     # 普通字典不同，OrderedDict 在迭代时会按照元素插入的顺序返回键值对。这对于需要保持元素顺序的场景非常有用

# 一个医学图像处理模型的训练脚本，用于训练生成对抗网络（GAN）模型。
def back(x):
    return x/2+0.5
# 初始化三个属性 sum、cnt、cur，分别表示累计和、计数和当前值。简单的指标记录方式
class Metric():
    def __init__(self):
        self.sum = 0
        self.cnt = 0
        self.cur = 0

    def add(self, x):
        self.sum += x
        self.cnt += 1
        self.cur = x

    def get_sum(self):
        return self.sum

    def get_avg(self):
        if self.cnt > 0:
            return self.sum/self.cnt
        else:
            return 0

    def clear(self):
        self.sum=0
        self.cnt=0
        self.cur=0

if __name__ == '__main__':

    param = OrderedDict()        # 初始化一个有序字典
    param['gpu_ids'] = [0]       # 设置使用的GPU的编号
    vis = Visualizer('MICCAI')   # 用于在训练过程中可视化模型的中间结果、损失曲线等信息
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True     # 启用 cuDNN 加速和优化的功能
    # 实例化一个可以加载数据集的对象，用于处理批数据
    trainData = make_datasetS()
    # 构建一个用于训练的模型
    model = CDiff(gpu_ids=param['gpu_ids'], is_Train=True, continue_train=True)
    batch =16  # dataloader的bs是1 但是由于数据是3D的 取出来时是1 1 185 6 512 512（其中第二个1是dataloder是人为误操作） 我们人工设置batch=2得到2 6 512 512（其中2是依次取到185）来达到2D输入网络
    # model.update_learning_rate()
    for epoch in range(295, 1000):    # 循环遍历训练的 epochs，从 295 开始一直到 1000。
        loss_l = Metric()   # 低剂量
        loss_h = Metric()   # 高剂量
        loss = Metric()     # 总损失

        epoch_start_time = time.time()  # 记录每一步的训练开始时间
        # 每一次循环中，从trainData中获取一个批次（batch）的样本，其中包括inputs（输入数据），target（目标数据），通道数等信息
        for ii, batch_sample in enumerate(trainData):
            # 在这一个批次的信息之中获取输入数据+目标数据+通道数+名称
            inputs, target, channel, name = batch_sample['inputs'], batch_sample['rd'],\
                                              batch_sample['channel'], batch_sample['name']
            # torch.Size([1, 140, 512, 512])
            # inputs张量中的第一个维度为1的维度去掉，这通常是为了去掉批次（batch）维度，以使数据更容易处理。
            inputs = inputs.squeeze(0)  # inputs shape:[154,512, 512]
            # inputs = inputs.unsqueeze(1)  # torch.Size([185, 1, 512, 512])
            # print(target.shape)
            target = target.squeeze(0)   # 同上
            # target = target.unsqueeze(1)  # torch.Size([185, 1, 512, 512])
            # print(target.shape)
            # 生成一个随机整数start_slice，用于确定从哪个切片（slice）开始处理数据。
            start_slice = random.randint(0, batch - 1)
            # start_slice=0
            inputs = inputs[start_slice:]
            target = target[start_slice:]
            # rng = np.random.default_rng()
            # idx=rng.permutation(list(range(channel)))
            # print(channel,idx.max())
            for i in range(((channel - start_slice) // batch)):


                # if batch * i + batch <= channel - 1:  # 这样会少最后一个batch 所以改为<=c
                    # print(batch*i + batch)

                main_inputs = inputs[batch * i:batch * (i + 1), :, :, :]  # inputs shape:[2, 1, 512, 512]
                main_target = target[batch * i:batch * (i + 1), :, :, :]
                # if main_target.max()<0.5:
                #     continue
                # ss = random.sample(range(0,(channel - start_slice)),batch)
                # main_inputs = inputs[ss, :, :, :]  # inputs shape:[2, 1, 512, 512]
                # main_target = target[ss, :, :, :]
                # else:
                #     main_inputs = inputs[batch * i:, :, :, :]  # inputs shape:[2, 1, 512, 512]
                #     main_target = target[batch * i:, :, :, :]
                # 获取的数据转换为PyTorch的Variable对象，并将其移动到GPU上
                main_inputs, main_target = Variable(main_inputs).cuda(), Variable(main_target).cuda()
                # print(torch.max(targets))
                # 实现了模型的反向传播和梯度更新。
                model.optimizer_parameters(main_inputs,main_target)
                #  通过模型的recon属性生成预测的剂量数据，并使用clamp方法确保在0到1的范围内。
                dose = back(model.recon).clamp(0.,1.)     # 2 1 512 512
                #  将预测的剂量数据转换为NumPy数组，并将其从GPU移动到CPU上。这是为了在可视化中使用。
                dose_show = dose.detach().cpu().numpy()   # 如果去掉detach 报错：loss.cpu().numpy())#报错 RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
                target_show = back(model.init_GT).detach().cpu().numpy()  # 就是说要numpy不能存有grad的东西 所以detach阻断反向传播（即去掉grad属性）
                # vis.img("dose", img_=model.input.detach().cpu().numpy())
                # 展示图像，预测的+目标的+初始的
                vis.img("pre", img_=dose_show )
                vis.img("target", img_=target_show)
                vis.img("init",img_=back(model.init_pre).clamp(0.,1.).detach().cpu().numpy())
                # vis.img("iso", img_=torch.argmax(model.iso,dim=1,keepdim=True).detach().cpu().numpy()*50)

                # print(dose_show.max(),target_show.max())
                # vis.img("ptv", img_=model.mask_ptv[0].detach().cpu().numpy() * 255)
                loss_h.add(model.loss)
                # dose = model.init_dose  # 2 1 512 512
                # dose_show = back(model.diffusion.data['SR'])[
                #     0].unsqueeze(1).detach().cpu().numpy()  # 如果去掉detach 报错：loss.cpu().numpy())#报错 RuntimeError: Can't call numpy() on Variable that requires grad. Use var.detach().numpy() instead.
                # target_show = model.data['HR'][0].detach().cpu().numpy()  # 就是说要numpy不能存有grad的东西 所以detach阻断反向传播（即去掉grad属性）
                # # vis.img("dose", img_=dose_show)
                # vis.img("target", img_=back(target_show))
                # # vis.img("res",abs(model.data['HR'][0].detach().cpu().numpy())/2)
                # loss_l.add(model.loss_l)
                # loss_h.add(model.loss_h)
                loss.add(model.loss)


        # print('lr=', model.lr_reg, ',loss_G=', model.loss_G.item(), ',loss_D=', model.loss_D.item())
        # vis.plot("loss_l",loss_l.get_avg().item())
        # 将当前 epoch 的平均高剂量区域损失 (loss_h) 和总体损失 (loss) 可视化。
        vis.plot("loss_h",loss_h.get_avg().item())
        vis.plot("loss", loss.get_avg().item())
        # 累积损失清零
        loss_l.clear()
        loss_h.clear()
        loss.clear()
        epoch_end_time = time.time()    # 记录这一步的结束时间
        print('time consuming:', (epoch_end_time - epoch_start_time) / 60)
        print("local time", time.strftime('%c'))     # 本地时间
        # if epoch == 0 or epoch >= 30:
        print('save_model....')
        print('epoch: %d' % epoch)
        # 如果当前 epoch 的倍数是 50，就保存该 epoch 的模型
        if (epoch)%50==0:
            model.save_model(epoch)
        model.save_model('latest')
        # if (epoch+1)%400 == 0 and epoch>1100:
        #     model.update_learning_rate()q
