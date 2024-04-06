import csv
import os
import time
import random

import torch
import numpy as np
from GAN import CDiff
# from GAN_pre import CDiff
# from GAN_pre_ffcat import CDiff
from Visualize import Visualizer
from datasets_npz import TrainDataset
from torch.autograd import Variable
from collections import OrderedDict
# from dwt import IWT
# from train_ab import Metric
from save import save_one
from torch.utils.data import DataLoader

# 生成图像，并将生成的图像与真实图像进行对比保存。
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)
def back(x):
    return x/2+0.5

if __name__ == '__main__':

    param = OrderedDict()
    # os.environ['CUDA_VISIBLE_DEVICES'] = '
    param['gpu_ids'] = [0]
    vis = Visualizer('test150')
    path = r"mha\diffdp"
    # path= r"comparison\mha\diffdp"
    if not  os.path.exists(path):
        os.makedirs(path)
    # fp_lossG = open('.\\result\\netG_losses.txt','w')
    # fp_lossD = open('.\\result\\netD_losses.txt', 'w')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    model = CDiff(gpu_ids=param['gpu_ids'], is_Train=True, continue_train=True)
    batch =16  # dataloader的bs是1 但是由于数据是3D的 取出来时是1 1 185 6 512 512（其中第二个1是dataloder是人为误操作） 我们人工设置batch=2得到2 6 512 512（其中2是依次取到185）来达到2D输入网络
    # model.update_learning_rate()
    dir = r"F:\fzh\dataset\rectum333_npz\test"
    # dir = r'D:\zhichang_process'

    # Syn_train = TrainDataset(dir,"keshihuaForComparison_DVH")
    Syn_test = TrainDataset(dir)

    trainData = DataLoader(dataset=(Syn_test), batch_size=1, shuffle=False, drop_last=True, num_workers=1)

    # iwt =IWT()
    # time_c = Metric()
    print(len(trainData))
    if True:

        for ii, batch_sample in enumerate(trainData):

            inputs, target, channel, name = batch_sample['inputs'], batch_sample['rd'],\
                                              batch_sample['channel'], batch_sample['name'][0]
            print(name)
            # torch.Size([1, 140, 512, 512])

            inputs = inputs.squeeze(0)  # inputs shape:[154,512, 512]
            # inputs = inputs.unsqueeze(1)  # torch.Size([185, 1, 512, 512])
            # print(target.shape)
            target = target.squeeze(0)
            # target = target.unsqueeze(1)
            # print(c)
            crop = (channel%32)//2
            endd = (channel-crop)%32
            start_slice = crop
            # start_slice=0
            inputs = inputs[start_slice:]
            target = target[start_slice:]

            fakes = np.zeros(shape=(channel, 160,160))
            reals = np.zeros(shape=(channel, 160,160))
            epoch_start_time = time.time()

            for i in range((channel-start_slice) // batch):
                # if i==0:
                #     continue

                if batch * i + batch <= channel :  # 这样会少最后一个batch 所以改为<=c
                    # print(batch*i + batch)
                    main_inputs = inputs[batch * i:batch * (i + 1), :, :, :]  # inputs shape:[2, 1, 512, 512]
                    main_target = target[batch * i:batch * (i + 1), :, :, :]

                else:
                    break

                # if main_target.max()<0.5:
                #     continue


                # if main_inputs.max()<0.9:
                #     continue
                # if main_inputs.max()<1e-3:#過濾掉那些沒有mask的切片（mask為1）
                #     continue




                main_inputs, main_target = Variable(main_inputs).cuda(), Variable(main_target).cuda()

                with torch.no_grad():
                    """
                                    c3d
                                    """
                    #
                    # model.forward(main_inputs.unsqueeze(0).transpose(1,2),main_target.transpose(0,1))
                    # fake = model.init_pre[1].clamp_(0., 1.).detach().cpu().numpy()  # c 1 512 512
                    #
                    # tar = model.init_GT.clamp_(0., 1.).detach().cpu().numpy()



                    """diffusion"""

                    model.forward(main_inputs, main_target)  # 加入unsqueeze是由于进入网络要多一层bs 所以是1 6 512 512
                    # # # model.diffusion.test(continous=False, vis=vis)
                    # # # visuals1 = model.diffusion.get_current_visuals(sample=True)
                    # # # fake1 = iwt((visuals1['SAM'] * 2.)).clamp_(0., 1.).detach().cpu().numpy()  # c 1 512 512
                    # # #
                    # vis.img("aux", back(model.a_out).clamp(0,1).detach().cpu().numpy())
                    model.diffusion.test(continous=False, vis=vis)
                    visuals = model.diffusion.get_current_visuals(sample=True)
                    fake = back((visuals['SAM'])).clamp_(0., 1.).detach().cpu().numpy()
                    # vis.img("sam",fake)
                    tar = back(model.data['HR']).clamp_(0., 1.).detach().cpu().numpy()

                    # fake = np.zeros_like(tar)

                    # for iii in range(len(tar)):
                    #     f1 = abs(tar[iii]-fake1[iii]).mean()
                    #     f2 = abs(tar[iii]-fake2[iii]).mean()
                    #     if f1<f2:
                    #         fake[iii]=fake1[iii]
                    #     else:
                    #         fake[iii]=fake2[iii]

                    # other 2D methods
                    # model.net.eval()
                    # model.forward(main_inputs, main_target)
                    #
                    # fake = model.init_pre.clamp(0,1).detach().cpu().numpy()  # c 1 512 512
                    #
                    # tar = model.init_GT.detach().cpu().numpy()
                    # #
                    # #
                    fakes[start_slice+batch * i:start_slice+batch * i + batch, :, :] = fake.squeeze()  # 1 512 512 的c不影响
                    reals[start_slice+batch * i:start_slice+batch * i + batch, :, :] = tar.squeeze()


        # print('lr=', model.lr_reg, ',loss_G=', model.loss_G.item(), ',loss_D=', model.loss_D.item())
        #     epoch_end_time = time.time()
        #     time_c.add((epoch_end_time - epoch_start_time) / 60)
        #     print('time consuming:', (epoch_end_time - epoch_start_time) / 60)
        #     print(abs(fakes - reals).mean())
        #
            save_one(fakes,reals,path,name)
        # print(time_c.get_avg(),time_c.get_var())
        # csv_path = os.path.join(path, "time.csv")
        # with open(csv_path, "w") as file:
        #     writer = csv.writer(file)
        #     # writer.writerow([i, dice_mean])
        #     writer.writerow([time_c.get_avg(),time_c.get_var()])
