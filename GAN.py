import os
import torch
import torch.nn as nn
# from Dis import define_D
# from GANLoss import GANLoss
# from Imagepool import ImagePool
from torch.nn.functional import interpolate
# from MtAANET import shared_encoder, seg_decoder, dose_decoder
import model as Model
import argparse    # 用于解析命令行参数。这使得脚本能够从命令行接受输入参数。
import core.logger as Logger     # 在脚本执行期间记录日志信息。
from model.sr3_modules.unet1 import UNet     # 导入U-net架构


# 用于训练的模型，其中涉及到了生成器网络、噪声扩散模型以及一些优化器的设置和使用。
class CDiff(nn.Module):
    def __init__(self, gpu_ids=[], is_Train=True, continue_train=False):
        super(CDiff, self).__init__()
        """
            self.isTrain: 表示是否处于训练模式。
            self.gpu_ids: 存储GPU设备的ID列表。
            self.continue_train: 表示是否继续之前的训练。
        """

        self.isTrain = is_Train
        self.gpu_ids = gpu_ids
        self.continue_train = continue_train
        # 初始化生成器网络，采用了一个 UNet 类的实例化对象。该对象的参数在初始化时进行了设置。
        self.netLF = UNet(in_channel=6,out_channel=1,inner_channel=32,norm_groups=16,channel_mults=(1, 2, 4, 8, 16),
                          attn_res=[],res_blocks=1,dropout=0,with_noise_level_emb=False,image_size=160,sca=False)

        # 使用 argparse 解析命令行参数，其中包括配置文件路径、训练/验证阶段、GPU ID、调试模式等。
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', type=str, default='config/sr_sr3_64_512.json',
                            help='JSON file for configuration')
        parser.add_argument('-p', '--phase', type=str, choices=['train', 'val'],
                            help='Run either train(training) or val(generation)', default='val')
        parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
        parser.add_argument('-debug', '-d', action='store_true')
        parser.add_argument('-enable_wandb', action='store_true')
        parser.add_argument('-log_wandb_ckpt', action='store_true')
        parser.add_argument('-log_eval', action='store_true')

        # parse configs
        args = parser.parse_args()
        opt = Logger.parse(args)
        opt = Logger.dict_to_nonedict(opt)  # 解析得到的参数转化为配置字典
        # 使用配置字典 opt 中的信息创建模型。
        self.diffusion = Model.create_model(opt)
        # 配置中获取噪声计划，并将其设置到模型中。
        self.diffusion.set_new_noise_schedule(
            opt['model']['beta_schedule'][opt['phase']], schedule_phase=opt['phase'])
        # print(self.diffusion.netG)

        # 打印生成器网络的模型信息
        print(self.diffusion.netG)


        if len(gpu_ids) > 0:
            assert (torch.cuda.is_available())
            self.netLF.cuda(gpu_ids[0])
            self.diffusion.netG.cuda(gpu_ids[0])


        if self.isTrain:
            print('----------load model-------------')
            save_dir = r'pretrain'
            netD_name = 'latest_netLF.pth'

            self.lr_l = 5e-5
            self.lr_h = 5e-5
            self.epoch = 0
            self.criterionL1 = nn.L1Loss()
            # 创建优化器（Adam优化器）self.optimizer_L，并将生成器网络 self.netLF 的参数传递给它。
            self.optimizer_L = torch.optim.Adam(self.netLF.parameters(), lr=self.lr_l, betas=(0.9, 0.999))

            # self.optimizer_dose_decoder = torch.optim.Adam(self.netG4dose.parameters(), lr=self.lr_reg,
            #                                                betas=(0.9, 0.999))
            # self.optimizer_seg_decoder = torch.optim.Adam(self.netG4seg.parameters(), lr=self.lr_cla,
            #                                               betas=(0.9, 0.999))
            # self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=self.lr, betas=(0.9, 0.999))

        if not self.isTrain or self.continue_train:
            print('----------load model-------------')
            save_dir = r'cpt'
            # netD_name = '1399_netLF2.pth'
            # netGenc_name = '1399_netHF2.pth'
            netD_name = '250_netLF2.pth'
            netGenc_name = '250_netHF2.pth'
            netG4seg_name = 'latest_LFopt.pth'
            netG4dose_name = 'latest_HFopt.pth'

            net_D = os.path.join(save_dir, netD_name)
            netGEnc = os.path.join(save_dir, netGenc_name)
            netG4seg = os.path.join(save_dir, netG4seg_name)
            netG4dose = os.path.join(save_dir, netG4dose_name)
            print(net_D,netGEnc)

            self.netLF.load_state_dict(torch.load(net_D))
            self.diffusion.netG.load_state_dict(torch.load(netGEnc))
            self.diffusion.set_new_noise_schedule(
                opt['model']['beta_schedule']['val1'], schedule_phase='val1')
            self.netLF.eval()

            # self.optimizer_L.load_state_dict(torch.load(netG4seg))
            # self.diffusion.optG.load_state_dict(torch.load(netG4dose))
            # self.netLF.eval()
            # self.diffusion.netG.eval()
            # self.netG4seg.load_state_dict(torch.load(netG4seg))
            # self.netG4dose.load_state_dict(torch.load(netG4dose))

    def forward(self, input, target):
        self.sigmoid = nn.Sigmoid()
        self.input = input
        # self.input_s = interpolate(self.input, size=[256,256], mode='bilinear')
        self.dose_GT =self.init_GT= target  # 初始化剂量目标
        # self.init_GT = interpolate(self.dose_GT,size=[256,256],mode='bilinear')
        self.init_pre, f_enc,feats = self.netLF(self.input,0)
        # self.init_pre = self.sigmoid(init_pre)
        # self.f_enc = interpolate(f_enc,scale_factor=1,mode='bilinear')
        # self.init_dose = interpolate(self.init_pre,scale_factor=1,mode='bilinear')
        #0~1 -> -1~1
        self.data = {'HR':self.init_GT,'SR':torch.tensor(0),'LF':f_enc,'MF':feats}
        # self.data = {'HR':self.dose_GT-self.init_dose,'SR':self.init_dose,'LF':self.f_enc}
        # print(self.f_enc.shape)

        self.diffusion.feed_data(self.data)


    #
    #
    # def backward_D(self):
    #     fake_AB = self.fake_AB_pool.query(torch.cat((self.origin, self.dose), 1))
    #     pred_fake = self.netD(fake_AB.detach())
    #     self.loss_D_fake = self.criterionGAN(pred_fake, False)
    #     real_AB = torch.cat((self.origin, self.dose_GT), 1)
    #     pred_real = self.netD(real_AB)
    #     self.loss_D_real = self.criterionGAN(pred_real, True)
    #
    #     self.loss_D = (self.loss_D_fake + self.loss_D_real)
    #     self.loss_D.backward()

    # 优化器的参数更新
    def optimizer_parameters(self, input, target):
        self.netLF.zero_grad()
        self.diffusion.netG.zero_grad()
        self.forward(input, target)
        # self.loss_l = self.criterionL1(self.init_GT,self.init_pre)

        self.loss_l=torch.tensor(0).cuda()
        self.loss_h,self.recon = self.diffusion.optimize_parameters()
        self.loss = self.loss_h
        self.loss.backward()
        self.optimizer_L.step()
        self.diffusion.optG.step()

    # 模型信息的保存
    def save_model(self, epoch):
        save_dir = r'cpt'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        netD_name = '%s_netLF2.pth' % epoch
        netGEnc_name = '%s_netHF2.pth' % epoch
        netG4seg_name = '%s_LFopt.pth' % epoch
        netG4dose_name = '%s_HFopt.pth' % epoch

        netD = os.path.join(save_dir, netD_name)
        netGEnc = os.path.join(save_dir, netGEnc_name)
        netG4seg = os.path.join(save_dir, netG4seg_name)
        netG4dose = os.path.join(save_dir, netG4dose_name)

        torch.save(self.netLF.cpu().state_dict(), netD)
        self.netLF.cuda(self.gpu_ids[0])
        torch.save(self.diffusion.netG.cpu().state_dict(), netGEnc)
        self.diffusion.netG.cuda(self.gpu_ids[0])
        torch.save(self.optimizer_L.state_dict(), netG4seg)
        # self.netG4seg.cuda(self.gpu_ids[0])
        torch.save(self.diffusion.optG.state_dict(), netG4dose)
        # self.netG4dose.cuda(self.gpu_ids[0])

    # 学习率的更新
    def update_learning_rate(self):

        # for param_group in self.optimizer_D.param_groups:
        #     param_group['lr'] = lr
        for param_group in self.optimizer_L.param_groups:
            param_group['lr'] /=2
        for param_group in self.diffusion.optG.param_groups:
            param_group['lr'] /=2
            lr = param_group['lr']
        print('update learning rate: %f' % (lr))