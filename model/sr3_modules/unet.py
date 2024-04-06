import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(
                batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x
class SCA(nn.Module):
    def __init__(self,in_channel,norm_groups=32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel,out_channels=in_channel,kernel_size=1,padding=0,stride=1)
        self.conv2 = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, padding=0, stride=1)
        self.n_head = 1

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)
        self.sk = nn.Conv2d(in_channel,in_channel,1)

    def forward(self,x1,x2):    #x1 provides Q
        batch, channel, height, width = x2.shape
        # print(x1.shape)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)

        n_head = self.n_head
        head_dim = channel // n_head

        norm1 = self.norm(x1)
        qkv1 = self.qkv(norm1).view(batch, n_head, head_dim * 3, height, width)
        query1, key1, value1 = qkv1.chunk(3, dim=2)  # bhdyx
        norm2 = self.norm(x2)
        qkv2 = self.qkv(norm2).view(batch, n_head, head_dim * 3, height, width)
        query2, key2, value2 = qkv2.chunk(3, dim=2)  # bhdyx

        # attn1 = torch.einsum(
        #     "bnchw, bncyx -> bnhwyx", query2, key1
        # ).contiguous() / math.sqrt(channel)
        # attn1 = attn1.view(batch, n_head, height, width, -1)
        # attn1 = torch.softmax(attn1, -1)
        # attn1 = attn1.view(batch, n_head, height, width, height, width)
        #
        # out1 = torch.einsum("bnhwyx, bncyx -> bnchw", attn1, value1).contiguous()
        # out1 = self.out(out1.view(batch, channel, height, width))

        attn2 = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query1, key2
        ).contiguous() / math.sqrt(channel)
        attn2 = attn2.view(batch, n_head, height, width, -1)
        attn2 = torch.softmax(attn2, -1)
        attn2 = attn2.view(batch, n_head, height, width, height, width)

        out2 = torch.einsum("bnhwyx, bncyx -> bnchw", attn2, value2).contiguous()
        out2 = self.out(out2.view(batch, channel, height, width))

        return out2+self.sk(x1)

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=1,
        out_channel=1,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        attn_res=[],
        res_blocks=1,
        dropout=0,
        with_noise_level_emb=True,
        image_size=160
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),   #True
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res*2
        self.avg2 = nn.AvgPool2d(2,2)
        self.avg4 = nn.AvgPool2d(4,4)
        self.avg = [self.avg4,self.avg2]
        self.ups = nn.ModuleList(ups)
        self.sca1 = SCA(32)
        self.sca2 = SCA(64)
        self.sca3 = SCA(128)
        self.sca4 = SCA(256)
        self.sca5 = SCA(512)
        self.sca6 = SCA(512)
        self.sca7 = SCA(512)
        self.sca = [0,self.sca1,self.sca2,self.sca3,self.sca4,self.sca5,self.sca6,self.sca7]
        # self.convs=[nn.Conv2d(32,32,1,1),nn.Conv2d(64,64,1,1)]

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

    def forward(self, x, time,fea=None):
        #x: x_t, fea: features from structure encoder
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        # fea0,fea1,fea2,fea3,fea4,fea5=fea
        hh,ww = x.shape[-2],x.shape[-1]


        feats = []
        ii=1
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
                if ii==1 or ii==2:
                    # sca = self.sca[ii]
                    # avg = self.avg[ii]
                    # x = sca(avg(x),avg(fea[ii]))
                    # x = F.interpolate(x,scale_factor=4-ii*2,mode='bilinear')
                    # conv = self.convs[ii]
                    x = x+fea[ii]
                else:
                    sca = self.sca[ii]
                    x = sca(x,fea[ii])
                ii+=1
            else:
                x = layer(x)
                if x.shape[-1]==ww:
                    x+=fea[0]
            feats.append(x)
        # feats = feats+feats2
        # for jj in range(len(feats)):
        #     feats[jj]+=feats2[jj]

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
                sca = self.sca[ii]
                x = sca(x, fea[ii])
                ii += 1
            else:
                x = layer(x)
        # x+=fea[-1]
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)


class StructEnc(nn.Module):
    def __init__(self,
                 in_channel=6,

                 inner_channel=32,
                 norm_groups=32,
                 channel_mults=(1, 2, 4, 8, 8),
                 attn_res=[],
                 res_blocks=1,
                 dropout=0,
                 with_noise_level_emb=True,
                 image_size=128
                 ):
        super().__init__()



        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (now_res in attn_res)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res//2
        self.downs = nn.ModuleList(downs)
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])


    def forward(self, x):
        # if xl:
        #     print(xl.shape)
        time=0

        feats = []
        fea=[]
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, time)
                fea.append(x)
            else:
                x = layer(x)
            feats.append(x)


        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x_m = layer(x, time)
                fea.append(x_m)
            else:
                x_m = layer(x)



        return fea,feats




# class UNet1(nn.Module):
#     def __init__(
#             self,
#             in_channel=6,
#             out_channel=3,
#             inner_channel=32,
#             norm_groups=32,
#             channel_mults=(1, 2, 4, 8, 8),
#             attn_res=(8),
#             res_blocks=3,
#             dropout=0,
#             with_noise_level_emb=True,
#             image_size=128
#     ):
#         super(UNet, self).__init__()
#         self.structure_enc = StructEnc(in_channel=in_channel-1,
#
#                                        out_channel=1, inner_channel=32, norm_groups=16,
#                                        channel_mults=(1, 2, 4, 8, 16),
#                                        attn_res=[], res_blocks=1, dropout=0, with_noise_level_emb=False, image_size=256,
#                                        )
#         self.diffnet = DiffUNet(
#
#         1,
#         out_channel,
#         inner_channel,
#         norm_groups,
#         channel_mults,
#         attn_res,
#         res_blocks,
#         dropout,
#         with_noise_level_emb,
#         image_size
#     )
#
#     def forward(self,x,time):
#         x1 = x[:,:-1,:,:]
#         x2 = x[:,-1,:,:].unsqueeze(1)
#
#
#         fea,feats = self.structure_enc(x1,torch.tensor(0).cuda())
#         out = self.diffnet(x2,time,fea,feats)
#         return out

