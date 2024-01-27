import torch
from torch import nn
from models import common
imgsize = [128,64,32,16,8]
class TwoBranch(nn.Module):
    def __init__(self, args):
        super(TwoBranch, self).__init__()
        num_group = 4
        num_every_group = args.base_num_every_group  #2
        self.args = args

        modules_head_fre = [common.ConvBNReLU2D(args.num_channels, out_channels=args.num_features,
                                            kernel_size=3, padding=1, act=args.act)]#可选的norm和激活
        self.head_fre = nn.Sequential(*modules_head_fre)#python 包裹传参方法 可接受不确定数目参数 一个*传入数组 **传入字典

        modules_down1_fre = [common.DownSample(args.num_features, False, False),#pixel shuffle的卷积
                             common.FreBlock9(args.num_features, imgsize=imgsize[1]) #频率分支计算  FRB分支模块
                         ]

        self.down1_fre = nn.Sequential(*modules_down1_fre)
        self.down1_fre_mo = nn.Sequential(
                                          common.InvertedResidual(args.num_features, args.num_features),
                                          common.FreBlock9(args.num_features,imgsize=int(imgsize[1]))  #加大卷积
                                          )


        modules_down2_fre = [common.DownSample(args.num_features, False, False),
                         common.FreBlock9(args.num_features,imgsize=imgsize[2])
                         ]
        self.down2_fre = nn.Sequential(*modules_down2_fre)
        self.down2_fre_mo = nn.Sequential(
                                          common.InvertedResidual(args.num_features, args.num_features),
                                          common.FreBlock9(args.num_features,imgsize=imgsize[2])
                                          )


        modules_down3_fre = [common.DownSample(args.num_features, False, False),
                         common.FreBlock9(args.num_features,imgsize=imgsize[3])
                         ]
        self.down3_fre = nn.Sequential(*modules_down3_fre)
        self.down3_fre_mo = nn.Sequential(
                                          common.InvertedResidual(args.num_features, args.num_features),
                                          common.FreBlock9(args.num_features,imgsize=imgsize[3])
                                          )

        modules_neck_fre = [
                            common.InvertedResidual(args.num_features,args.num_features),
                            common.FreBlock9(args.num_features,imgsize=imgsize[3])
                         ]
        self.neck_fre = nn.Sequential(*modules_neck_fre)
        self.neck_fre_mo = nn.Sequential(
                                         common.InvertedResidual(args.num_features,args.num_features),
                                         common.FreBlock9(args.num_features,imgsize=imgsize[3]))
        modules_up1_fre = [common.UpSampler(2, args.num_features),
                           common.FreBlock9(args.num_features,imgsize=imgsize[2])
                         ]
        self.up1_fre = nn.Sequential(*modules_up1_fre)
        self.up1_fre_mo = nn.Sequential(
                                        common.InvertedResidual(args.num_features, args.num_features),
                                        common.FreBlock9(args.num_features,imgsize=imgsize[2])
                                        )

        modules_up2_fre = [common.UpSampler(2, args.num_features),
                       common.FreBlock9(args.num_features,imgsize=imgsize[1])
                         ]
        self.up2_fre = nn.Sequential(*modules_up2_fre)
        self.up2_fre_mo = nn.Sequential(
                                        common.InvertedResidual(args.num_features, args.num_features),
                                        common.FreBlock9(args.num_features,imgsize=imgsize[1])
                                        )

        modules_up3_fre = [common.UpSampler(2, args.num_features),
                       common.FreBlock9(args.num_features,imgsize=imgsize[0])
                         ]
        self.up3_fre = nn.Sequential(*modules_up3_fre)
        self.up3_fre_mo = nn.Sequential(
                                        common.InvertedResidual(args.num_features, args.num_features),
                                        common.FreBlock9(args.num_features,imgsize=imgsize[0])
                                        )

        modules_tail_fre = [
            common.ConvBNReLU2D(args.num_features, out_channels=args.num_channels, kernel_size=3, padding=1,
                         act=args.act)]
        self.tail_fre = nn.Sequential(*modules_tail_fre)


        ### spatial branch

        modules_head = [common.ConvBNReLU2D(args.num_channels, out_channels=args.num_features,
                                            kernel_size=3, padding=1, act=args.act)]
        self.head = nn.Sequential(*modules_head)

        modules_down1 = [common.DownSample(args.num_features, False, False),
                         common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down1 = nn.Sequential(*modules_down1)
        self.down1_mo = nn.Sequential(
                                      common.InvertedResidual(args.num_features, args.num_features),
                                      common.ResidualGroup(
                                        args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                                      )


        modules_down2 = [common.DownSample(args.num_features, False, False),
                         common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down2 = nn.Sequential(*modules_down2)
        self.down2_mo = nn.Sequential(
                                      common.InvertedResidual(args.num_features, args.num_features),
                                      common.ResidualGroup(
                                        args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                                      )
        modules_down3 = [common.DownSample(args.num_features, False, False),
                         common.ResidualGroup(
                             args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.down3 = nn.Sequential(*modules_down3)
        self.down3_mo = nn.Sequential(
                                      common.InvertedResidual(args.num_features, args.num_features),
                                      common.ResidualGroup(
                                            args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                                      )
        modules_neck = [
                        common.InvertedResidual(args.num_features,args.num_features),
                        common.ResidualGroup(
                            args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.neck = nn.Sequential(*modules_neck)

        self.neck_mo = nn.Sequential(
                        common.InvertedResidual(args.num_features,args.num_features),
                        common.ResidualGroup(
                            args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
        )



        modules_up1 = [common.UpSampler(2, args.num_features),
                       common.ResidualGroup(
                           args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.up1 = nn.Sequential(*modules_up1)
        self.up1_mo = nn.Sequential(
                                    common.InvertedResidual(args.num_features,args.num_features),
                                    common.ResidualGroup(
                                        args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                                    )



        modules_up2 = [common.UpSampler(2, args.num_features),
                       common.ResidualGroup(
                           args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.up2 = nn.Sequential(*modules_up2)
        self.up2_mo = nn.Sequential(
                                    common.InvertedResidual(args.num_features,args.num_features),
                                    common.ResidualGroup(
                                        args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
            )


        modules_up3 = [common.UpSampler(2, args.num_features),
                       common.ResidualGroup(
                           args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                         ]
        self.up3 = nn.Sequential(*modules_up3)
        self.up3_mo = nn.Sequential(
                                    common.InvertedResidual(args.num_features, args.num_features),
                                    common.ResidualGroup(
                                        args.num_features, 3, 4, act=args.act, n_resblocks=num_every_group, norm=None)
                                    )

        # define tail module
        modules_tail = [
            common.ConvBNReLU2D(args.num_features, out_channels=args.num_channels, kernel_size=3, padding=1,
                         act=args.act)]

        self.tail = nn.Sequential(*modules_tail)

        self.ed_fuse = nn.Sequential(
                                     nn.Conv2d(2*args.num_features,args.num_features,1,1,0),
                                    common.ChannelAttentionLayer(2*args.num_features),
                                     common.HourGlassBlock(depth=3, c_in=args.num_features, c_out=args.num_features))

        conv_fuse = []
        for i in range(14):
            conv_fuse.append(common.FuseBlock7(args.num_features))  #FSIB 每个组件注意都要用 residual

        self.conv_fuse = nn.Sequential(*conv_fuse)

    def forward(self, sample):

#### frequency domain
        x_fre = self.head_fre(sample["lr_up"]) # 128
        down1_fre = self.down1_fre(x_fre)# 64
        down1_fre_mo = self.down1_fre_mo(down1_fre)
        down2_fre = self.down2_fre(down1_fre_mo) # 32
        down2_fre_mo = self.down2_fre_mo(down2_fre)
        down3_fre = self.down3_fre(down2_fre_mo) # 16
        down3_fre_mo = self.down3_fre_mo(down3_fre)

        neck_fre = self.neck_fre(down3_fre_mo) # 16
        neck_fre_mo = self.neck_fre_mo(neck_fre)
        neck_fre_mo = neck_fre_mo + down3_fre_mo

        up1_fre = self.up1_fre(neck_fre_mo) # 32
        up1_fre_mo = self.up1_fre_mo(up1_fre)
        up1_fre_mo = up1_fre_mo + down2_fre_mo
        up2_fre = self.up2_fre(up1_fre_mo) # 64
        up2_fre_mo = self.up2_fre_mo(up2_fre)
        up2_fre_mo = up2_fre_mo + down1_fre_mo
        up3_fre = self.up3_fre(up2_fre_mo) # 128
        up3_fre_mo = self.up3_fre_mo(up3_fre)
        up3_fre_mo = up3_fre_mo + x_fre
        res_fre = self.tail_fre(up3_fre_mo)

#  spatial domain
        x = self.head(sample["lr_up"])  # 128
        down1 = self.down1(x) # 64
        down1_fuse = self.conv_fuse[0](down1_fre, down1)
        down1_mo = self.down1_mo(down1_fuse)
        down1_fuse_mo = self.conv_fuse[1](down1_fre_mo, down1_mo)
        down2 = self.down2(down1_fuse_mo) # 32
        down2_fuse = self.conv_fuse[2](down2_fre, down2)
        down2_mo = self.down2_mo(down2_fuse)  # 32
        down2_fuse_mo = self.conv_fuse[3](down2_fre_mo, down2_mo)
        down3 = self.down3(down2_fuse_mo) # 16
        down3_fuse = self.conv_fuse[4](down3_fre, down3)
        down3_mo = self.down3_mo(down3_fuse)  # 16
        down3_fuse_mo = self.conv_fuse[5](down3_fre_mo, down3_mo)

        neck = self.neck(down3_fuse_mo) # 16
        neck_fuse = self.conv_fuse[6](neck_fre, neck)
        neck_mo = self.neck_mo(neck_fuse)
        neck_mo = neck_mo + down3_mo
        neck_fuse_mo = self.conv_fuse[7](neck_fre_mo, neck_mo)
        residual_fuse = torch.cat((neck_fuse_mo,down3_fuse_mo),dim=1)
        residual_fuse = self.ed_fuse(residual_fuse)
        neck_fuse_mo = neck_fuse_mo+residual_fuse

        up1 = self.up1(neck_fuse_mo) # 32
        up1_fuse = self.conv_fuse[8](up1_fre, up1)
        up1_mo = self.up1_mo(up1_fuse)
        up1_mo = up1_mo + down2_mo
        up1_fuse_mo = self.conv_fuse[9](up1_fre_mo, up1_mo)
        residual_fuse = torch.cat((up1_fuse_mo,down2_fuse_mo),dim=1)
        residual_fuse = self.ed_fuse(residual_fuse)
        up1_fuse_mo = up1_fuse_mo+residual_fuse
        up2 = self.up2(up1_fuse_mo) # 64
        up2_fuse = self.conv_fuse[10](up2_fre, up2)
        up2_mo = self.up2_mo(up2_fuse)
        up2_mo = up2_mo + down1_mo
        up2_fuse_mo = self.conv_fuse[11](up2_fre_mo, up2_mo)
        residual_fuse = torch.cat((up2_fuse_mo,down1_fuse_mo),dim=1)
        residual_fuse = self.ed_fuse(residual_fuse)
        up2_fuse_mo = up2_fuse_mo+residual_fuse
        up3 = self.up3(up2_fuse_mo) # 128
        up3_fuse = self.conv_fuse[12](up3_fre, up3)
        up3_mo = self.up3_mo(up3_fuse)
        up3_mo = up3_mo + x
        up3_fuse_mo = self.conv_fuse[13](up3_fre_mo, up3_mo)
        res = self.tail(up3_fuse_mo)

        return {'img_out': res + sample["lr_up"], 'img_fre': res_fre + sample["lr_up"]}

def make_model(args):
    return TwoBranch(args)

