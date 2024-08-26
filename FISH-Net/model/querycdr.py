import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from core.spectral_norm import use_spectral_norm
from .basic_net import Conv2dBlock, ResBlocks, TransConv2dBlock
from .resample2d import resample_image

from .utils.conv import Conv
from einops.layers.torch import Rearrange
from .lewin import LeWinTransformerBlock

class attention2d(nn.Module):
    def __init__(self, in_planes, hidden_planes, init_weight=True):
        super(attention2d, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(hidden_planes, 2, 1, bias=True)
        self.sotmax = nn.Softmax(dim=1)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m ,nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return self.sotmax(x)

class BaseNetwork(nn.Module):
  def __init__(self):
    super(BaseNetwork, self).__init__()

  def print_network(self):
    if isinstance(self, list):
      self = self[0]
    num_params = 0
    for param in self.parameters():
      num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f million. '
          'To see the architecture, do print(network).'% (type(self).__name__, num_params / 1000000))

  def init_weights(self, init_type='normal', gain=0.02):
    '''
    initialize network's weights
    init_type: normal | xavier | kaiming | orthogonal
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
    '''
    def init_func(m):
      classname = m.__class__.__name__
      if classname.find('InstanceNorm2d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
          nn.init.constant_(m.weight.data, 1.0)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)
      elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
          nn.init.normal_(m.weight.data, 0.0, gain)
        elif init_type == 'xavier':
          nn.init.xavier_normal_(m.weight.data, gain=gain)
        elif init_type == 'xavier_uniform':
          nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        elif init_type == 'kaiming':
          nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        elif init_type == 'orthogonal':
          nn.init.orthogonal_(m.weight.data, gain=gain)
        elif init_type == 'none':  # uses pytorch's default init method
          m.reset_parameters()
        else:
          raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        if hasattr(m, 'bias') and m.bias is not None:
          nn.init.constant_(m.bias.data, 0.0)

    self.apply(init_func)

    # propagate to children
    for m in self.children():
      if hasattr(m, 'init_weights'):
        m.init_weights(init_type, gain)

class InpaintGenerator(nn.Module):
  def __init__(self):
    super(InpaintGenerator, self).__init__()

    self.flow_column = FlowColumn()
    self.conv_column = InpaintNet()

  def forward(self, inputs, sigma):
      flow_map, flows = self.flow_column(inputs, sigma)
      pyramid_imgs, images_out = self.conv_column(inputs, flows, sigma)
      return pyramid_imgs, images_out

class InpaintNet(BaseNetwork):
  def __init__(self, init_weights=True):
    super(InpaintNet, self).__init__()

    cnum = 32

    self.dw_conv01 = nn.Sequential(
        nn.Conv2d(3, cnum, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True))
    self.dw_conv02 = nn.Sequential(
        nn.Conv2d(cnum, cnum * 2, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True))
    self.dw_conv03 = nn.Sequential(
        nn.Conv2d(cnum * 2, cnum * 4, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True))

    self.up_conv05 = nn.Sequential(
        nn.Conv2d(cnum * 16, cnum * 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True))

    self.up_conv01 = nn.Sequential(
        nn.Conv2d(cnum * 4, cnum * 1, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True))

    self.decoder = nn.Sequential(
        nn.Conv2d(cnum * 2, cnum, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(cnum, 3, kernel_size=3, stride=1, padding=1),
        nn.Tanh())

    self.torgb5 = nn.Sequential(
        nn.Conv2d(cnum * 32, 3, kernel_size=1, stride=1, padding=0),
        nn.Tanh())
    self.torgb4 = nn.Sequential(
        nn.Conv2d(cnum * 16, 3, kernel_size=1, stride=1, padding=0),
        nn.Tanh())
    self.torgb3 = nn.Sequential(
        nn.Conv2d(cnum * 8, 3, kernel_size=1, stride=1, padding=0),
        nn.Tanh())
    self.torgb2 = nn.Sequential(
        nn.Conv2d(cnum * 4, 3, kernel_size=1, stride=1, padding=0),
        nn.Tanh())
    self.torgb1 = nn.Sequential(
        nn.Conv2d(cnum * 2, 3, kernel_size=1, stride=1, padding=0),
        nn.Tanh())
    
    in_channels = 3
    
    self.query_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=cnum, kernel_size=3, stride=2, padding=1)
    self.query_conv2 = nn.Conv2d(in_channels=cnum, out_channels=cnum*2, kernel_size=3, stride=2, padding=1)
    self.query_conv3 = nn.Conv2d(in_channels=cnum*2, out_channels=cnum*4, kernel_size=3, stride=2, padding=1)
    self.query_conv4 = nn.Conv2d(in_channels=cnum*4, out_channels=cnum*8, kernel_size=3, stride=2, padding=1)
    self.query_conv5 = nn.Conv2d(in_channels=cnum*8, out_channels=cnum*16, kernel_size=3, stride=2, padding=1)
    self.query_conv6 = nn.Conv2d(in_channels=cnum*16, out_channels=cnum*16, kernel_size=3, stride=2, padding=1)

    self.query1 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_02 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_04 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_06 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_08 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_2 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_4 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_6 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_8 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.queries = [self.query1, self.query_02, self.query_04, self.query_06, self.query_08, self.query_2, self.query_4, self.query_6, self.query_8]
    
    self.attn3 = attention2d(cnum * 4, cnum)
    self.attn2 = attention2d(cnum * 2, cnum)
    self.attn1 = attention2d(6, cnum)

    self.tr6 = LeWinTransformerBlock(dim=cnum*16, input_resolution=(8,8),
                            num_heads=2, win_size=2,
                            shift_size=0,
                            mlp_ratio=4,
                            qkv_bias=True, qk_scale=None,
                            drop=0, attn_drop=0,
                            drop_path=0.1,
                            norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',
                            modulator=False,cross_modulator=False)

    self.tr5 = LeWinTransformerBlock(dim=cnum*8, input_resolution=(16,16),
                                num_heads=2, win_size=2,
                                shift_size=0,
                                mlp_ratio=4,
                                qkv_bias=True, qk_scale=None,
                                drop=0, attn_drop=0,
                                drop_path=0.1,
                                norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',
                                modulator=False,cross_modulator=False)

    self.tr4 = LeWinTransformerBlock(dim=cnum*4, input_resolution=(32,32),
                                    num_heads=2, win_size=2,
                                    shift_size=0,
                                    mlp_ratio=4,
                                    qkv_bias=True, qk_scale=None,
                                    drop=0, attn_drop=0,
                                    drop_path=0.1,
                                    norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',
                                    modulator=False,cross_modulator=False)

    self.tr_conv4 = nn.Sequential(
        nn.Conv2d(cnum * 4, cnum * 8, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True))
    

    
    self.tr_conv5 = nn.Sequential(
        nn.Conv2d(cnum * 8, cnum * 16, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True))
    
    self.tr_conv6 = nn.Sequential(
        nn.Conv2d(cnum * 16, cnum * 16, kernel_size=3, stride=2, padding=1),
        nn.LeakyReLU(0.2, inplace=True))
    


    self.uptr4 = LeWinTransformerBlock(dim=cnum*16, input_resolution=(8,8),
                                    num_heads=2, win_size=2,
                                    shift_size=0,
                                    mlp_ratio=4,
                                    qkv_bias=True, qk_scale=None,
                                    drop=0, attn_drop=0,
                                    drop_path=0.1,
                                    norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',
                                    modulator=False,cross_modulator=False)

    self.uptr3 = LeWinTransformerBlock(dim=cnum*8, input_resolution=(16,16),
                                    num_heads=2, win_size=2,
                                    shift_size=0,
                                    mlp_ratio=4,
                                    qkv_bias=True, qk_scale=None,
                                    drop=0, attn_drop=0,
                                    drop_path=0.1,
                                    norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',
                                    modulator=False,cross_modulator=False)

    self.uptr2 = LeWinTransformerBlock(dim=cnum*4, input_resolution=(32,32),
                                    num_heads=2, win_size=2,
                                    shift_size=0,
                                    mlp_ratio=4,
                                    qkv_bias=True, qk_scale=None,
                                    drop=0, attn_drop=0,
                                    drop_path=0.1,
                                    norm_layer=nn.LayerNorm,token_projection='linear',token_mlp='leff',
                                    modulator=False,cross_modulator=False)

    self.up_trconv04 = nn.Sequential(
        nn.Conv2d(cnum * 32, cnum * 8, kernel_size=1, stride=1, padding=0),
        nn.ReLU(inplace=True))
    self.up_trconv03 = nn.Sequential(
        nn.Conv2d(cnum * 16, cnum * 4, kernel_size=1, stride=1, padding=0),
        nn.ReLU(inplace=True))
    self.up_trconv02 = nn.Sequential(
        nn.Conv2d(cnum * 8, cnum * 2, kernel_size=1, stride=1, padding=0),
        nn.ReLU(inplace=True))


    if init_weights:
        self.init_weights()

  
  def copy_weight(self):
    with torch.no_grad():
      self.query_02.data.copy_(self.query1.data)
      self.query_04.data.copy_(self.query1.data)
      self.query_06.data.copy_(self.query1.data)
      self.query_08.data.copy_(self.query1.data)
      self.query_2.data.copy_(self.query1.data)
      self.query_4.data.copy_(self.query1.data)
      self.query_6.data.copy_(self.query1.data)
      self.query_8.data.copy_(self.query1.data)

  def forward(self, img, flows, sigma):
        x = img
        b,_,_,_ = x.shape
        sig_q = self.queries[sigma]
        sig_q1 = self.query_conv1(sig_q)
        sig_q2 = self.query_conv2(sig_q1)
        sig_q3 = self.query_conv3(sig_q2)
        sig_q4 = self.query_conv4(sig_q3)
        sig_q5 = self.query_conv5(sig_q4)

        sc1 = self.attn1(torch.cat((x, sig_q.expand(b, -1, -1, -1)),dim = 1))
        x1 = self.dw_conv01(sc1[:,0].view(-1,1,1,1)*x + sc1[:,1].view(-1,1,1,1)*torch.mul(x, sig_q))
        sc2 = self.attn2(torch.cat((x1, sig_q1.expand(b, -1, -1, -1)),dim = 1))
        x2 = self.dw_conv02(sc2[:,0].view(-1,1,1,1)*x1 + sc2[:,1].view(-1,1,1,1)*torch.mul(x1,sig_q1))
        sc3 = self.attn3(torch.cat((x2, sig_q2.expand(b, -1, -1, -1)),dim = 1))
        x3 = self.dw_conv03(sc3[:,0].view(-1,1,1,1)*x2 + sc3[:,1].view(-1,1,1,1)*torch.mul(x2,sig_q2))

        x4 = self.tr_conv4(self.tr4(sig_q3.expand(b, -1, -1, -1),torch.mul(x3, sig_q3)))

        x5 = self.tr_conv5(self.tr5(sig_q4.expand(b, -1, -1, -1),torch.mul(x4, sig_q4)))

        x6 = self.tr_conv6(self.tr6(sig_q5.expand(b, -1, -1, -1),torch.mul(x5, sig_q5)))

        x5 = resample_image(x5, flows[4])
        x4 = resample_image(x4, flows[3])
        x3 = resample_image(x3, flows[2])
        x2 = resample_image(x2, flows[1])
        x1 = resample_image(x1, flows[0])

        upx5 = self.up_conv05(F.interpolate(x6, scale_factor=2, mode='bilinear', align_corners=True))

        upx4 = self.up_trconv04(F.interpolate(torch.cat([x5, self.uptr4(sig_q5.expand(b, -1, -1, -1), torch.mul(upx5, sig_q5))],dim = 1), 
                                              scale_factor=2, mode='bilinear', align_corners=True))
        upx3 = self.up_trconv03(F.interpolate(torch.cat([x4, self.uptr3(sig_q4.expand(b, -1, -1, -1), torch.mul(upx4, sig_q4))],dim = 1),
                                              scale_factor=2, mode='bilinear', align_corners=True))
        upx2 = self.up_trconv02(F.interpolate(torch.cat([x3, self.uptr2(sig_q3.expand(b, -1, -1, -1), torch.mul(upx3, sig_q3))],dim = 1),
                                              scale_factor=2, mode='bilinear', align_corners=True))
        u_sc1 = self.attn3(torch.cat((upx2, sig_q2.expand(b, -1, -1, -1)),dim = 1))
        upx1 = self.up_conv01(F.interpolate(
            u_sc1[:,0].view(-1,1,1,1)*torch.cat((upx2, x2),1) + u_sc1[:,1].view(-1,1,1,1)*torch.mul(torch.cat((upx2, x2),1),sig_q2.repeat(1,2,1,1)), scale_factor=2, mode='bilinear', align_corners=True))


        img5 = self.torgb5(torch.cat([upx5, x5], dim=1))
        img4 = self.torgb4(torch.cat([upx4, x4], dim=1))
        img3 = self.torgb3(torch.cat([upx3, x3], dim=1))
        img2 = self.torgb2(torch.cat([upx2, x2], dim=1))
        img1 = self.torgb1(torch.cat([upx1, x1], dim=1))

        output = self.decoder(
            F.interpolate(torch.cat([upx1, x1], dim=1), scale_factor=2, mode='bilinear', align_corners=True))
        pyramid_imgs = [img1, img2, img3, img4, img5]
        return pyramid_imgs, output

class FlowColumn(nn.Module):
    def __init__(self, input_dim=3, dim=64, n_res=2, activ='lrelu',
                 norm='in', pad_type='reflect', use_sn=True):
        super(FlowColumn, self).__init__()

        self.down_flow01 = nn.Sequential(
            Conv2dBlock(input_dim, dim // 2, 7, 1, 3, norm, activ, pad_type, use_sn=use_sn),
            Conv2dBlock(dim // 2, dim // 2, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))
        self.down_flow02 = nn.Sequential(
            Conv2dBlock(dim // 2, dim // 2, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))
        self.down_flow03 = nn.Sequential(
            Conv2dBlock(dim // 2, dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))
        self.down_flow04 = nn.Sequential(
            Conv2dBlock(dim, 2 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))
        self.down_flow05 = nn.Sequential(
            Conv2dBlock(2 * dim, 4 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))
        self.down_flow06 = nn.Sequential(
            Conv2dBlock(4 * dim, 8 * dim, 4, 2, 1, norm, activ, pad_type, use_sn=use_sn))

        dim = 8 * dim

        self.up_flow05 = nn.Sequential(
            ResBlocks(n_res, dim, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim, dim // 2, 6, 2, 2, norm=norm, activation=activ))

        self.up_flow04 = nn.Sequential(
            Conv2dBlock(dim, dim // 2, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 2, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 2, dim // 4, 6, 2, 2, norm=norm, activation=activ))

        self.up_flow03 = nn.Sequential(
            Conv2dBlock(dim // 2, dim // 4, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 4, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 4, dim // 8, 6, 2, 2, norm=norm, activation=activ))

        self.up_flow02 = nn.Sequential(
            Conv2dBlock(dim // 4, dim // 8, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 8, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 8, dim // 16, 6, 2, 2, norm=norm, activation=activ))

        self.up_flow01 = nn.Sequential(
            Conv2dBlock(dim // 8, dim // 16, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 16, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 16, dim // 16, 6, 2, 2, norm=norm, activation=activ))

        self.location = nn.Sequential(
            Conv2dBlock(dim // 8, dim // 16, 5, 1, 2, norm, activ, pad_type, use_sn=use_sn),
            ResBlocks(n_res, dim // 16, norm, activ, pad_type=pad_type),
            TransConv2dBlock(dim // 16, dim // 16, 6, 2, 2, norm=norm, activation=activ),
            Conv2dBlock(dim // 16, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False))

        self.to_flow05 = Conv2dBlock(dim // 2, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False)
        self.to_flow04 = Conv2dBlock(dim // 4, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False)
        self.to_flow03 = Conv2dBlock(dim // 8, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False)
        self.to_flow02 = Conv2dBlock(dim // 16, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False)
        self.to_flow01 = Conv2dBlock(dim // 16, 2, 3, 1, 1, norm='none', activation='none', pad_type=pad_type, use_bias=False)

        in_channels = 3
        cnum = 32

        self.query_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=cnum, kernel_size=3, stride=2, padding=1)
        self.query_conv2 = nn.Conv2d(in_channels=cnum, out_channels=cnum, kernel_size=3, stride=2, padding=1)
        self.query_conv3 = nn.Conv2d(in_channels=cnum, out_channels=cnum*2, kernel_size=3, stride=2, padding=1)
        self.query_conv4 = nn.Conv2d(in_channels=cnum*2, out_channels=cnum*4, kernel_size=3, stride=2, padding=1)
        self.query_conv5 = nn.Conv2d(in_channels=cnum*4, out_channels=cnum*8, kernel_size=3, stride=2, padding=1)
        #self.query_conv6 = nn.Conv2d(in_channels=cnum*16, out_channels=cnum*16, kernel_size=3, stride=2, padding=1)

        self.query1 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
        self.query_02 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
        self.query_04 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
        self.query_06 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
        self.query_08 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
        self.query_2 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
        self.query_4 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
        self.query_6 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
        self.query_8 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
        self.queries = [self.query1, self.query_02, self.query_04, self.query_06, self.query_08, self.query_2, self.query_4, self.query_6, self.query_8]

        self.attn6 = attention2d(cnum * 16, cnum)
        self.attn5 = attention2d(cnum * 8, cnum)
        self.attn4 = attention2d(cnum * 4, cnum)
        self.attn3 = attention2d(cnum * 2, cnum)
        self.attn2 = attention2d(cnum * 2, cnum)
        self.attn1 = attention2d(6, cnum)

    def copy_weight(self):
      with torch.no_grad():
        self.query_02.data.copy_(self.query1.data)
        self.query_04.data.copy_(self.query1.data)
        self.query_06.data.copy_(self.query1.data)
        self.query_08.data.copy_(self.query1.data)
        self.query_2.data.copy_(self.query1.data)
        self.query_4.data.copy_(self.query1.data)
        self.query_6.data.copy_(self.query1.data)
        self.query_8.data.copy_(self.query1.data)

    def forward(self, inputs, sigma):
        
        b,_,_,_ = inputs.shape

        sig_q = self.queries[sigma]
        sig_q1 = self.query_conv1(sig_q)
        sig_q2 = self.query_conv2(sig_q1)
        sig_q3 = self.query_conv3(sig_q2)
        sig_q4 = self.query_conv4(sig_q3)
        sig_q5 = self.query_conv5(sig_q4)

        sc1 = self.attn1(torch.cat((inputs, sig_q.expand(b, -1, -1, -1)),dim = 1))
        f_x1 = self.down_flow01(sc1[:,0].view(-1,1,1,1)*inputs + sc1[:,1].view(-1,1,1,1)*torch.mul(inputs, sig_q))
        sc2 = self.attn2(torch.cat((f_x1, sig_q1.expand(b, -1, -1, -1)),dim = 1))
        f_x2 = self.down_flow02(sc2[:,0].view(-1,1,1,1)*f_x1 + sc2[:,1].view(-1,1,1,1)*torch.mul(f_x1,sig_q1))
        sc3 = self.attn3(torch.cat((f_x2, sig_q2.expand(b, -1, -1, -1)),dim = 1))
        f_x3 = self.down_flow03(sc3[:,0].view(-1,1,1,1)*f_x2 + sc3[:,1].view(-1,1,1,1)*torch.mul(f_x2,sig_q2))
        sc4 = self.attn4(torch.cat((f_x3, sig_q3.expand(b, -1, -1, -1)),dim = 1))
        f_x4 = self.down_flow04(sc4[:,0].view(-1,1,1,1)*f_x3 + sc4[:,1].view(-1,1,1,1)*torch.mul(f_x3,sig_q3))
        sc5 = self.attn5(torch.cat((f_x4, sig_q4.expand(b, -1, -1, -1)),dim = 1))
        f_x5 = self.down_flow05(sc5[:,0].view(-1,1,1,1)*f_x4 + sc5[:,1].view(-1,1,1,1)*torch.mul(f_x4,sig_q4))
        sc6 = self.attn6(torch.cat((f_x5, sig_q5.expand(b, -1, -1, -1)),dim = 1))
        f_x6 = self.down_flow06(sc6[:,0].view(-1,1,1,1)*f_x5 + sc6[:,1].view(-1,1,1,1)*torch.mul(f_x5,sig_q5))

        f_u5 = self.up_flow05(f_x6)
        f_u4 = self.up_flow04(torch.cat((f_u5, f_x5), 1))
        f_u3 = self.up_flow03(torch.cat((f_u4, f_x4), 1))
        f_u2 = self.up_flow02(torch.cat((f_u3, f_x3), 1))
        f_u1 = self.up_flow01(torch.cat((f_u2, f_x2), 1))
        flow_map = self.location(torch.cat((f_u1, f_x1), 1))

        flow05 = self.to_flow05(f_u5)
        flow04 = self.to_flow04(f_u4)
        flow03 = self.to_flow03(f_u3)
        flow02 = self.to_flow02(f_u2)
        flow01 = self.to_flow01(f_u1)

        return flow_map, [flow01, flow02, flow03, flow04, flow05]

class Discriminator(BaseNetwork):
  def __init__(self, in_channels, use_sigmoid=False, use_sn=True, init_weights=True):
    super(Discriminator, self).__init__()
    self.use_sigmoid = use_sigmoid
    cnum = 64
    self.encoder1 = nn.Sequential(
      nn.Conv2d(in_channels=in_channels, out_channels=cnum,
        kernel_size=5, stride=2, padding=1, bias=False),
      nn.LeakyReLU(0.2, inplace=True))

    self.encoder2 = nn.Sequential(
      nn.Conv2d(in_channels=cnum, out_channels=cnum*2,
        kernel_size=5, stride=2, padding=1, bias=False),
      nn.LeakyReLU(0.2, inplace=True))

    self.encoder3 = nn.Sequential(
      nn.Conv2d(in_channels=cnum*2, out_channels=cnum*4,
        kernel_size=5, stride=2, padding=1, bias=False),
      nn.LeakyReLU(0.2, inplace=True))

    self.encoder4 = nn.Sequential(
      nn.Conv2d(in_channels=cnum*4, out_channels=cnum*8,
        kernel_size=5, stride=1, padding=1, bias=False),
      nn.LeakyReLU(0.2, inplace=True))

    emb_dim = 512

    self.query_conv1 = nn.Conv2d(in_channels=in_channels, out_channels=cnum, kernel_size=3, stride=2, padding=0)
    self.query_conv2 = nn.Conv2d(in_channels=cnum, out_channels=cnum*2, kernel_size=3, stride=2, padding=0)
    self.query_conv3 = nn.Conv2d(in_channels=cnum*2, out_channels=cnum*4, kernel_size=3, stride=2, padding=0)

    self.query1 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_02 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_04 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_06 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_08 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_2 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_4 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_6 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)
    self.query_8 = nn.Parameter(torch.ones(1, in_channels, 256 , 256), requires_grad=True)


    self.queries = [self.query1, self.query_02, self.query_04, self.query_06, self.query_08, self.query_2, self.query_4, self.query_6, self.query_8]

    self.attn4 = attention2d(cnum * 8, cnum)
    self.attn3 = attention2d(cnum * 4, cnum)
    self.attn2 = attention2d(cnum * 2, cnum)
    self.attn1 = attention2d(6, cnum)

    self.classifier = nn.Conv2d(in_channels=cnum*8, out_channels=1, kernel_size=5, stride=1, padding=1)
    if init_weights:
      self.init_weights()

  def copy_weight(self):
    with torch.no_grad():
      self.query_02.data.copy_(self.query1.data)
      self.query_04.data.copy_(self.query1.data)
      self.query_06.data.copy_(self.query1.data)
      self.query_08.data.copy_(self.query1.data)
      self.query_2.data.copy_(self.query1.data)
      self.query_4.data.copy_(self.query1.data)
      self.query_6.data.copy_(self.query1.data)
      self.query_8.data.copy_(self.query1.data)

  def forward(self, x, sigma):
    b,_,_,_ = x.shape
    sig_q = self.queries[sigma]

    sig_q1 = self.query_conv1(sig_q)
    sig_q2 = self.query_conv2(sig_q1)
    sig_q3 = self.query_conv3(sig_q2)

    sc1 = self.attn1(torch.cat((x, sig_q.expand(b, -1, -1, -1)),dim = 1))
    x1 = self.encoder1(sc1[:,0].view(-1,1,1,1)*x + sc1[:,1].view(-1,1,1,1)*torch.mul(x, sig_q))
    sc2 = self.attn2(torch.cat((x1, sig_q1.expand(b, -1, -1, -1)),dim = 1))
    x2 = self.encoder2(sc2[:,0].view(-1,1,1,1)*x1 + sc2[:,1].view(-1,1,1,1)*torch.mul(x1,sig_q1))
    sc3 = self.attn3(torch.cat((x2, sig_q2.expand(b, -1, -1, -1)),dim = 1))
    x3 = self.encoder3(sc3[:,0].view(-1,1,1,1)*x2 + sc3[:,1].view(-1,1,1,1)*torch.mul(x2,sig_q2))
    sc4 = self.attn4(torch.cat((x3, sig_q3.expand(b, -1, -1, -1)),dim = 1))
    x4 = self.encoder4(sc4[:,0].view(-1,1,1,1)*x3 + sc4[:,1].view(-1,1,1,1)*torch.mul(x3,sig_q3))

    label_x = self.classifier(x4)
    if self.use_sigmoid:
      label_x = torch.sigmoid(label_x)
    return label_x

