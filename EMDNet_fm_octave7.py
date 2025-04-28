import torch
import torch.nn as nn
import torch.nn.functional as F
#from net.ResNet import resnet50
from math import log
from net.Res2Net import res2net50_v1b_26w_4s
import pdb
from utils.psp import PSP_1, PSP_2, PSP_3, PSP_4, PSP_5
from utils.octave_conv import OctaveConv_v1, OctaveConv_v2, OctaveConv_v3, OctaveConv_v4, OctaveConv_v5, OctaveConv_v6, OctaveConv_v7

class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


###################################################################
# ################## Channel Attention Block ######################
###################################################################
class CA_Block(nn.Module):
    def __init__(self, in_dim):
        super(CA_Block, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : channel attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

###################################################################
# ################## Spatial Attention Block ######################
###################################################################
class SA_Block(nn.Module):
    def __init__(self, in_dim):
        super(SA_Block, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps (B X C X H X W)
            returns :
                out : spatial attentive features
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        import pdb
      #  pdb.set_trace()
        return out

###################################################################
class DSC_Block(nn.Module):
    def __init__(self, channel):
        super(DSC_Block, self).__init__()
        self.conv3x3_1 = ConvBNR(channel, channel, 3)
        self.conv3x3_2 = ConvBNR(channel, channel, 3)
        self.conv3x3_3 = ConvBNR(channel, channel, 3)
        self.conv3x3_4 = ConvBNR(channel, channel, 3)
        self.conv3x3_5 = ConvBNR(channel*3, channel, 3)
        self.conv1x1_1 = Conv1x1(channel*2, channel)
        self.conv1x1_2 = Conv1x1(channel*2, channel)
        self.conv1x1_3 = Conv1x1(channel*2, channel)
        self.conv1x1_4 = Conv1x1(channel*2, channel)
    def forward(self, x1 ,x2 ,x3):

        x12 = self.conv3x3_1(self.conv1x1_1(torch.cat((x1, x2), dim=1)) + x1)
        x32 = self.conv3x3_2(self.conv1x1_2(torch.cat((x3, x2), dim=1)) + x3)
        x1232 = self.conv3x3_3(self.conv1x1_3(torch.cat((x12, x32), dim=1)) + x12)
        x3212 = self.conv3x3_4(self.conv1x1_4(torch.cat((x32, x12), dim=1)) + x32)
        import pdb
        # pdb.set_trace()
        out = self.conv3x3_5(torch.cat((x1232, x3212, x2), dim=1))

        return out

###################################################################
# ################## Context Exploration Block ####################
###################################################################
class Context_Exploration_Block(nn.Module):
    def __init__(self, input_channels):
        super(Context_Exploration_Block, self).__init__()
        self.input_channels = input_channels
        self.channels_single = int(input_channels / 4)

        self.p1_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p3_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_channel_reduction = nn.Sequential(
            nn.Conv2d(self.input_channels, self.channels_single, 1, 1, 0),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p1 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 1, 1, 0),  # kernal[1,3,5,7]
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p1_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())  # dilation[1,2,4,8]

        self.p2 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 3, 1, 1),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p2_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 5, 1, 2),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p3_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.p4 = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, 7, 1, 3),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())
        self.p4_dc = nn.Sequential(
            nn.Conv2d(self.channels_single, self.channels_single, kernel_size=3, stride=1, padding=8, dilation=8),
            nn.BatchNorm2d(self.channels_single), nn.ReLU())

        self.fusion = nn.Sequential(nn.Conv2d(self.input_channels, self.input_channels, 1, 1, 0),
                                    nn.BatchNorm2d(self.input_channels), nn.ReLU())

    def forward(self, x):
        # 1
        p1_input = self.p1_channel_reduction(x)
        p1 = self.p1(p1_input)
        p1_dc = self.p1_dc(p1)
        # 2
        p2_input = self.p2_channel_reduction(x) + p1_dc
        p2 = self.p2(p2_input)
        p2_dc = self.p2_dc(p2)
        # 3
        p3_input = self.p3_channel_reduction(x) + p2_dc
        p3 = self.p3(p3_input)
        p3_dc = self.p3_dc(p3)
        # 4
        p4_input = self.p4_channel_reduction(x) + p3_dc
        p4 = self.p4(p4_input)
        p4_dc = self.p4_dc(p4)

        ce = self.fusion(torch.cat((p1_dc, p2_dc, p3_dc, p4_dc), 1))

        return ce

class DenseASPP(nn.Module):
    def __init__(self, channel):
        super(DenseASPP, self).__init__()
        self.conv3_1 = ConvBNR(channel // 4, channel // 4, 3)
        self.dconv5_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=2)
        self.dconv7_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=3)
        self.dconv9_1 = ConvBNR(channel // 4, channel // 4, 3, dilation=4)
        self.conv1_2 = Conv1x1(channel, channel)
        self.conv3_3 = ConvBNR(channel, channel, 3)

    def forward(self, x):
        xc = torch.chunk(x, 4, dim=1)
        x0 = self.conv3_1(xc[0] + xc[1])
        x1 = self.dconv5_1(xc[1] + x0 + xc[2])
        x2 = self.dconv7_1(xc[2] + x1 + xc[3])
        x3 = self.dconv9_1(xc[3] + x2)
        xx = self.conv1_2(torch.cat((x0, x1, x2, x3), dim=1))
        x = self.conv3_3(x + xx)

        return x

class GrounpEnhance(nn.Module):
    def __init__(self, channel):
        super(GrounpEnhance, self).__init__()
        # self.g_conv1 = nn.Conv2d(512, 128, kernel_size=3, groups=4, bias=False)
        # self.g_conv2 = nn.Conv2d(512, 128, kernel_size=3, groups=8, bias=False)
        # self.g_conv3 = nn.Conv2d(512, 128, kernel_size=3, groups=16, bias=False)
        # self.g_conv3 = nn.Conv2d(512, 128, kernel_size=3, groups=32, bias=False)

        self.g_conv4 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=7, dilation=7, groups=4, bias=False),
                                     nn.BatchNorm2d(128), nn.ReLU())
        self.g_conv3 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=5, dilation=5, groups=8, bias=False),
                                     nn.BatchNorm2d(128), nn.ReLU())
        self.g_conv2 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=3, dilation=3, groups=16, bias=False),
                                     nn.BatchNorm2d(128), nn.ReLU())
        self.g_conv1 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1, dilation=1, groups=32, bias=False),
                                     nn.BatchNorm2d(128), nn.ReLU())

    def forward(self, x):
        g1 = self.g_conv1(x)
        g2 = self.g_conv2(x)
        g3 = self.g_conv3(x)
        g4 = self.g_conv4(x)
        # import pdb
        # pdb.set_trace()

        out = torch.cat((g1, g2, g3, g4), 1)
        return out

###################################################################
# ##################### Positioning Module ########################
###################################################################
class Positioning(nn.Module):
    def __init__(self, channel):  # 128
        super(Positioning, self).__init__()
        self.channel = channel
        self.denseaspp = DenseASPP(channel)
        #self.ge = GrounpEnhance(self.channel)
        # self.cab = CA_Block(self.channel)
        self.sab = SA_Block(self.channel)
        self.cab = CA_Block(self.channel)
        # self.map = nn.Conv2d(self.channel, 1, 7, 1, 3)
        self.dsc1 = DSC_Block(self.channel)
        self.dsc2 = DSC_Block(self.channel)
        self.dsc3 = DSC_Block(self.channel)
        self.conv3_3 = ConvBNR(channel*3, channel*4, 3)
        self.conv1x1_1 = nn.Sequential(nn.Conv2d(2048, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv1x1_2 = nn.Sequential(nn.Conv2d(2048, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())
        self.conv1x1_3 = nn.Sequential(nn.Conv2d(2048, channel, 3, 1, 1), nn.BatchNorm2d(channel), nn.ReLU())

    def forward(self, x):

        x_1 = self.conv1x1_1(x)
        x_2 = self.conv1x1_2(x)
        x_3 = self.conv1x1_3(x)
       # x = self.ge(x)
        x1 = self.denseaspp(x_1)
        # cab = self.cab(x)
        x2= self.sab(x_2)
        x3 = self.cab(x_3)
        x123 =self.dsc2(x1,x2,x3)
        x132 = self.dsc3(x1, x3, x2)
        x213 = self.dsc1(x2, x1, x3)
        out = torch.cat((x123, x132, x213), 1)
        out = self.conv3_3(out)
        # map = self.map(sab)

        return out  # , map

###################################################################
# ######################## Focus Module ###########################
###################################################################
class Focus(nn.Module):
    def __init__(self, channel1, channel2):  # [64, 128, 256] [128, 256, 512]
        super(Focus, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU())  # nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.Sigmoid())
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        # in_channels=self.channel1, out_channels, kernel_size
        self.octaveconv7_1 = OctaveConv_v7(self.channel1, self.channel1, (3,3))
        self.octaveconv7_2= OctaveConv_v7(self.channel1, self.channel1, (3,3))

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

        self.convadd1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.pre4 = nn.Conv2d(512, 1, 7, 1, 3)
        self.conv1x1 = nn.Sequential(nn.Conv2d(512, channel1, 3, 1, 1), nn.BatchNorm2d(channel1), nn.ReLU())
        self.conv3_3_1 = ConvBNR(channel1 + 512, channel1, 3)
        self.conv3_3_2 = ConvBNR(channel1 + 512, channel1, 3)
        self.conv3_3_3 = ConvBNR(channel1, channel1, 3)
        self.conv3_3_4 = ConvBNR(channel1, channel1, 3)
    def forward(self, x, y, in_map, h):  # x low 256 y high 512->256
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction
        # h: global position feature

        up = self.up(y)  # conv7x7 chl H -> chl L
        up = nn.functional.interpolate(up, scale_factor=2, mode="bilinear", align_corners=True)  # upx2

        in_map = nn.functional.interpolate(in_map, scale_factor=2, mode="bilinear", align_corners=True)  # upx2
        input_map = self.input_map(in_map)  # sigmoid

        f_feature = x + up
        b_feature = x * input_map


        f_feature = self.conv3_3_3(self.octaveconv7_1(f_feature, h) + f_feature)
        b_feature = self.conv3_3_4(self.octaveconv7_2(b_feature, h) + b_feature)


        h = F.interpolate(h, size=f_feature.size()[2:], mode='bilinear', align_corners=True)
        fp = self.conv3_3_1(torch.cat((f_feature, h), 1))
        h = F.interpolate(h, size=f_feature.size()[2:], mode='bilinear', align_corners=True)
        fn = self.conv3_3_2(torch.cat((b_feature, h), 1))

        refine1 = self.alpha * fp
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = self.beta * fn
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)
        output_map = self.output_map(refine1 + refine2)

        return refine2, output_map


class Focus1(nn.Module):
    def __init__(self, channel1, channel2):  # [64, 128, 256] [128, 256, 512]
        super(Focus1, self).__init__()
        self.channel1 = channel1
        self.channel2 = channel2

        self.up = nn.Sequential(nn.Conv2d(self.channel2, self.channel1, 7, 1, 3),
                                nn.BatchNorm2d(self.channel1), nn.ReLU())  # nn.UpsamplingBilinear2d(scale_factor=2))

        self.input_map = nn.Sequential(nn.Sigmoid())
        self.output_map = nn.Conv2d(self.channel1, 1, 7, 1, 3)

        # in_channels=self.channel1, out_channels, kernel_size
        self.octaveconv2_1 = OctaveConv_v2(self.channel1, self.channel1, (3,3))
        self.octaveconv2_2= OctaveConv_v2(self.channel1, self.channel1, (3,3))

        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))
        self.bn1 = nn.BatchNorm2d(self.channel1)
        self.relu1 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(self.channel1)
        self.relu2 = nn.ReLU()

        self.convadd1 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.pre4 = nn.Conv2d(512, 1, 7, 1, 3)
        self.conv1x1 = nn.Sequential(nn.Conv2d(512, channel1, 3, 1, 1), nn.BatchNorm2d(channel1), nn.ReLU())
        self.conv3_3_1 = ConvBNR(channel1 + 512, channel1, 3)
        self.conv3_3_2 = ConvBNR(channel1 + 512, channel1, 3)
        self.conv3_3_3 = ConvBNR(channel1, channel1, 3)
        self.conv3_3_4 = ConvBNR(channel1, channel1, 3)
    def forward(self, x, y, in_map, h):  # x low 256 y high 512->256
        # x; current-level features
        # y: higher-level features
        # in_map: higher-level prediction
        # h: global position feature

        up = self.up(y)  # conv7x7 chl H -> chl L
        up = nn.functional.interpolate(up, scale_factor=2, mode="bilinear", align_corners=True)  # upx2

        in_map = nn.functional.interpolate(in_map, scale_factor=2, mode="bilinear", align_corners=True)  # upx2
        input_map = self.input_map(in_map)  # sigmoid

        f_feature = x + up
        b_feature = x * input_map


        f_feature = self.conv3_3_3(self.octaveconv2_1(f_feature, h) + f_feature)
        b_feature = self.conv3_3_4(self.octaveconv2_2(b_feature, h) + b_feature)


        h = F.interpolate(h, size=f_feature.size()[2:], mode='bilinear', align_corners=True)
        fp = self.conv3_3_1(torch.cat((f_feature, h), 1))
        h = F.interpolate(h, size=f_feature.size()[2:], mode='bilinear', align_corners=True)
        fn = self.conv3_3_2(torch.cat((b_feature, h), 1))

        refine1 = self.alpha * fp
        refine1 = self.bn1(refine1)
        refine1 = self.relu1(refine1)

        refine2 = self.beta * fn
        refine2 = self.bn2(refine2)
        refine2 = self.relu2(refine2)
        output_map = self.output_map(refine1 + refine2)

        return refine2, output_map
###################################################################

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # if self.training:
        # self.initialize_weights()

        # channel reduction
        # self.cr4 = nn.Sequential(nn.Conv2d(2048, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.cr3 = nn.Sequential(nn.Conv2d(1024, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.cr2 = nn.Sequential(nn.Conv2d(512, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.cr1 = nn.Sequential(nn.Conv2d(256, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU())

        # positioning
        self.positioning = Positioning(128)

        # focus
        self.focus3 = Focus(256, 512)
        self.focus2 = Focus(128, 256)
        self.focus1 = Focus1(64, 128)

        # self.GECA = nn.Sequential(nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.pre4 = nn.Conv2d(512, 1, 7, 1, 3)

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        # x: [batch_size, channel=3, h, w]
        layer1, layer2, layer3, layer4 = self.resnet(x)
         # [-1, 64, h/2, w/2]
         # [-1, 256, h/4, w/4]
         # [-1, 512, h/8, w/8]
         # [-1, 1024, h/16, w/16]
         # [-1, 2048, h/32, w/32]

        # channel reduction
        # cr4 = self.cr4(layer4)  # [-1, 2048, 224, 224] -> [-1, 512, 224, 224]
        cr3 = self.cr3(layer3)  # [-1, 1024, 224, 224] -> [-1, 256, 224, 224]
        cr2 = self.cr2(layer2)  # [-1, 512, 224, 224] -> [-1, 128, 224, 224]
        cr1 = self.cr1(layer1)  # [-1, 256, 224, 224] -> [-1, 64, 224, 224]
        '''
        print('-cr1', cr1.shape)
        print('-cr2', cr2.shape)
        print('-cr3', cr3.shape)
        print('-cr4', cr4.shape)
        print('----------------')

        -cr1 torch.Size([16, 64, 112, 112])
        -cr2 torch.Size([16, 128, 56, 56])
        -cr3 torch.Size([16, 256, 28, 28])
        -cr4 torch.Size([16, 512, 14, 14])
        '''

        # positioning
        positioning = self.positioning(layer4)

        import pdb
        #pdb.set_trace()
        s = positioning
        predict4 = self.pre4(positioning)

        # focus
        '''
        print('-cr3', cr3.shape)
        print('-positioning', positioning.shape)
        print('-predict4', predict4.shape)'''

        # print('-cr3', cr3.shape)
        # print('-positioning', positioning.shape)
        
        #print('-focus3')

        focus3, predict3 = self.focus3(cr3, positioning, predict4, s)
        #print('-focus2')
        focus2, predict2 = self.focus2(cr2, focus3, predict3, s)
        #print('-focus1')
        focus1, predict1 = self.focus1(cr1, focus2, predict2, s)

        # rescale
        predict4 = F.interpolate(predict4, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict3 = F.interpolate(predict3, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict2 = F.interpolate(predict2, size=x.size()[2:], mode='bilinear', align_corners=True)
        predict1 = F.interpolate(predict1, size=x.size()[2:], mode='bilinear', align_corners=True)

        if self.training:
            return predict4, predict3, predict2, predict1

        return torch.sigmoid(predict4), torch.sigmoid(predict3), torch.sigmoid(predict2), torch.sigmoid(
            predict1)
