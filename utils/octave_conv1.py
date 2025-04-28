
import torch
import torch.nn as nn
import torch.nn.functional as F
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

class OctaveConv_v2(nn.Module):  # self.channel1, self.channel1
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = 'nearest', weights = None):
        super(OctaveConv_v2, self).__init__()
        # if weights is None:
        self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        # else:
        #     self.weights = weights
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias is not False:
            self.bias = bias
        else:
            self.bias = torch.zeros(out_channels).cuda()
        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

        self.Conv1x1 = Conv1x1(512, int(in_channels/2))
        self.Conv1x1_ = Conv1x1(in_channels, int(in_channels/2))

    def forward(self, x, y):  # f_feature, h
        y = self.Conv1x1(y)  #
        # X_h, X_l = torch.chunk(x,2,1)   # 44x44
        X_h = self.Conv1x1_(x)
        #X_l=F.upsample(X_l,scale_factor=0.5,mode='nearest')
        X_l = y # 11x11
        #size = X_h.size()[2:]
        size=X_h.size()[2:]
        w = int(size[0]/2)
        import pdb
        # pdb.set_trace()
        X_l = F.interpolate(X_l, (w,w), mode='bilinear')  # BCHW ->[HW] 22x22

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)

        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.upsample(X_l2h, scale_factor=2, mode='nearest')#, **self.up_kwargs)

        X_h = X_h2h + X_l2h
        X_l = X_l2l + X_h2l
        X_l = F.upsample(X_l, scale_factor=2, mode='nearest')

        X = torch.cat((X_h, X_l), dim=1)

        return X
#octaveconv = OctaveConv_v2(kernel_size=(3, 3), in_channels=in_channel, out_channels=out_channel, bias=conv_bias, stride=1, weights=conv_weight)

#y = octaveconv(x)
'''
=============================================================================================
'''

class OctaveConv_v1(nn.Module):  # self.channel1, self.channel1
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = 'nearest', weights = None):
        super(OctaveConv_v1, self).__init__()
        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size[0], kernel_size[1]))
        else:
            self.weights = weights
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        if bias is not False:
            self.bias = bias
        else:
            self.bias = torch.zeros(out_channels).cuda()
        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2,2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, x):
        X_h, X_l = torch.chunk(x,2,1)
        X_l=F.upsample(X_l,scale_factor=0.5,mode='nearest')

        if self.stride ==2:
            X_h, X_l = self.h2g_pool(X_h), self.h2g_pool(X_l)

        X_h2l = self.h2g_pool(X_h)

        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)

        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)

        X_l2h = F.upsample(X_l2h, scale_factor=2, **self.up_kwargs)

        X_h = X_h2h + X_l2h
        X_l = X_l2l + X_h2l
        X_l = F.upsample(X_l, scale_factor=2, mode='nearest')

        X = torch.cat((X_h, X_l), dim=1)

        return X
#octaveconv = OctaveConv_v1(kernel_size=(3, 3), in_channels=in_channel, out_channels=out_channel, bias=conv_bias, stride=1, weights=conv_weight)
#y = octaveconv(x)


