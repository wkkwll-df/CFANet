
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


class SA_Block_PoolKV(nn.Module):
    '''
    Multi-scale Spatial Attention (scales are 1,3,5 follow LKD)
    outputs of backbone: 256 512 1024 2048
    channel unify: 64 64 64 64
    '''

    def __init__(self, inc):  # 512
        super(SA_Block_PoolKV, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)

        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        '''
        self.adp_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.adp_pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.adp_pool5 = nn.AdaptiveAvgPool2d((5, 5))'''
        self.avg_pool = nn.AvgPool2d((2, 2), stride=(2, 2))

    def forward(self, x):
        """
        x :       input feature maps (b c h w)
        returns : spatial attentive features (b c h w)
        """
        # get shape
        B, C, height, width = x.size()  # 16 512 H W

        ### Q
        q = self.query_conv(x).view(B, -1, width * height).permute(0, 2, 1)  # BNC
        ### K
        k = self.key_conv(x).view(B, -1, width, height)  # BCHW
        k = self.avg_pool(k).view(B, -1, int(width * height / 4))  # B C (N/4)
        ### V
        v = self.value_conv(x).view(B, -1, width, height)  # BCHW
        v = self.avg_pool(v).view(B, -1, int(width * height / 4))  # BC(N/4)

        ### soft(QK)
        energy = torch.bmm(q, k)  # BNC x BC(N/4) = BN(N/4)
        attention = self.softmax(energy)  # Softs

        # SA_poolKV = VxA
        out = torch.bmm(v, attention.permute(0, 2, 1))  # BC(N/4) x B(N/4)N = BCN
        # reshape
        out = out.view(B, C, height, width)  # BCHW
        # res
        out = self.gamma * out + x
        return out


class SA_Block_CrossPoolKV(nn.Module):
    '''
    Multi-scale Spatial Attention (scales are 1,3,5 follow LKD)
    outputs of backbone: 256 512 1024 2048
    channel unify: 64 64 64 64
    '''

    def __init__(self, inc):  # 512
        super(SA_Block_CrossPoolKV, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)

        self.Query_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.Key_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.Value_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)

        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        '''
        self.adp_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.adp_pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.adp_pool5 = nn.AdaptiveAvgPool2d((5, 5))'''
        self.avg_pool = nn.AvgPool2d((2, 2), stride=(2, 2))

    def forward(self, x):
        """
        x :       input feature maps (b c h w)
        returns : spatial attentive features (b c h w)
        """
        # get shape
        B, C, height, width = x.size()  # 16 512 H W

        ### 1 prepare q K V
        # q: BCHW->BChw->BC(HW/4)->B(HW/4)C
        q = self.query_conv(x).view(B, -1, width, height)
        q = self.avg_pool(q).view(B, -1, int(width * height / 4)).permute(0, 2, 1)
        # K: BCHW->BC(HW)
        K = self.Key_conv(x).view(B, -1, width * height)
        # V: BCHW->BC(HW)
        V = self.Value_conv(x).view(B, -1, width * height)

        ### 1 softmax(q*K)*V
        # q*K: B(HW/4)C * BC(HW) = B(HW/4)(HW)
        energy = torch.bmm(q, K)
        attention = self.softmax(energy)  # Softs
        # v*a: BC(HW) * B(HW)(HW/4) -> BC(HW/4)
        out = torch.bmm(V, attention.permute(0, 2, 1))

        ### 2 prepare Q k v
        # Q: BCHW->BC(HW)->B(HW)C
        Q = self.Query_conv(x).view(B, -1, width * height).permute(0, 2, 1)  # BNC
        # reshape: BC(HW/4)->BChw
        out = out.view(B, C, int(height / 2), int(width / 2))
        # k: BChw->BC(hw)
        k = self.key_conv(out).view(B, -1, int(height / 2) * int(width / 2))  # BCHW
        # v: BChw->BC(hw)
        v = self.value_conv(out).view(B, -1, int(height / 2) * int(width / 2))  # BCHW

        ### 2 softmax(Q*k)*v
        # Q*k: B(HW)C*BC(HW/4)->(HW)(HW/4)
        energy = torch.bmm(Q, k)
        attention = self.softmax(energy)  # Softs
        # v*A: BC(HW/4)*(HW/4)(HW) -> BC(HW)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        # reshape: BCHW
        out = out.view(B, C, height, width)  # BCHW
        # res
        out = self.gamma * out + x
        return out


class SA_Block_Ancher(nn.Module):
    '''
    Multi-scale Spatial Attention (scales are 1,3,5 follow LKD)
    outputs of backbone: 256 512 1024 2048
    channel unify: 64 64 64 64
    '''

    def __init__(self, inc):  # 512
        super(SA_Block_Ancher, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)

        self.Query_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.Key_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.Value_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)

        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        '''
        self.adp_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.adp_pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.adp_pool5 = nn.AdaptiveAvgPool2d((5, 5))'''
        self.avg_pool = nn.AvgPool2d((2, 2), stride=(2, 2))

    def forward(self, x):  # q和attention需要转置
        """
        x :       input feature maps (b c h w)
        returns : spatial attentive features (b c h w)
        """
        # get shape
        B, C, height, width = x.size()  # 16 512 H W

        ### 1 prepare q K V
        # q: BCHW->BChw->BC(HW/4)->B(HW/4)C
        q = self.query_conv(x).view(B, -1, width, height)
        q_ = self.avg_pool(q).view(B, -1, int(width * height / 4))  # BC(HW/4)
        q = q_.permute(0, 2, 1)
        # K: BCHW->BC(HW)
        K = self.Key_conv(x).view(B, -1, width * height)
        # V: BCHW->BC(HW)
        V = self.Value_conv(x).view(B, -1, width * height)

        ### 2 softmax(q*K)*V
        # q*K: B(HW/4)C * BC(HW) = B(HW/4)(HW)
        energy = torch.bmm(q, K)
        attention = self.softmax(energy)  # Softs
        # v*a: BC(HW) * B(HW)(HW/4) -> BC(HW/4)
        out = torch.bmm(V, attention.permute(0, 2, 1))  # BC(HW/4)

        ### 3 prepare Q
        # Q: BCHW->BC(HW)->B(HW)C
        Q = self.Query_conv(x).view(B, -1, width * height).permute(0, 2, 1)  # BNC

        ### 4 Q*qt
        # Q*q: B(HW)C*BC(HW/4)->(HW)(HW/4)
        energy = torch.bmm(Q, q_)
        attention = self.softmax(energy)  # Softs

        # v*A: BC(HW/4)*(HW/4)(HW) -> BC(HW)
        out = torch.bmm(out, attention.permute(0, 2, 1))
        # reshape: BCHW
        out = out.view(B, C, height, width)  # BCHW
        # res
        out = self.gamma * out + x
        return out


class SA_Block_CrossPoolKV_res(nn.Module):
    '''
    Multi-scale Spatial Attention (scales are 1,3,5 follow LKD)
    outputs of backbone: 256 512 1024 2048
    channel unify: 64 64 64 64
    '''

    def __init__(self, inc):  # 512
        super(SA_Block_CrossPoolKV_res, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)

        self.Query_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.Key_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.Value_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)

        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        '''
        self.adp_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.adp_pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.adp_pool5 = nn.AdaptiveAvgPool2d((5, 5))'''
        self.avg_pool = nn.AvgPool2d((2, 2), stride=(2, 2))

    def forward(self, x):
        """
        x :       input feature maps (b c h w)
        returns : spatial attentive features (b c h w)
        """
        # get shape
        B, C, height, width = x.size()  # 16 512 H W

        ### 1 prepare q K V
        # q: BCHW->BChw->BC(HW/4)->B(HW/4)C
        q = self.query_conv(x).view(B, -1, width, height)
        q = self.avg_pool(q).view(B, -1, int(width * height / 4)).permute(0, 2, 1)  # BNC
        # K: BCHW->BC(HW)
        K = self.Key_conv(x).view(B, -1, width * height)
        # V: BCHW->BC(HW)
        V = self.Value_conv(x).view(B, -1, width * height)

        ### 1 softmax(q*K)*V
        # q*K: B(HW/4)C * BC(HW) = B(HW/4)(HW)
        energy = torch.bmm(q, K)
        attention = self.softmax(energy)  # Softs
        # v*a: BC(HW) * B(HW)(HW/4) -> BC(HW/4)
        out = torch.bmm(V, attention.permute(0, 2, 1))
        out = out.permute(0, 2, 1)  # BCN -> BNC
        out = out + q
        out = out.permute(0, 2, 1)  # BNC -> BCN

        ### 2 prepare Q k v
        # Q: BCHW->BC(HW)->B(HW)C
        Q = self.Query_conv(x).view(B, -1, width * height).permute(0, 2, 1)  # BNC
        # reshape: BC(HW/4)->BChw
        out = out.view(B, C, int(height / 2), int(width / 2))
        # k: BChw->BC(hw)
        k = self.key_conv(out).view(B, -1, int(height / 2) * int(width / 2))  # BCHW
        # v: BChw->BC(hw)
        v = self.value_conv(out).view(B, -1, int(height / 2) * int(width / 2))  # BCHW

        ### 2 softmax(Q*k)*v
        # Q*k: B(HW)C*BC(HW/4)->(HW)(HW/4)
        energy = torch.bmm(Q, k)
        attention = self.softmax(energy)  # Softs
        # v*A: BC(HW/4)*(HW/4)(HW) -> BC(HW)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        # reshape: BCHW
        out = out.view(B, C, height, width)  # BCHW
        # res
        out = self.gamma * out + x
        return out


class SA_Block_CrossPoolKV_res3x3(nn.Module):
    '''
    Multi-scale Spatial Attention (scales are 1,3,5 follow LKD)
    outputs of backbone: 256 512 1024 2048
    channel unify: 64 64 64 64
    '''

    def __init__(self, inc):  # 512
        super(SA_Block_CrossPoolKV_res3x3, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        # 3x3
        self.key_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, stride=1, padding=1)
        self.value_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=3, stride=1, padding=1)

        self.Query_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.Key_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)
        self.Value_conv = nn.Conv2d(in_channels=inc, out_channels=inc, kernel_size=1)

        self.gamma = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        '''
        self.adp_pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.adp_pool3 = nn.AdaptiveAvgPool2d((3, 3))
        self.adp_pool5 = nn.AdaptiveAvgPool2d((5, 5))'''
        self.avg_pool = nn.AvgPool2d((2, 2), stride=(2, 2))

    def forward(self, x):
        """
        x :       input feature maps (b c h w)
        returns : spatial attentive features (b c h w)
        """
        # get shape
        B, C, height, width = x.size()  # 16 512 H W

        ### 1 prepare q K V
        # q: BCHW->BChw->BC(HW/4)->B(HW/4)C
        q = self.query_conv(x).view(B, -1, width, height)
        q = self.avg_pool(q).view(B, -1, int(width * height / 4)).permute(0, 2, 1)  # BNC
        # K: BCHW->BC(HW)
        K = self.Key_conv(x).view(B, -1, width * height)
        # V: BCHW->BC(HW)
        V = self.Value_conv(x).view(B, -1, width * height)

        ### 1 softmax(q*K)*V
        # q*K: B(HW/4)C * BC(HW) = B(HW/4)(HW)
        energy = torch.bmm(q, K)
        attention = self.softmax(energy)  # Softs
        # v*a: BC(HW) * B(HW)(HW/4) -> BC(HW/4)
        out = torch.bmm(V, attention.permute(0, 2, 1))
        out = out.permute(0, 2, 1)  # BCN -> BNC
        out = out + q
        out = out.permute(0, 2, 1)  # BNC -> BCN

        ### 2 prepare Q k v
        # Q: BCHW->BC(HW)->B(HW)C
        Q = self.Query_conv(x).view(B, -1, width * height).permute(0, 2, 1)  # BNC
        # reshape: BC(HW/4)->BChw
        out = out.view(B, C, int(height / 2), int(width / 2))
        # k: BChw->BC(hw)
        k = self.key_conv(out).view(B, -1, int(height / 2) * int(width / 2))  # BCHW
        # v: BChw->BC(hw)
        v = self.value_conv(out).view(B, -1, int(height / 2) * int(width / 2))  # BCHW

        ### 2 softmax(Q*k)*v
        # Q*k: B(HW)C*BC(HW/4)->(HW)(HW/4)
        energy = torch.bmm(Q, k)
        attention = self.softmax(energy)  # Softs
        # v*A: BC(HW/4)*(HW/4)(HW) -> BC(HW)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        # reshape: BCHW
        out = out.view(B, C, height, width)  # BCHW
        # res
        out = self.gamma * out + x
        return out
###################################################################

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

        self.Conv1x1_h = Conv1x1(512, int(in_channels/2))
        self.Conv1x1_x = Conv1x1(in_channels, int(in_channels/2))

    def forward(self, x, h):
        '''
        352->   88--44--22--11(官方)
        448->  112--56--28--14(当前)
        通道->   64-128-256-512
        输入:
            f_feature -> x (假定x是第2层特征(B,128,56,56)) C=128
            h -> h (B,512,14,14)
        '''
        ### 1x1卷积统一通道为C/2
        X_l = self.Conv1x1_h(h)  # Chn:512->C/2 | HW:14,14
        X_h = self.Conv1x1_x(x)  # Chn:C->C/2   | HW:56,56

        ### 获得当前层特征size的2分之1,作为低频尺度
        size=X_h.size()[2:]  # BCHW->[HW]
        H_size = int(size[0]/2)  # W/2=28
        W_size = int(size[1]/2)  # W/2=28

        ### 低频特征up,尺度变为28,28
        X_l = F.interpolate(X_l, (H_size,H_size), mode='bilinear')  # BCHW ->[HW] 28,28

        ### X_l X_h 准备就绪
        ### 通道对半分
        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        ### 高频处理 h2h, l2h
        # h2h: conv3x3
        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)
        # l2h: conv3x3+up
        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :, :], self.bias[0:end_h_y], 1,
                         self.padding, self.dilation, self.groups)
        X_l2h = F.upsample(X_l2h, scale_factor=2, mode='nearest')  # , **self.up_kwargs)

        ### 低频处理 l2l, h2l
        # l2l: conv3x3
        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)
        # h2l: avgpool+conv
        X_h2l = self.h2g_pool(X_h)
        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        ### 高频+
        X_h = X_h2h + X_l2h
        ### 低频+
        X_l = X_l2l + X_h2l

        ### 合并高低频cat
        X_l = F.upsample(X_l, scale_factor=2, mode='nearest')
        X = torch.cat((X_h, X_l), dim=1)

        return X
#octaveconv = OctaveConv_v2(kernel_size=(3, 3), in_channels=in_channel, out_channels=out_channel, bias=conv_bias, stride=1, weights=conv_weight)

#y = octaveconv(x)
'''
=============================================================================================
'''
class OctaveConv_v3(nn.Module):  # self.channel1, self.channel1
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = 'nearest', weights = None):
        super(OctaveConv_v3, self).__init__()
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

        self.Conv1x1_h = Conv1x1(512, int(in_channels/2))
        self.Conv1x1_x = Conv1x1(in_channels, int(in_channels/2))

        self.sa = SA_Block(int(self.in_channels/2))
    def forward(self, x, h):
        '''
        352->   88--44--22--11(官方)
        448->  112--56--28--14(当前)
        通道->   64-128-256-512
        输入:
            f_feature -> x (假定x是第2层特征(B,128,56,56)) C=128
            h -> h (B,512,14,14)
        '''
        ### 1x1卷积统一通道为C/2
        X_l = self.Conv1x1_h(h)  # Chn:512->C/2 | HW:14,14
        X_h = self.Conv1x1_x(x)  # Chn:C->C/2   | HW:56,56

        ### 获得当前层特征size的2分之1,作为低频尺度
        size=X_h.size()[2:]  # BCHW->[HW]
        H_size = int(size[0]/2)  # W/2=28
        W_size = int(size[1]/2)  # W/2=28

        ### 低频特征up,尺度变为28,28
        X_l = F.interpolate(X_l, (H_size,H_size), mode='bilinear')  # BCHW ->[HW] 28,28

        ### X_l X_h 准备就绪
        ### 通道对半分
        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        ### 高频处理 h2h, l2h
        # h2h: conv3x3
        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)
        # l2h: conv3x3+up
        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :, :], self.bias[0:end_h_y], 1,
                         self.padding, self.dilation, self.groups)
        X_l2h = F.upsample(X_l2h, scale_factor=2, mode='nearest')  # , **self.up_kwargs)

        ### 低频处理 l2l, h2l
        # l2l: conv3x3
        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)
        # h2l: avgpool+conv
        X_h2l = self.h2g_pool(X_h)
        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        ### 高频+
        X_h = X_h2h + X_l2h
        ### 低频+
        X_l = X_l2l + X_h2l
        X_l = self.sa( X_l)

        ### 合并高低频cat
        X_l = F.upsample(X_l, scale_factor=2, mode='nearest')
        X = torch.cat((X_h, X_l), dim=1)

        return X


'''
=============================================================================================
'''

class OctaveConv_v4(nn.Module):  # self.channel1, self.channel1
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = 'nearest', weights = None):
        super(OctaveConv_v4, self).__init__()
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

        self.Conv1x1_h = Conv1x1(512, int(in_channels/2))
        self.Conv1x1_x = Conv1x1(in_channels, int(in_channels/2))
        self.conv3x3_1 = ConvBNR(int(in_channels/2), int(in_channels/2), 3)
        self.sa = SA_Block_PoolKV(int(in_channels/2))
    def forward(self, x, h):
        '''
        352->   88--44--22--11(官方)
        448->  112--56--28--14(当前)
        通道->   64-128-256-512
        输入:
            f_feature -> x (假定x是第2层特征(B,128,56,56)) C=128
            h -> h (B,512,14,14)
        '''
        ### 1x1卷积统一通道为C/2
        X_l = self.Conv1x1_h(h)  # Chn:512->C/2 | HW:14,14
        X_h = self.Conv1x1_x(x)  # Chn:C->C/2   | HW:56,56

        ### 获得当前层特征size的2分之1,作为低频尺度
        size=X_h.size()[2:]  # BCHW->[HW]
        H_size = int(size[0]/2)  # W/2=28
        W_size = int(size[1]/2)  # W/2=28

        ### 低频特征up,尺度变为28,28
        X_l = F.interpolate(X_l, (H_size,H_size), mode='bilinear')  # BCHW ->[HW] 28,28

        ### X_l X_h 准备就绪
        ### 通道对半分
        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        ### 高频处理 h2h, l2h
        # h2h: conv3x3
        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)
        # l2h: conv3x3+up
        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :, :], self.bias[0:end_h_y], 1,
                         self.padding, self.dilation, self.groups)
        X_l2h = F.upsample(X_l2h, scale_factor=2, mode='nearest')  # , **self.up_kwargs)

        ### 低频处理 l2l, h2l
        # l2l: conv3x3
        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)
        # h2l: avgpool+conv
        X_h2l = self.h2g_pool(X_h)
        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        ### 高频+
        X_h = X_h2h + X_l2h
        ### 低频+
        X_l = X_l2l + X_h2l
        X_l = self.conv3x3_1( X_l)
        X_l = self.sa( X_l)

        ### 合并高低频cat
        X_l = F.upsample(X_l, scale_factor=2, mode='nearest')
        X = torch.cat((X_h, X_l), dim=1)

        return X
'''
=============================================================================================
'''

class OctaveConv_v5(nn.Module):  # self.channel1, self.channel1
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = 'nearest', weights = None):
        super(OctaveConv_v5, self).__init__()
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

        self.Conv1x1_h = Conv1x1(512, int(in_channels/2))
        self.Conv1x1_x = Conv1x1(in_channels, int(in_channels/2))
        self.conv3x3_1 = ConvBNR(int(in_channels/2), int(in_channels/2), 3)
        self.sa = SA_Block_PoolKV(int(self.in_channels/2))
    def forward(self, x, h):
        '''
        352->   88--44--22--11(官方)
        448->  112--56--28--14(当前)
        通道->   64-128-256-512
        输入:
            f_feature -> x (假定x是第2层特征(B,128,56,56)) C=128
            h -> h (B,512,14,14)
        '''
        ### 1x1卷积统一通道为C/2
        X_l = self.Conv1x1_h(h)  # Chn:512->C/2 | HW:14,14
        X_h = self.Conv1x1_x(x)  # Chn:C->C/2   | HW:56,56

        ### 获得当前层特征size的2分之1,作为低频尺度
        size=X_h.size()[2:]  # BCHW->[HW]
        H_size = int(size[0]/2)  # W/2=28
        W_size = int(size[1]/2)  # W/2=28

        ### 低频特征up,尺度变为28,28
        X_l = F.interpolate(X_l, (H_size,H_size), mode='bilinear')  # BCHW ->[HW] 28,28

        ### X_l X_h 准备就绪
        ### 通道对半分
        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        ### 高频处理 h2h, l2h
        # h2h: conv3x3
        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)
        # l2h: conv3x3+up
        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :, :], self.bias[0:end_h_y], 1,
                         self.padding, self.dilation, self.groups)
        X_l2h = F.upsample(X_l2h, scale_factor=2, mode='nearest')  # , **self.up_kwargs)

        ### 低频处理 l2l, h2l
        # l2l: conv3x3
        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)
        # h2l: avgpool+conv
        X_h2l = self.h2g_pool(X_h)
        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        ### 高频+
        X_h = X_h2h + X_l2h
        X_h = self.conv3x3_1( X_h)
        X_h = self.sa( X_h)
        ### 低频+
        X_l = X_l2l + X_h2l


        ### 合并高低频cat
        X_l = F.upsample(X_l, scale_factor=2, mode='nearest')
        X = torch.cat((X_h, X_l), dim=1)

        return X


class OctaveConv_v6(nn.Module):  # self.channel1, self.channel1
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = 'nearest', weights = None):
        super(OctaveConv_v6, self).__init__()
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

        self.Conv1x1_h = Conv1x1(512, int(in_channels/2))
        self.Conv1x1_x = Conv1x1(in_channels, int(in_channels/2))
        self.conv3x3_1 = ConvBNR(int(in_channels/2), int(in_channels/2), 3)
        self.sa = SA_Block(int(self.in_channels/2))
    def forward(self, x, h):
        '''
        352->   88--44--22--11(官方)
        448->  112--56--28--14(当前)
        通道->   64-128-256-512
        输入:
            f_feature -> x (假定x是第2层特征(B,128,56,56)) C=128
            h -> h (B,512,14,14)
        '''
        ### 1x1卷积统一通道为C/2
        X_l = self.Conv1x1_h(h)  # Chn:512->C/2 | HW:14,14
        X_h = self.Conv1x1_x(x)  # Chn:C->C/2   | HW:56,56

        ### 获得当前层特征size的2分之1,作为低频尺度
        size=X_h.size()[2:]  # BCHW->[HW]
        H_size = int(size[0]/2)  # W/2=28
        W_size = int(size[1]/2)  # W/2=28

        ### 低频特征up,尺度变为28,28
        X_l = F.interpolate(X_l, (H_size,H_size), mode='bilinear')  # BCHW ->[HW] 28,28

        ### X_l X_h 准备就绪
        ### 通道对半分
        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        ### 高频处理 h2h, l2h
        # h2h: conv3x3
        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)
        # l2h: conv3x3+up
        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :, :], self.bias[0:end_h_y], 1,
                         self.padding, self.dilation, self.groups)
        X_l2h = F.upsample(X_l2h, scale_factor=2, mode='nearest')  # , **self.up_kwargs)

        ### 低频处理 l2l, h2l
        # l2l: conv3x3
        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)
        # h2l: avgpool+conv
        X_h2l = self.h2g_pool(X_h)
        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        ### 高频+
        X_h = X_h2h + X_l2h
        X_h = self.conv3x3_1( X_h)
        X_h = self.sa( X_h)
        ### 低频+
        X_l = X_l2l + X_h2l


        ### 合并高低频cat
        X_l = F.upsample(X_l, scale_factor=2, mode='nearest')
        X = torch.cat((X_h, X_l), dim=1)

        return X


class OctaveConv_v7(nn.Module):  # self.channel1, self.channel1
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = 'nearest', weights = None):
        super(OctaveConv_v7, self).__init__()
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

        self.Conv1x1_h = Conv1x1(512, int(in_channels/2))
        self.Conv1x1_x = Conv1x1(in_channels, int(in_channels/2))
        self.conv3x3_1 = ConvBNR(int(in_channels/2), int(in_channels/2), 3)
        self.sa = SA_Block_CrossPoolKV(int(self.in_channels/2))
    def forward(self, x, h):
        '''
        352->   88--44--22--11(官方)
        448->  112--56--28--14(当前)
        通道->   64-128-256-512
        输入:
            f_feature -> x (假定x是第2层特征(B,128,56,56)) C=128
            h -> h (B,512,14,14)
        '''
        ### 1x1卷积统一通道为C/2
        X_l = self.Conv1x1_h(h)  # Chn:512->C/2 | HW:14,14
        X_h = self.Conv1x1_x(x)  # Chn:C->C/2   | HW:56,56

        ### 获得当前层特征size的2分之1,作为低频尺度
        size=X_h.size()[2:]  # BCHW->[HW]
        H_size = int(size[0]/2)  # W/2=28
        W_size = int(size[1]/2)  # W/2=28

        ### 低频特征up,尺度变为28,28
        X_l = F.interpolate(X_l, (H_size,H_size), mode='bilinear')  # BCHW ->[HW] 28,28

        ### X_l X_h 准备就绪
        ### 通道对半分
        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        ### 高频处理 h2h, l2h
        # h2h: conv3x3
        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)
        # l2h: conv3x3+up
        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :, :], self.bias[0:end_h_y], 1,
                         self.padding, self.dilation, self.groups)
        X_l2h = F.upsample(X_l2h, scale_factor=2, mode='nearest')  # , **self.up_kwargs)

        ### 低频处理 l2l, h2l
        # l2l: conv3x3
        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)
        # h2l: avgpool+conv
        X_h2l = self.h2g_pool(X_h)
        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        ### 高频+
        X_h = X_h2h + X_l2h
        X_h = self.conv3x3_1( X_h)
        X_h = self.sa( X_h)
        ### 低频+
        X_l = X_l2l + X_h2l


        ### 合并高低频cat
        X_l = F.upsample(X_l, scale_factor=2, mode='nearest')
        X = torch.cat((X_h, X_l), dim=1)

        return X


class OctaveConv_v8(nn.Module):  # self.channel1, self.channel1
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = 'nearest', weights = None):
        super(OctaveConv_v8, self).__init__()
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

        self.Conv1x1_h = Conv1x1(512, int(in_channels/2))
        self.Conv1x1_x = Conv1x1(in_channels, int(in_channels/2))
        self.conv3x3_1 = ConvBNR(int(in_channels/2), int(in_channels/2), 3)
        self.sa = SA_Block_Ancher(int(self.in_channels/2))
    def forward(self, x, h):
        '''
        352->   88--44--22--11(官方)
        448->  112--56--28--14(当前)
        通道->   64-128-256-512
        输入:
            f_feature -> x (假定x是第2层特征(B,128,56,56)) C=128
            h -> h (B,512,14,14)
        '''
        ### 1x1卷积统一通道为C/2
        X_l = self.Conv1x1_h(h)  # Chn:512->C/2 | HW:14,14
        X_h = self.Conv1x1_x(x)  # Chn:C->C/2   | HW:56,56

        ### 获得当前层特征size的2分之1,作为低频尺度
        size=X_h.size()[2:]  # BCHW->[HW]
        H_size = int(size[0]/2)  # W/2=28
        W_size = int(size[1]/2)  # W/2=28

        ### 低频特征up,尺度变为28,28
        X_l = F.interpolate(X_l, (H_size,H_size), mode='bilinear')  # BCHW ->[HW] 28,28

        ### X_l X_h 准备就绪
        ### 通道对半分
        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        ### 高频处理 h2h, l2h
        # h2h: conv3x3
        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)
        # l2h: conv3x3+up
        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :, :], self.bias[0:end_h_y], 1,
                         self.padding, self.dilation, self.groups)
        X_l2h = F.upsample(X_l2h, scale_factor=2, mode='nearest')  # , **self.up_kwargs)

        ### 低频处理 l2l, h2l
        # l2l: conv3x3
        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)
        # h2l: avgpool+conv
        X_h2l = self.h2g_pool(X_h)
        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        ### 高频+
        X_h = X_h2h + X_l2h
        X_h = self.conv3x3_1( X_h)
        X_h = self.sa( X_h)
        ### 低频+
        X_l = X_l2l + X_h2l


        ### 合并高低频cat
        X_l = F.upsample(X_l, scale_factor=2, mode='nearest')
        X = torch.cat((X_h, X_l), dim=1)

        return X

class OctaveConv_v9(nn.Module):  # self.channel1, self.channel1
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = 'nearest', weights = None):
        super(OctaveConv_v9, self).__init__()
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

        self.Conv1x1_h = Conv1x1(512, int(in_channels/2))
        self.Conv1x1_x = Conv1x1(in_channels, int(in_channels/2))
        self.conv3x3_1 = ConvBNR(int(in_channels/2), int(in_channels/2), 3)
        self.sa = SA_Block_CrossPoolKV_res(int(self.in_channels/2))
    def forward(self, x, h):
        '''
        352->   88--44--22--11(官方)
        448->  112--56--28--14(当前)
        通道->   64-128-256-512
        输入:
            f_feature -> x (假定x是第2层特征(B,128,56,56)) C=128
            h -> h (B,512,14,14)
        '''
        ### 1x1卷积统一通道为C/2
        X_l = self.Conv1x1_h(h)  # Chn:512->C/2 | HW:14,14
        X_h = self.Conv1x1_x(x)  # Chn:C->C/2   | HW:56,56

        ### 获得当前层特征size的2分之1,作为低频尺度
        size=X_h.size()[2:]  # BCHW->[HW]
        H_size = int(size[0]/2)  # W/2=28
        W_size = int(size[1]/2)  # W/2=28

        ### 低频特征up,尺度变为28,28
        X_l = F.interpolate(X_l, (H_size,H_size), mode='bilinear')  # BCHW ->[HW] 28,28

        ### X_l X_h 准备就绪
        ### 通道对半分
        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        ### 高频处理 h2h, l2h
        # h2h: conv3x3
        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)
        # l2h: conv3x3+up
        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :, :], self.bias[0:end_h_y], 1,
                         self.padding, self.dilation, self.groups)
        X_l2h = F.upsample(X_l2h, scale_factor=2, mode='nearest')  # , **self.up_kwargs)

        ### 低频处理 l2l, h2l
        # l2l: conv3x3
        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)
        # h2l: avgpool+conv
        X_h2l = self.h2g_pool(X_h)
        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        ### 高频+
        X_h = X_h2h + X_l2h
        X_h = self.conv3x3_1( X_h)
        X_h = self.sa( X_h)
        ### 低频+
        X_l = X_l2l + X_h2l


        ### 合并高低频cat
        X_l = F.upsample(X_l, scale_factor=2, mode='nearest')
        X = torch.cat((X_h, X_l), dim=1)

        return X


class OctaveConv_v10(nn.Module):  # self.channel1, self.channel1
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=0.5, alpha_out=0.5, stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs = 'nearest', weights = None):
        super(OctaveConv_v10, self).__init__()
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

        self.Conv1x1_h = Conv1x1(512, int(in_channels/2))
        self.Conv1x1_x = Conv1x1(in_channels, int(in_channels/2))
        self.conv3x3_1 = ConvBNR(int(in_channels/2), int(in_channels/2), 3)
        self.sa = SA_Block_CrossPoolKV_res3x3(int(self.in_channels/2))
    def forward(self, x, h):
        '''
        352->   88--44--22--11(官方)
        448->  112--56--28--14(当前)
        通道->   64-128-256-512
        输入:
            f_feature -> x (假定x是第2层特征(B,128,56,56)) C=128
            h -> h (B,512,14,14)
        '''
        ### 1x1卷积统一通道为C/2
        X_l = self.Conv1x1_h(h)  # Chn:512->C/2 | HW:14,14
        X_h = self.Conv1x1_x(x)  # Chn:C->C/2   | HW:56,56

        ### 获得当前层特征size的2分之1,作为低频尺度
        size=X_h.size()[2:]  # BCHW->[HW]
        H_size = int(size[0]/2)  # W/2=28
        W_size = int(size[1]/2)  # W/2=28

        ### 低频特征up,尺度变为28,28
        X_l = F.interpolate(X_l, (H_size,H_size), mode='bilinear')  # BCHW ->[HW] 28,28

        ### X_l X_h 准备就绪
        ### 通道对半分
        end_h_x = int(self.in_channels*(1- self.alpha_in))
        end_h_y = int(self.out_channels*(1- self.alpha_out))

        ### 高频处理 h2h, l2h
        # h2h: conv3x3
        X_h2h = F.conv2d(X_h, self.weights[0:end_h_y, 0:end_h_x, :,:], self.bias[0:end_h_y], 1,
                        self.padding, self.dilation, self.groups)
        # l2h: conv3x3+up
        X_l2h = F.conv2d(X_l, self.weights[0:end_h_y, end_h_x:, :, :], self.bias[0:end_h_y], 1,
                         self.padding, self.dilation, self.groups)
        X_l2h = F.upsample(X_l2h, scale_factor=2, mode='nearest')  # , **self.up_kwargs)

        ### 低频处理 l2l, h2l
        # l2l: conv3x3
        X_l2l = F.conv2d(X_l, self.weights[end_h_y:, end_h_x:, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)
        # h2l: avgpool+conv
        X_h2l = self.h2g_pool(X_h)
        X_h2l = F.conv2d(X_h2l, self.weights[end_h_y:, 0: end_h_x, :,:], self.bias[end_h_y:], 1,
                        self.padding, self.dilation, self.groups)

        ### 高频+
        X_h = X_h2h + X_l2h
        X_h = self.conv3x3_1( X_h)
        X_h = self.sa( X_h)
        ### 低频+
        X_l = X_l2l + X_h2l


        ### 合并高低频cat
        X_l = F.upsample(X_l, scale_factor=2, mode='nearest')
        X = torch.cat((X_h, X_l), dim=1)

        return X
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


