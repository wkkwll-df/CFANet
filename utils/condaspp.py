import torch
import torch.nn as nn
import torch.nn.functional as F


# 输入为 [N, C, H, W]，需要两个参数，Cin为输入特征通道数，K 为专家个数
class Attention(nn.Module):
    def __init__(self,Cin,K):
        super().__init__()
        self.avgpool=nn.AdaptiveAvgPool2d(1)  # 池化操作
        self.net=nn.Conv2d(Cin, K, kernel_size=1)  # Cin通道变为K
        self.sigmoid=nn.Sigmoid()  # 归一化
 
    def forward(self,x):
        # 将输入特征全局池化为 [N,Cin,H,W]->[N, Cin, 1, 1]
        att=self.avgpool(x)
        # 使用1*1卷积，转化为 [N, Cin, 1, 1]->[N, K, 1, 1]
        att=self.net(att)
        # 将特征转化为二维 [N, K, 1, 1]->[N, K]
        att=att.view(x.shape[0], -1)
        # 使用 sigmoid 函数输出归一化到 [0,1] 区间
        return self.sigmoid(att)
    

# ## choice1: cond conv
class CondConv(nn.Module):
    def __init__(self, Cin=256, Cout=256, kernel_size=3, stride=1, padding=1,
                 groups=1, K=4):
        super().__init__()
        self.Cin = Cin  # 输入通道
        self.Cout = Cout  # 输出通道
        self.K = K  # K个权重
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention = Attention(Cin=Cin,K=K)
        # weight [K, Cout, Cin, kernelz_size, kernel_size]
        self.weight = nn.Parameter(torch.randn(K,Cout,Cin//groups,
                                             kernel_size,kernel_size),requires_grad=True)

    def forward(self, x):

        ### part1 weight
        N,Cin, H, W = x.shape  # x.shape [5,256,7,7]
        softmax_att = self.attention(x)
        # 调用 attention 函数得到归一化的权重 [N,Cin,H,W]->[N, K]
        # x = [1,1280,7,7]
        # softmasx_att.shape [5,4]

        ### part2 x
        x = x.contiguous().view(1, -1, H, W)  # [N, Cin, H, W]->[1, N*Cin, H, W]
        # x.shape s[1,5*256,7,7]

        ### part3 conv
        weight = self.weight
        # 生成随机weight[K, Cout, C_in/groups, 3, 3] (卷积核一般为3*3)
        # 注意添加了 requires_grad=True，这样里面的参数是可以优化的
        weight = weight.view(self.K, -1)
        # 改变 weight 形状为 [K,Cout,Cin,3,3]->[K, C_out*(C_in/groups)*3*3]
        # [4, 256, 256, 3, 3] -> [4, 589824]
        # weight.shape [4, 589824]

        ### part4: 新的wconv = weight*conv
        aggregate_weight = torch.mm(softmax_att, weight)
        # 矩阵相乘：[N, K]*[K, Cout*(Cin/groups)*3*3] = [N, Cout*(Cin/groups)*3*3]
        # [5,4]*[4,589xxx]  = [5, 589xxx]
        # 改变形状为：[N, Cout*Cin/groups*3*3]->[N*Cout, Cin/groups, 3, 3]，即新的卷积核权重
        aggregate_weight = aggregate_weight.view(
            N*self.Cout, self.Cin//self.groups,
            self.kernel_size, self.kernel_size)
        # aggregate_weight.shape [5, 589xxx] -> [5*256, 256, 3, 3]

        output=F.conv2d(x, weight=aggregate_weight,
                        stride=self.stride, padding=self.padding,
                        groups=self.groups*N)
        # 用新生成的卷积核进行卷积 x[1, N*Cin, H, W] w[N*Cout, Cin/groups, 3, 3]
        # x x[1,1280,7,7] w[1280,256,3,3]
        # 输出为 [1, N*Cout, H, W]
        # [1,1280,7,7]

        output=output.view(N, self.Cout, H, W)
        # N=5 Cin=256 H=7 W=7
        # 形状恢复为 [N, Cout, H, W]
        # [5, 256, 7, 7]
        return output


# ## choice2: dilation cond conv
class DCondConv(nn.Module):
    def __init__(self, Cin, Cout, d, kernel_size=3, stride=1, padding=1,
                 groups=1, K=4):
        super().__init__()
        self.Cin = Cin  # 输入通道
        self.Cout = Cout  # 输出通道
        self.K = K  # K个权重
        self.groups = groups
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.attention = Attention(Cin=Cin,K=K)
        # weight [K, Cout, Cin, kernelz_size, kernel_size]
        self.weight = nn.Parameter(torch.randn(K,Cout,Cin//groups,
                                             kernel_size,kernel_size),requires_grad=True)
        self.dilation = d
        self.padding = d

    def forward(self, x):
        ### part1 weight
        N, Cin, H, W = x.shape  # x.shape [5,64,14,14]
        softmax_att = self.attention(x)
        # 调用 attention 函数得到归一化的权重 [N,Cin,H,W]->[N, K]
        # x = [5,64,14,14]
        # softmasx_att.shape [5,4]

        ### part2 x
        x = x.contiguous().view(1, -1, H, W)  # [N, Cin, H, W]->[1, N*Cin, H, W]
        # x.shape s[1,5*64,14,14]

        ### part3 conv
        weight = self.weight
        # 生成随机weight[K, Cout, C_in/groups, 3, 3] (卷积核一般为3*3)
        # 注意添加了 requires_grad=True，这样里面的参数是可以优化的
        # weight[4,64,64,3,3]
        weight = weight.view(self.K, -1)
        # 改变 weight 形状为 [K,Cout,Cin,3,3]->[K, C_out*(C_in/groups)*3*3]
        # [4, 256, 256, 3, 3] -> [4, 589824]
        # weight.shape [4, 589824]

        ### part4: 新的wconv = weight*conv
        aggregate_weight = torch.mm(softmax_att, weight)
        # 矩阵相乘：[N, K]*[K, Cout*(Cin/groups)*3*3] = [N, Cout*(Cin/groups)*3*3]
        # [5,4]*[4,589xxx]  = [5, 589xxx]
        # 改变形状为：[N, Cout*Cin/groups*3*3]->[N*Cout, Cin/groups, 3, 3]，即新的卷积核权重
        aggregate_weight = aggregate_weight.view(
            N*self.Cout, self.Cin//self.groups,
            self.kernel_size, self.kernel_size)
        # aggregate_weight.shape [5, 365864] -> [5*64, 64, 3, 3]

        output=F.conv2d(x, weight=aggregate_weight,
                        stride=self.stride, padding=self.padding, dilation=self.dilation,
                        groups=self.groups*N)
        # 用新生成的卷积核进行卷积 x[1, N*Cin, H, W] w[N*Cout, Cin/groups, 3, 3]
        # x x[1,320,14,14] w[320,64,3,3]
        # 输出为 [1, N*Cout, H, W]
        # [1,5*64,14,14]

        output=output.view(N, self.Cout, H, W)
        # N=5 Cin=256 H=7 W=7
        # 形状恢复为 [N, Cout, H, W]
        # [5, 256, 7, 7]
        return output


# 4 branch dilated cond conv
class CondASPP(nn.Module):
    def __init__(self, Cin, Cout): # [64 128 256]
        super(CondASPP, self).__init__()

        # self.ai_layer = nn.Linear(256, )

        # self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.branch_3 = DCondConv(Cin, Cout//4, d=3)
        self.branch_6 = DCondConv(Cin, Cout//4, d=6)
        self.branch_12 = DCondConv(Cin, Cout//4, d=12)
        self.branch_18 = DCondConv(Cin, Cout//4, d=18)

        self.CBR_1x1 = nn.Sequential(nn.Conv2d(Cin, Cin, (1, 1)), nn.BatchNorm2d(Cin), nn.ReLU())

    def forward(self, x):  # x[bchw]

        # y = self.gap(x).squeeze(-1).squeeze(-1)  # x[bf 256 h w] y[bf 256]
        # 256 = 4*64
        # x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        x1, x2, x3, x4 = x, x, x, x
        # x [5,256,14,14]
        # x1 [5,64,14,14]
        out3 = self.branch_3(x1)   # c -> c//4
        out6 = self.branch_6(x2)  # c -> c//4
        out12 = self.branch_12(x3)  # c -> c//4
        out18 = self.branch_18(x4)  # c -> c//4
        out = torch.cat((out3, out6, out12, out18), dim=1)  # 4*(c//4)

        out = self.CBR_1x1(out)

        return out
        

if __name__ == "__main__":
    imgs = torch.randn(2, 3, 224, 224)
    y = CondConv(imgs)
    print(y.shape)
