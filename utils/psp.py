import torch
import torch.nn as nn
import torch.nn.functional as F

'''
# =====官方原版====
class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):  # 2048, 1024, (1,2,3,6)
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        # self.stages->先AAP后Conv->上采样到原尺度
        # [s1 s2 s3 s6 特征自身]
        bottle = self.bottleneck(torch.cat(priors, 1))  # BCHW在C维度拼接 Conv1x1
        return self.relu(bottle)  # ReLU
'''
class PSP_5(nn.Module):

    def __init__(self, features, out_features, sizes=(1,2,3,6)):  # 2048, 1024, (1,2,3,6)
        '''
        [features, out_features]输入 输出通道
        [256, 512]
        [128, 256]
        [64, 128]
        '''
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features,
                                    kernel_size=1)  # 1x1conv 这里+2因为有high和features
        self.relu = nn.ReLU()
        self.features = features

        self.conv_h = nn.Conv2d(512, features, kernel_size=1, bias=False)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):  # high是高级特征14x14

        c, h, w = feats.size(1), feats.size(2), feats.size(3)

       # high = self.conv_h(high)
       # high = self.relu(high)

        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        # self.stages->先AAP后Conv->上采样到原尺度
        # [s1 s2 s3 s6 特征自身 high]
        bottle = self.bottleneck(torch.cat(priors, 1))  # BCHW在C维度拼接 Conv1x1
        return self.relu(bottle)  # ReLU


'''
========================================================================
'''

class PSP_4(nn.Module):

    def __init__(self, features, out_features, sizes=(1,)):  # 2048, 1024, (1,2,3,6)
        '''
        [features, out_features]输入 输出通道
        [256, 512]
        [128, 256]
        [64, 128]
        '''
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (1 + 2), out_features,
                                    kernel_size=1)  # 1x1conv 这里+2因为有high和features
        self.relu = nn.ReLU()
        self.features = features

        self.conv_h = nn.Conv2d(512, features, kernel_size=1, bias=False)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats, high):  # high是高级特征14x14

        c, h, w = feats.size(1), feats.size(2), feats.size(3)

        high = self.conv_h(high)
        high = self.relu(high)

        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats] + [
            F.upsample(high, size=(h, w), mode='bilinear')]
        # self.stages->先AAP后Conv->上采样到原尺度
        # [s1 s2 s3 s6 特征自身 high]
        bottle = self.bottleneck(torch.cat(priors, 1))  # BCHW在C维度拼接 Conv1x1
        return self.relu(bottle)  # ReLU


'''
========================================================================
'''

class PSP_3(nn.Module):

    def __init__(self, features, out_features, sizes=(7,14)):  # 2048, 1024, (1,2,3,6)
        '''
        [features, out_features]输入 输出通道
        [256, 512]
        [128, 256]
        [64, 128]
        '''
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 2), out_features,
                                    kernel_size=1)  # 1x1conv 这里+2因为有high和features
        self.relu = nn.ReLU()
        self.features = features

        self.conv_h = nn.Conv2d(512, features, kernel_size=1, bias=False)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    '''
    ===注释说明===
    init 填入三个：
    features是输入特征feats的通道数
    out_features=1024
    sizes=(1, 3, 6)

    forward 输入2个:
    feats.shape ->  b c h w
    high.shape -> b c h w
    return 出1个:
    output -> features*(3+1+1) # (1,3,6)三个特征 外加feats和high两个特征

    '''

    def forward(self, feats, high):  # high是高级特征14x14

        c, h, w = feats.size(1), feats.size(2), feats.size(3)

        high = self.conv_h(high)
        high = self.relu(high)

        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats] + [
            F.upsample(high, size=(h, w), mode='bilinear')]
        # self.stages->先AAP后Conv->上采样到原尺度
        # [s1 s2 s3 s6 特征自身 high]
        bottle = self.bottleneck(torch.cat(priors, 1))  # BCHW在C维度拼接 Conv1x1
        return self.relu(bottle)  # ReLU


'''
=====我的版本-2输入=====
'''
class PSP_2(nn.Module):

    def __init__(self, features, out_features, sizes=(1, 3)):  # 2048, 1024, (1,2,3,6)
        '''
        [features, out_features]输入 输出通道
        [256, 512]
        [128, 256]
        [64, 128]
        '''
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 2), out_features, kernel_size=1)  # 1x1conv 这里+2因为有high和features
        self.relu = nn.ReLU()
        self.features = features
        
        self.conv_h = nn.Conv2d(512, features, kernel_size=1, bias=False)

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    '''
    ===注释说明===
    init 填入三个：
    features是输入特征feats的通道数
    out_features=1024
    sizes=(1, 3, 6)
    
    forward 输入2个:
    feats.shape ->  b c h w
    high.shape -> b c h w
    return 出1个:
    output -> features*(3+1+1) # (1,3,6)三个特征 外加feats和high两个特征
    
    '''
    def forward(self, feats, high):  # high是高级特征14x14
        
        c, h, w = feats.size(1), feats.size(2), feats.size(3)
        
        high = self.conv_h(high)
        high = self.relu(high)
        
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats] + [F.upsample(high, size=(h, w), mode='bilinear')]
        # self.stages->先AAP后Conv->上采样到原尺度
        # [s1 s2 s3 s6 特征自身 high]
        bottle = self.bottleneck(torch.cat(priors, 1))  # BCHW在C维度拼接 Conv1x1
        return self.relu(bottle)  # ReLU

'''
=====我的版本-1输入=====
'''
class PSP_1(nn.Module):
    def __init__(self, features, out_features, sizes=(1, 3, 6)):  # 2048, 1024, (1,2,3,6)
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        # self.stages->先AAP后Conv->上采样到原尺度
        # [s1 s2 s3 s6 特征自身]
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    '''
    输入1个
    '''
    
    def forward(self, feats):  # high是高级特征14x14
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        # self.stages->先AAP后Conv->上采样到原尺度
        # [s1 s2 s3 s6 特征自身]
        bottle = self.bottleneck(torch.cat(priors, 1))  # BCHW在C维度拼接 Conv1x1
        return self.relu(bottle)  # ReLU
