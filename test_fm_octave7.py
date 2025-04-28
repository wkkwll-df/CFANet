import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # todo
from net.EMDNet_fm_octave7 import Net
from utils.tdataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=448, help='testing size')  # 352 原版416  # todo
sname = 'EMDNet_fm_octave7'  # todo
parser.add_argument('--pth_path', type=str, default='./checkpoints/'+sname+'/BGNet-24.pth')  # todo

for _data_name in ['COD10K-TE']:  # 'CAMO','CHAMELEON', ,'NC4K'
    data_path = '/cluster/home3/zjc/Dataset/COD/COD-TE/{}/'.format(_data_name)
    save_path = './results/'+sname+'/{}/'.format(_data_name)
    opt = parser.parse_args()
    model = Net()

    print('=>load model', opt.pth_path)
    print('=>save_path', save_path)
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()

    os.makedirs(save_path, exist_ok=True)
    # os.makedirs(save_path+'edge/', exist_ok=True)
    image_root = '{}image/'.format(data_path)
    gt_root = '{}mask/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)

    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        res4, res3, res2, res1 = model(image)
        res1 = F.upsample(res1, size=gt.shape, mode='bilinear', align_corners=False)
        res1 = res1.sigmoid().data.cpu().numpy().squeeze()
        res1 = (res1 - res1.min()) / (res1.max() - res1.min() + 1e-8)
        imageio.imwrite(save_path+name, (res1*255).astype(np.uint8))
        # e = F.upsample(e, size=gt.shape, mode='bilinear', align_corners=True)
        # e = e.data.cpu().numpy().squeeze()
        # e = (e - e.min()) / (e.max() - e.min() + 1e-8)
        # imageio.imwrite(save_path+'edge/'+name, (e*255).astype(np.uint8))
