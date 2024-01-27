import os
from option import args
os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_name
import torch
import torch.optim as optim
import torch.nn as nn
from data import dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.metric import psnr_ssim,lpips_niqe
import utils.util as util
import torchvision
import models


print('test phase lr')

testdata1 = dataset.Data(root=os.path.join(args.data_path,), args=args, train=False)

testset1 = DataLoader(testdata1, batch_size=1, shuffle=False, num_workers=1)

device = torch.device(args.device)
def to_device(sample, device):
    for key, value in sample.items():
        if key != 'img_name':
            sample[key] = value.to(device, non_blocking=True)
    return sample

def eval_model(model, dataset, name, epoch, args):

    model.eval()
    val_psnr_dic = 0
    val_ssim_dic = 0
    val_lpips_dic = 0
    val_niqe_dic = 0
    os.makedirs(os.path.join(args.save_path, args.writer_name, 'result-{}'.format(name)), exist_ok=True)
    timer_test = util.timer()
    for batch, data in enumerate(dataset):

        sr = model(to_device(data, device))

        psnr_c, ssim_c = psnr_ssim(data['img_gt'], sr['img_out'])
        val_psnr_dic = val_psnr_dic + psnr_c
        val_ssim_dic = val_ssim_dic + ssim_c

        torchvision.utils.save_image(sr['img_out'][0],
                                     os.path.join(args.save_path, args.writer_name, 'result-{}'.format(name),
                                                  '{}'.format(str(data['img_name'][0]))))


    print("Epoch：{}, {}, psnr: {:.3f}".format(epoch+1, name, val_psnr_dic/(len(dataset))))#out（局部空间）分支和fre分支
    print("Epoch：{}, {}, ssim: {:.3f}".format(epoch + 1, name, val_ssim_dic / (len(dataset))))

    print('Forward: {:.2f}s\n'.format(timer_test.toc()))


if __name__ == "__main__":
    model = models.get_model(args)
    pretrained_dict = torch.load(
        args.load )
    model.load_state_dict(pretrained_dict)
    print(len(testset1))
    eval_model(model, testset1, args.save_name, 0, args)












