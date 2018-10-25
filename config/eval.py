import os
import sys
import cv2
import argparse
import numpy as np

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import settings
from dataset import TestDataset
from model import RESCAN 
from cal_ssim import SSIM

logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        

class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        self.net = RESCAN().cuda()
        self.crit = MSELoss().cuda()
        self.ssim = SSIM().cuda()
        self.dataloaders = {}

    def get_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=1, 
                            shuffle=False, num_workers=1, drop_last=False)
        return self.dataloaders[dataset_name]

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])

    def inf_batch(self, name, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        R = O - B

        with torch.no_grad():
            O_Rs = self.net(O)
        loss_list = [self.crit(O_R, R) for O_R in O_Rs]
        ssim_list = [self.ssim(O - O_R, O - R) for O_R in O_Rs]

        losses = {
            'loss%d' % i: loss.item()
            for i, loss in enumerate(loss_list)
        }
        ssimes = {
            'ssim%d' % i: ssim.item()
            for i, ssim in enumerate(ssim_list)
        }
        losses.update(ssimes)

        return losses


def run_test(ckp_name):
    sess = Session()
    sess.net.eval()
    sess.load_checkpoints(ckp_name)
    dt = sess.get_dataloader('test')

    all_num = 0
    all_losses = {}
    for i, batch in enumerate(dt):
        losses = sess.inf_batch('test', batch)
        batch_size = batch['O'].size(0)
        all_num += batch_size
        for key, val in losses.items():
            if i == 0:
                all_losses[key] = 0.
            all_losses[key] += val * batch_size
            logger.info('batch %d mse %s: %f' % (i, key, val))

    for key, val in all_losses.items():
        logger.info('total mse %s: %f' % (key, val / all_num))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='latest')

    args = parser.parse_args(sys.argv[1:])
    run_test(args.model)

