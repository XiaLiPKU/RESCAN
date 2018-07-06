import os
import sys
import cv2
import argparse
import numpy as np
import itertools

import torch
from torch import nn
from torch.nn import DataParallel
from torch.optim import Adam
from torch.autograd import Variable 
from torch.utils.data import DataLoader

import settings
from dataset import ShowDataset
from model import DetailNet

logger = settings.logger
torch.cuda.manual_seed_all(66)
torch.manual_seed(66)
torch.cuda.set_device(settings.device_id)


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
        

class Session:
    def __init__(self):
        self.show_dir = settings.show_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.show_dir)
        ensure_dir(settings.model_dir)
        logger.info('set show dir as %s' % settings.show_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        self.net = DetailNet().cuda()
        self.dataloaders = {}

    def get_dataloader(self, dataset_name, use_iter=True):
        dataset = ShowDataset(dataset_name)
        self.dataloaders[dataset_name] = \
                    DataLoader(dataset, batch_size=1, 
                            shuffle=False, num_workers=1)
        return self.dataloaders[dataset_name]

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
        except FileNotFoundError:
            return
        self.net.load_state_dict(obj['net'])

    def inf_batch(self, name, batch):
        O = batch['O'].cuda()
        O = Variable(O)

        O_Rs = self.net(O)
        O_Rs = [O - O_R for O_R in O_Rs]
        
        return O_Rs

    def save_image(self, No, imgs):
        for i, img in enumerate(imgs):
            img = (img.cpu().data * 255).numpy()
            img = np.clip(img, 0, 255)[0]
            img = np.transpose(img, (1, 2, 0))
            h, w, c = img.shape
            img = cv2.resize(img, (2 * w, 2 * h))

            img_file = os.path.join(self.show_dir, '%02d_%d.jpg' % (No, i))
            cv2.imwrite(img_file, img)


def run_show(ckp_name):
    sess = Session()
    sess.load_checkpoints(ckp_name)
    sess.net.eval()

    dt = sess.get_dataloader('test', use_iter=False)

    for i, batch in enumerate(dt):
        logger.info(i)
        imgs = sess.inf_batch('test', batch)
        sess.save_image(i, imgs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='latest')

    args = parser.parse_args(sys.argv[1:])
    
    run_show(args.model)

