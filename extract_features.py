import os
import sys
import argparse

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable

import torchvision
from torchvision import datasets, transforms
import videotransforms

from tqdm import tqdm
import numpy as np

from pytorch_i3d import InceptionI3d

from dataset_lsmdc import LSMDC as Dataset
from config import config


def run(cfg):
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = Dataset(root=cfg['data_dir'], mode=cfg['mode'], transforms=test_transforms, num=-1, save_dir=cfg['save_dir'])
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=16, pin_memory=True)

    # setup the model
    if cfg['mode'] == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:
        i3d = InceptionI3d(400, in_channels=3)
    # i3d.replace_logits(157)
    i3d.load_state_dict(torch.load(cfg['load_model']))
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    tot_loss = 0.0
    tot_loc_loss = 0.0
    tot_cls_loss = 0.0

    map_dir = cfg['save_dir']+'_map'

    if not os.path.exists(cfg['save_dir']):
        os.mkdir(cfg['save_dir'])
    if not os.path.exists(map_dir):
        os.mkdir(map_dir)

    # Iterate over data.
    for data in tqdm(dataloader):
        # get the inputs
        inputs, name = data
        mov = '_'.join(name[0].split('_')[:-1])

        if not os.path.exists(os.path.join(cfg['save_dir'], mov)):
            os.mkdir(os.path.join(cfg['save_dir'], mov))
        elif os.path.exists(os.path.join(cfg['save_dir'], mov, name[0]+'.npy')):
            continue

        if not os.path.exists(os.path.join(map_dir, mov)):
            os.mkdir(os.path.join(map_dir, mov))

        b,c,t,h,w = inputs.shape
        print('LOG: {} shape: {}'.format(name[0], inputs.shape))
        if t > 1600:
            features = []
            maps = []
            for start in range(1, t-56, 1600):
                end = min(t-1, start+1600+56)
                do_end_crop = True if end == start+1600+56 else False
                start = max(1, start-48)
                do_start_crop = True if start != 1 else False
                ip = Variable(torch.from_numpy(inputs.numpy()[:,:,start:end]).cuda(), volatile=True)
                map_pool, avg_pool = i3d.extract_features(ip)
                map_pool = map_pool.squeeze(0).permute(1,2,3,0).data.cpu().numpy()
                avg_pool = avg_pool.squeeze(0).squeeze(-1).squeeze(-1).permute(-1,0).data.cpu().numpy()
                if do_end_crop:
                    print('LOG: do end crop')
                    map_pool = map_pool[:-6,:,:,:]
                    avg_pool = avg_pool[:-6,:]
                if do_start_crop:
                    print('LOG: do start crop')
                    map_pool = map_pool[6:,:,:,:]
                    avg_pool = avg_pool[6:,:]
                maps.append(map_pool)
                features.append(avg_pool)
                print('LOG: maps: {}, features: {}'.format(map_pool.shape, avg_pool.shape))
            np.save(os.path.join(cfg['save_dir'], mov, name[0]), np.concatenate(features, axis=0))
            np.save(os.path.join(map_dir, mov, name[0]), np.concatenate(maps, axis=0))
        else:
            inputs = Variable(inputs.cuda(), volatile=True)
            map_pool, avg_pool = i3d.extract_features(inputs)
            print('LOG: maps: {}, features: {}'.format(map_pool.shape, avg_pool.shape))
            np.save(
                os.path.join(cfg['save_dir'], mov, name[0]),
                avg_pool.squeeze(0).squeeze(-1).squeeze(-1).permute(-1,0).data.cpu().numpy()
            )
            np.save(
                os.path.join(map_dir, mov, name[0]),
                map_pool.squeeze(0).permute(1,2,3,0).data.cpu().numpy()
            )


if __name__ == '__main__':
    run(config)
