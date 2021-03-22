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

from dataset_20bn import TwentyBN as Dataset

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, help='rgb or flow', default=cfg['mode'])
parser.add_argument('--load_model', type=str, default=cfg['load_model'])
parser.add_argument('--data_dir', type=str, default=cfg['data_dir'])
parser.add_argument('--save_dir', type=str, default=cfg['save_dir'])
parser.add_argument('--start_index', type=int)
parser.add_argument('--end_index', type=int)

args = parser.parse_args()
cfg = {
    'mode' : args.mode,
    'load_model' : args.load_model,
    'data_dir' : args.data_dir,
    'save_dir' : args.save_dir,
    'start_index' : args.start_index,
    'end_index' : args.end_index,
}

def run():
    # setup dataset
    test_transforms = transforms.Compose([videotransforms.CenterCrop(224)])
    dataset = Dataset(
        data_dir=cfg['data_dir'],
        mode=cfg['mode'],
        transforms=test_transforms,
        save_dir=cfg['save_dir'],
        start_index=args.start_index,
        end_index=args.end_index)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    # setup the model
    if args.mode == 'flow':
        i3d = InceptionI3d(400, in_channels=2)
    else:   # rgb
        i3d = InceptionI3d(400, in_channels=3)

    i3d.load_state_dict(torch.load(args.load_model))
    i3d.cuda()
    i3d.train(False)  # Set model to evaluate mode

    save_dir = args.save_dir
    map_dir = save_dir + '_map'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not os.path.exists(map_dir):
        os.mkdir(map_dir)

    # Iterate over data.
    for data in tqdm(dataloader):
        # get the inputs and video name
        inputs, vid_name = data

        if not os.path.exists(os.path.join(save_dir, vid_name)):
            os.mkdir(os.path.join(save_dir, vid_name))
        elif os.path.exists(os.path.join(save_dir, vid_name + '.npy')):
            continue

        if not os.path.exists(os.path.join(map_dir, vid_name)):
            os.mkdir(os.path.join(map_dir, vid_name))

        b,c,t,h,w = inputs.shape

        inputs = Variable(inputs.cuda(), volatile=True)
        map_pool, avg_pool = i3d.extract_features(inputs)
        np.save(
            os.path.join(save_dir, vid_name),
            avg_pool.squeeze(0).squeeze(-1).squeeze(-1).permute(-1,0).data.cpu().numpy()
        )
        np.save(
            os.path.join(map_dir, vid_name),
            map_pool.squeeze(0).permute(1,2,3,0).data.cpu().numpy()
        )


if __name__ == '__main__':
    run()
