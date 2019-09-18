import torch
import torch.utils.data as data_utl
from torch.utils.data.dataloader import default_collate

import numpy as np
import json
import csv
import h5py

from tqdm import tqdm
import glob
import os
import os.path

import cv2

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames(vid, start, num):
    frames = [
        cv2.resize(
            cv2.imread(os.path.join(vid, str(i).zfill(6)+'.jpg')),
            (455,256)
        )[:,:,[2,1,0]]/127.5 - 1 for i in range(start, start+num)
    ]
    return np.asarray(frames, dtype=np.float32)


def load_flow_frames(vid, start, num):
    frames = [
        np.asarray([
            cv2.resize(cv2.imread(os.path.join(vid,'x_'+str(i).zfill(5)+'.jpg'), cv2.IMREAD_GRAYSCALE), (455,256))/127.5-1,
            cv2.resize(cv2.imread(os.path.join(vid,'y_'+str(i).zfill(5)+'.jpg'), cv2.IMREAD_GRAYSCALE), (455,256))/127.5-1
        ]).transpose([1,2,0]) for i in range(start, start+num)
    ]
    return np.asarray(frames, dtype=np.float32)


def make_dataset(root, mode):
    dataset = []
    mode = 'image' if mode=='rgb' else 'flow'
    data = list(filter(
        lambda f: os.path.isdir(f) and f.split('/')[-1] in [
"0019_Pulp_Fiction_00.48.08.271-00.49.52.747",
"0008_Fargo_00.16.06.297-00.17.26.000",
"3012_BRUNO_00.34.48.536-00.37.16.363",
"0025_THE_LORD_OF_THE_RINGS_THE_RETURN_OF_THE_KING_03.07.32.821-03.08.48.393",
"1012_Unbreakable_01.21.21.721-01.21.22.815",
"1012_Unbreakable_00.55.56.510-00.55.59.634",
"1050_Harry_Potter_and_the_deathly_hallows_Disk_One_01.50.41.516-01.53.37.673"
],
        glob.glob(os.path.join(root,'*','*'))
    ))

    for vid in tqdm(data, desc='load path'):
        num_frames = len(glob.glob(os.path.join(vid,'*')))
        if mode=='flow':
            # Flow directory contains both x-dim and y-dim files,
            # thereby dividing directory_length by 2 is necessary
            num_frames = num_frames // 2
        else:
            # Aligning rgb with flow
            num_frames = num_frames - 1
        dataset.append((vid, num_frames))
    return dataset


class LSMDC(data_utl.Dataset):
    def __init__(self, root, mode, transforms=None, save_dir='', num=0):
        self.data = make_dataset(root, mode)
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        vid, nf = self.data[index]
        #mov = '_'.join(vid.split('_')[:-1])
        vname = vid.split('/')[-1]
        #back_dir = '/data1/yj/optflow/npy'
        if os.path.exists(os.path.join(self.save_dir, vid+'.npy')):
            return 0, 0, vid
        #elif os.path.exists(os.path.join(back_dir,self.mode,mov,vname+'.npy')):
        #    #Should be configured
        #    return np.load(os.path.join(back_dir,self.mode,mov,vname+'.npy')), vname

        if self.mode == 'rgb':
            imgs = load_rgb_frames(vid, 2, nf)
        else:
            imgs = load_flow_frames(vid, 1, nf)

        imgs = self.transforms(imgs)

        # Should be configured
        #if not os.path.exists(os.path.join(back_dir,self.mode,mov)):
        #    os.mkdir(os.path.join(back_dir,self.mode,mov))
        #np.save(os.path.join(back_dir,self.mode,mov,vname), imgs)

        return video_to_tensor(imgs), vname

    def __len__(self):
        return len(self.data)
