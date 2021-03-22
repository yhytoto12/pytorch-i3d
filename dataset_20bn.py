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

# imread --> (H, W, C) tensor C (B, G, R)
# [:, :, [2, 1, 0]] --convet--> (R, G, B)
def load_rgb_frames(vid, start, num):
    frames = [
        cv2.imread(os.path.join(vid, str(i).zfill(5)+'.jpg'))[:,:,[2,1,0]]/127.5 - 1 for i in range(start, start + num)
    ]
    return np.asarray(frames, dtype=np.float32)

# transpose([1,2,0]) means (2, H, W) --convert--> (H, W, 2)
def load_flow_frames(vid, start, num):
    frames = [
        np.asarray([
            cv2.imread(os.path.join(vid, 'x_'+str(i).zfill(5)+'.jpg'), cv2.IMREAD_GRAYSCALE)/127.5 - 1,
            cv2.imread(os.path.join(vid, 'y_'+str(i).zfill(5)+'.jpg'), cv2.IMREAD_GRAYSCALE)/127.5 - 1
        ]).transpose([1,2,0]) for i in range(start, start + num)
    ]
    return np.asarray(frames, dtype=np.float32)

def make_dataset(root, mode):
    dataset = []
    data = list(filter(
        lambda f: os.path.isdir(f),
        glob.glob(os.path.join(root, "*"))
    ))

    for vid in tqdm(data, desc="load path"):
        num_frames = len(glob.glob(os.path.join(vid, "*")))
        if mode == "flow":
            # Flow directory contains both x-dim and y-dim files
            # thereby dividing directory_lenght by 2 is neccesary
            num_frames = num_frames // 2
        else:
            # Aligning rgb with flow
            num_frames = num_frames - 1
        dataset.append((vid, num_frames))

    return dataset

class TwentyBN(data_utl.Dataset):
    def __init__(self, root, mode, transforms=None, save_dir='', num=0):
        self.data = make_dataset(root, mode)
        self.transforms = transforms
        self.mode = mode
        self.root = root
        self.save_dir = save_dir
        print("[Log] number of data : {}".format(len(self.data)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where ``target`` is class_index of the target class.
        """
        vid, nf = self.data[index] # get index-th video and the number of frames
        vid_name = vid.split('/')[-1]

        # if the video is already processed and saved correctly, then ignore it!
        if os.path.exists(os.path.join(self.save_dir, vid + '.npy')):
            return 0, 0, vid

        if self.mode == 'rgb':
            imgs = load_rgb_frames(vid, 2, nf)
        else:
            imgs = load_flow_frames(vid, 1, nf)

        imgs = self.transforms(imgs)


        return video_to_tensor(imgs), vid_name

    def __len__(self):
        return len(self.data)




