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

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3,0,1,2]))


def load_rgb_frames_LSMDC(vid, start, num):
    frames = [
        cv2.resize(
            cv2.imread(os.path.join(vid, str(i).zfill(6)+'.jpg')),
            (455,256)
        )[:,:,[2,1,0]]/127.5 - 1 for i in range(start, start+num)
    ]
    return np.asarray(frames, dtype=np.float32)

def load_rgb_frames(image_dir, vid, start, num):
  frames = []
  for i in tqdm(range(start, start+num), desc=vid):
    img = cv2.imread(os.path.join(vid, str(i).zfill(6)+'.jpg'))[:, :, [2, 1, 0]]
    w,h,c = img.shape
    if w < 226 or h < 226:
        d = 226.-min(w,h)
        sc = 1+d/min(w,h)
        img = cv2.resize(img,dsize=(0,0),fx=sc,fy=sc)
    img = (img/255.)*2 - 1
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def load_flow_frames_LSMDC(vid, start, num):
    frames = [
        np.asarray([
            cv2.resize(cv2.imread(os.path.join(vid,'x_'+str(i).zfill(5)+'.jpg'), cv2.IMREAD_GRAYSCALE), (455,256))/127.5-1,
            cv2.resize(cv2.imread(os.path.join(vid,'y_'+str(i).zfill(5)+'.jpg'), cv2.IMREAD_GRAYSCALE), (455,256))/127.5-1
        ]).transpose([1,2,0]) for i in range(start, start+num)
    ]
    return np.asarray(frames, dtype=np.float32)

def load_flow_frames(image_dir, vid, start, num):
  frames = []
  for i in tqdm(range(start, start+num), desc=vid):
    imgx = cv2.imread(os.path.join(vid, 'x_'+str(i).zfill(5)+'.jpg'), cv2.IMREAD_GRAYSCALE)
    imgy = cv2.imread(os.path.join(vid, 'y_'+str(i).zfill(5)+'.jpg'), cv2.IMREAD_GRAYSCALE)

    w,h = imgx.shape
    if w < 224 or h < 224:
        d = 224.-min(w,h)
        sc = 1+d/min(w,h)
        imgx = cv2.resize(imgx,dsize=(0,0),fx=sc,fy=sc)
        imgy = cv2.resize(imgy,dsize=(0,0),fx=sc,fy=sc)

    imgx = (imgx/255.)*2 - 1
    imgy = (imgy/255.)*2 - 1
    img = np.asarray([imgx, imgy]).transpose([1,2,0])
    frames.append(img)
  return np.asarray(frames, dtype=np.float32)

def make_dataset_LSMDC(root, mode):
    dataset = []
    mode = 'image' if mode=='rgb' else 'flow'
    data = list(filter(
        lambda f: os.path.isdir(f),
        glob.glob(os.path.join(root,'*','*'))
    ))

    for vid in tqdm(data, desc='load path'):
        num_frames = len(glob.glob(os.path.join(vid,'*')))-1
        if mode=='flow':
            # Flow directory contains both x-dim and y-dim files,
            # thereby dividing directory_length by 2 is necessary
            num_frames = num_frames // 2
        dataset.append((vid, num_frames))
    return dataset

def make_dataset(split_file, split, root, mode, num_classes=400):
    dataset = []
    with open(split_file, 'r') as f:
        data = json.load(f)

    i = 0
    for vid in data.keys():
        if data[vid]['subset'] != split:
            continue

        if not os.path.exists(os.path.join(root, vid)):
            continue
        num_frames = len(os.listdir(os.path.join(root, vid)))
        if mode == 'flow':
            num_frames = num_frames//2

        label = np.zeros((num_classes,num_frames), np.float32)

        fps = num_frames/data[vid]['duration']
        for ann in data[vid]['actions']:
            for fr in range(0,num_frames,1):
                if fr/fps > ann[1] and fr/fps < ann[2]:
                    label[ann[0], fr] = 1 # binary classification
        dataset.append((vid, label, data[vid]['duration'], num_frames))
        i += 1

    return dataset


class LSMDC(data_utl.Dataset):
    def __init__(self, root, mode, transforms=None, save_dir='', num=0):
        self.data = make_dataset_LSMDC(root, mode)
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
            #imgs = load_rgb_frames(self.root, vid, 2, nf)
            imgs = load_rgb_frames_LSMDC(vid, 2, nf)
        else:
            imgs = load_flow_frames_LSMDC(vid, 1, nf)

        imgs = self.transforms(imgs)

        # Should be configured
        #if not os.path.exists(os.path.join(back_dir,self.mode,mov)):
        #    os.mkdir(os.path.join(back_dir,self.mode,mov))
        #np.save(os.path.join(back_dir,self.mode,mov,vname), imgs)

        return video_to_tensor(imgs), vname

    def __len__(self):
        return len(self.data)
