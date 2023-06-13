from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from torchvision import utils
import random

class Data(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, valid=False, transform=None):

        self.root_dir = root_dir
        self.files = []
        self.valid = valid

        newdir = root_dir + '/datasets/airplane_part_seg/02691156/expert_verified/points_label/'

        for file in os.listdir(newdir):
            o = {}
            o['category'] = newdir + file
            o['img_path'] = root_dir + '/datasets/airplane_part_seg/02691156/points/' + file.replace('.seg', '.pts')
            self.files.append(o)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]['img_path']
        category = self.files[idx]['category']
        with open(img_path, 'r') as f:
            image1 = self.read_pts(f)
        with open(category, 'r') as f:
            category1 = self.read_seg(f)
        image2, category2 = self.sample_2000(image1, category1)
        if not self.valid:
            theta = random.random() * 360
            image2 = utils.rotation_z(utils.add_noise(image2), theta)

        return {'image': np.array(image2, dtype="float32"), 'category': category2.astype(int)}

    def read_pts(self, file):
        verts = np.genfromtxt(file)
        return utils.cent_norm(verts)
        # return verts

    def read_seg(self, file):
        verts = np.genfromtxt(file, dtype=(int))
        return verts

    def sample_2000(self, pts, pts_cat):
        res1 = np.concatenate((pts, np.reshape(pts_cat, (pts_cat.shape[0], 1))), axis=1)
        res = np.asarray(random.choices(res1, weights=None, cum_weights=None, k=2000))
        images = res[:, 0:3]
        categories = res[:, 3]
        categories -= np.ones(categories.shape)
        return images, categories
