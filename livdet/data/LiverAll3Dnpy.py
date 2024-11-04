import os.path
import sys

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
import torchvision.transforms as TVT
from torchvision import transforms
from PIL import Image
import random

from .data_loader import register_data_params, register_dataset_obj
from .dataaugment import tensor_transforms3D

class LiverAll3Dnpy(data.Dataset):
    def __init__(self, root, datasets=tuple(), mode='train', normalize='minmax_image', transform=None, target_transform=None, crop_size=128, label_type='lesion'):
        self.root = root
        sys.path.append(root)
        self.datasets = datasets
        self.datasize = len(datasets)
        self.mode = mode
        self.normalize = normalize
        self.ids, self.datasize_accum, self.imgsize = self.collect_ids()
        self.transform = transform
        self.target_transform = target_transform
        self.center_crop = TVT.CenterCrop((crop_size, crop_size))
        self.crop_size = crop_size
        self.label_type = label_type
        self.image_size = 128
        self.slide_per_iter = 64
        self.num_cls = 1
        self.contourname = 'Contours'
        self.decayparam = {'scale': 1.0, 'alpha': 3.0, 'r': 15.0}

    def collect_ids(self):
        ids_dict = dict()
        num_img_accum = list()
        num_img_accum.append(0)
        num_img = 0
        for data in self.datasets:
            im_dir = os.path.join(self.root, data, 'images')
            ids = []
            for filename in sorted(os.listdir(im_dir)):
                if filename.endswith('.png'):
                    ids.append(filename[:-4])
            ids_dict[data] = ids
            # ids_dict[data+'-size'] = len(ids)
            num_img += len(ids) # total number of images in the training set in a single value
            num_img_accum.append(num_img) # total number of images in the training set in a list
        return ids_dict, num_img_accum, num_img

    def img_path(self, data, id):
        fmt = '{}/images/{}.png'
        path = fmt.format(data, id)
        return os.path.join(self.root, path)

    def img_path_npy(self, data):
        fmt = '{}/image.npy'
        path = fmt.format(data)
        return os.path.join(self.root, path)

    def label_path(self, data, id):
        fmt = '{}/labels/{}_label.png'
        path = fmt.format(data, id)
        return os.path.join(self.root, path)

    def label_path_npy(self, data):
        fmt = '{}/lesion.npy'
        path = fmt.format(data)
        return os.path.join(self.root, path)

    def mask_path(self, data, id):
        fmt = '{}/masks/{}_mask.png'
        path = fmt.format(data, id)
        return os.path.join(self.root, path)

    def mask_path_npy(self, data):
        fmt = '{}/liver.npy'
        path = fmt.format(data)
        return os.path.join(self.root, path)

    def __getitem__(self, index):
        if self.mode == 'train':
            cur_data = self.datasets[index % self.datasize]
            if len(self.ids[cur_data]) < self.slide_per_iter:
                start_id = 0
                end_id = len(self.ids[cur_data])
            else:
                start_id = random.randint(0, len(self.ids[cur_data]) - self.slide_per_iter) # including the ending point
                end_id = start_id + self.slide_per_iter

            imagename = self.img_path_npy(cur_data)
            labelname = self.label_path_npy(cur_data)
            maskname = self.mask_path_npy(cur_data)
            image3d = torch.zeros(self.slide_per_iter, self.crop_size, self.crop_size)
            image3d[0:end_id-start_id] = self.center_crop(torch.from_numpy(np.load(imagename)[:,:,start_id:end_id]).permute(2,0,1))
            image3d = image3d.float() / 255.0
            if self.label_type == 'lesion':
                label3d = torch.zeros(self.slide_per_iter, self.crop_size, self.crop_size, dtype=torch.uint8)
                label3d[0:end_id-start_id] = self.center_crop(torch.from_numpy(np.load(labelname)[:,:,start_id:end_id]).permute(2,0,1))
                image3d, label3d, _ = tensor_transforms3D(image3d, label=label3d)
                image3d = torch.unsqueeze(image3d, 0)
                label3d = torch.unsqueeze(label3d, 0)
                return image3d, label3d
        elif self.mode == 'validation':
            cur_data = self.datasets[index]
            if len(self.ids[cur_data]) < self.slide_per_iter:
                start_id = 0
                end_id = len(self.ids[cur_data])
            else:
                start_id = random.randint(0, len(self.ids[cur_data]) - self.slide_per_iter)
                end_id = start_id + self.slide_per_iter

            imagename = self.img_path_npy(cur_data)
            labelname = self.label_path_npy(cur_data)
            maskname = self.mask_path_npy(cur_data)
            image3d = torch.zeros(self.slide_per_iter, self.crop_size, self.crop_size)
            image3d[0:end_id-start_id] = self.center_crop(torch.from_numpy(np.load(imagename)[:,:,start_id:end_id]).permute(2,0,1))
            image3d = image3d.float() / 255.0
            image3d = torch.unsqueeze(image3d, 0)
            if self.label_type == 'lesion':
                label3d = torch.zeros(self.slide_per_iter, self.crop_size, self.crop_size, dtype=torch.uint8)
                label3d[0:end_id-start_id] = self.center_crop(torch.from_numpy(np.load(labelname)[:,:,start_id:end_id]).permute(2,0,1))
                label3d = torch.unsqueeze(label3d, 0)
                return image3d, label3d
        else:
            cur_data = self.datasets[index]
            imagename = self.img_path_npy(cur_data)
            image3d = torch.from_numpy(np.load(imagename)).permute(2,0,1)
            image3d = image3d.float() / 255.0
            image3d = torch.unsqueeze(image3d, 0)
            return image3d, cur_data, self.ids[cur_data]

    def __len__(self):
        return self.datasize

@register_dataset_obj('liver3Dnpy')
def liver3Dnpy(root, datasets=tuple(), mode=None, normalize=None, transform=None, target_transform=None, crop_size=128, label_type='lesion'):
    assert 'liver' == root.split('/')[1]
    print('liver3Dnpy: ' + 'root = ' + root + ', mode = ' + mode)
    print('datasets', datasets)
    data = LiverAll3Dnpy(root=root, datasets=datasets, mode=mode, normalize=normalize, transform=transform, target_transform=target_transform, crop_size=crop_size, label_type=label_type)
    return data

if __name__ == '__main__':
    cs = LiverAll3Dnpy('/x/LiverAll3Dnpy')
