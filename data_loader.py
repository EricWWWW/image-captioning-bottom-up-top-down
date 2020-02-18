import os
import json
import h5py
import torch
from torch.utils.data import Dataset

# # 多线程报错，进行单线程
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')


class CaptionDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, data_name, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}

        # Open hdf5 file where images are stored
        self.train_hf = h5py.File(data_folder + '/train36.hdf5', 'r')
        self.train_features = self.train_hf['image_features']
        self.val_hf = h5py.File(data_folder + '/val36.hdf5', 'r')
        self.val_features = self.val_hf['image_features']

        # Captions per image
        self.cpi = 5

        # Load encoded captions
        with open(os.path.join(data_folder, self.split + '_CAPTIONS_' + data_name + '.json'), 'r') as j:
            self.captions = json.load(j)

        # Load caption lengths
        with open(os.path.join(data_folder, self.split + '_CAPLENS_' + data_name + '.json'), 'r') as j:
            self.caplens = json.load(j)

        # Load bottom up image features distribution
        with open(os.path.join(data_folder, self.split + '_GENOME_DETS_' + data_name + '.json'), 'r') as j:
            self.objdet = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = len(self.captions)

    def __getitem__(self, i):

        # The Nth caption corresponds to the (N // captions_per_image)th image
        # 如果在类中定义了__getitem__()方法，那么他的实例对象（假设为P）就可以这样P[key]取值。
        # 当实例对象做P[key]运算时，就会调用类中的__getitem__()方法。
        objdet = self.objdet[i // self.cpi]
        img_type = objdet[0]
        img_index = objdet[1]

        # Load bottom up image features
        if img_type == "v":
            img_feature = torch.FloatTensor(self.val_features[img_index])
        else:
            img_feature = torch.FloatTensor(self.train_features[img_index])

        caption = torch.LongTensor(self.captions[i])

        caplen = torch.LongTensor([self.caplens[i]])

        if self.split is 'TRAIN':
            return img_feature, caption, caplen
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_captions = torch.LongTensor(self.captions[((i // self.cpi) * self.cpi):(((i // self.cpi) * self.cpi) + self.cpi)])
            return img_feature, caption, caplen, all_captions

    def __len__(self):
        return self.dataset_size
