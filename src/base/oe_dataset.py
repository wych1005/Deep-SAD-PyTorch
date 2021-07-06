from torch.utils.data import DataLoader, Subset
from base.base_dataset import BaseADDataset
import PIL
from PIL import Image
import torch
import pickle
from pathlib import Path
from torch.utils.data import Dataset
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import numpy as np


class OEDataset(Dataset):

    def __init__(self, root: str, modality: str, dataset_name: str, train=True, random_state=None, download=False):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train  # training set or test sets
        self.modality = modality

        if self.modality == "ir":
            with open('~/Deep-SAD-PyTorch/pickles/X_train_ir.pkl', 'rb') as f:
                X_train = pickle.load(f).T

            with open('~/Deep-SAD-PyTorch/pickles/X_test_ir.pkl', 'rb') as f:
                X_test = pickle.load(f).T

            with open('~/Deep-SAD-PyTorch/pickles/y_train_ir.pkl', 'rb') as f:
                y_train = pickle.load(f).reshape(741,)

            with open('~/Deep-SAD-PyTorch/pickles/y_test_ir.pkl', 'rb') as f:
                y_test = pickle.load(f).reshape(259,)

        else:

            with open('~/Deep-SAD-PyTorch/pickles/X_train_eo.pkl','rb') as f:
                X_train = pickle.load(f).T

            with open('~/Deep-SAD-PyTorch/pickles/X_test_eo.pkl','rb') as f:
                X_test = pickle.load(f).T

            with open('~/Deep-SAD-PyTorch/pickles/y_train_eo.pkl','rb') as f:
                y_train = pickle.load(f).reshape(370,)

            with open('~/Deep-SAD-PyTorch/pickles/y_test_eo.pkl','rb') as f:
                y_test = pickle.load(f).reshape(81,)

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance) 
        
        scaler = StandardScaler().fit(X_train)
        X_train_stand = scaler.transform(X_train)
        X_test_stand = scaler.transform(X_test)

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_train_stand)
        X_train_scaled = minmax_scaler.transform(X_train_stand)
        X_test_scaled = minmax_scaler.transform(X_test_stand)

        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)
        self.semi_targets = torch.zeros_like(self.targets)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """

        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])
        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)


class Tiled32(Dataset):

    def __init__(self, root: str, modality: str, dataset_name: str, train=True, random_state=None, download=False):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train  # training set or test sets
        self.modality = modality

        if self.modality == "eo":
            with open('~/Deep-SAD-PyTorch/pickles/32tiled_X_train_eo.pkl', 'rb') as f:
                X_train = pickle.load(f)

            with open('~/Deep-SAD-PyTorch/pickles/32tiled_X_test_eo.pkl', 'rb') as f:
                X_test = pickle.load(f)

            with open('~/Deep-SAD-PyTorch/pickles/32tiled_y_train_eo.pkl', 'rb') as f:
                # 370 for ir and 741 for ir
                y_train = pickle.load(f)
                # y_train = pickle.load(f).reshape(370, )

            with open('~/Deep-SAD-PyTorch/pickles/32tiled_y_test_eo.pkl', 'rb') as f:
                # 81 for ir and 259 for ir
                # y_test = pickle.load(f).reshape(81, )
                y_test = pickle.load(f)

        else:

            with open('~/Deep-SAD-PyTorch/pickles/32tiled_X_train_ir.pkl', 'rb') as f:
                X_train = pickle.load(f)

            with open('~/Deep-SAD-PyTorch/pickles/32tiled_X_test_ir.pkl', 'rb') as f:
                X_test = pickle.load(f)

            with open('~/Deep-SAD-PyTorch/pickles/32tiled_y_train_ir.pkl', 'rb') as f:
                # 370 for ir and 741 for ir
                # y_train = pickle.load(f).reshape(741,)
                y_train = pickle.load(f)

            with open('~/Deep-SAD-PyTorch/pickles/32tiled_y_test_ir.pkl', 'rb') as f:
                # 81 for ir and 259 for ir
                y_test = pickle.load(f)
                # y_test = pickle.load(f).reshape(259,)

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)
        scaler = StandardScaler().fit(X_train)
        X_train_stand = scaler.transform(X_train)
        X_test_stand = scaler.transform(X_test)

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_train_stand)
        X_train_scaled = minmax_scaler.transform(X_train_stand)
        X_test_scaled = minmax_scaler.transform(X_test_stand)

        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)
        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """

        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])
        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)

class Tiled64(Dataset):

    def __init__(self, root: str, modality: str, dataset_name: str, train=True, random_state=None, download=False):
        super(Dataset, self).__init__()

        self.classes = [0, 1]

        if isinstance(root, torch._six.string_classes):
            root = os.path.expanduser(root)
        self.root = Path(root)
        self.dataset_name = dataset_name
        self.train = train  # training set or test sets
        self.modality = modality
        print("MODALITY: ", self.modality)

        if self.modality == "eo":
            with open('~/Deep-SAD-PyTorch/pickles/64tiled_X_train_eo.pkl', 'rb') as f:
                X_train = pickle.load(f)

            with open('~/Deep-SAD-PyTorch/pickles/64tiled_X_test_eo.pkl', 'rb') as f:
                X_test = pickle.load(f)

            with open('~/Deep-SAD-PyTorch/pickles/64tiled_y_train_eo.pkl', 'rb') as f:
                # 370 for ir and 741 for ir
                y_train = pickle.load(f)
                # y_train = pickle.load(f).reshape(370, )

            with open('~/Deep-SAD-PyTorch/pickles/64tiled_y_test_eo.pkl', 'rb') as f:
                # 81 for ir and 259 for ir
                # y_test = pickle.load(f).reshape(81, )
                y_test = pickle.load(f)

        else:

            with open('~/Deep-SAD-PyTorch/pickles/64tiled_X_train_ir.pkl', 'rb') as f:
                X_train = pickle.load(f)

            with open('~/Deep-SAD-PyTorch/pickles/64tiled_X_test_ir.pkl', 'rb') as f:
                X_test = pickle.load(f)

            with open('~/Deep-SAD-PyTorch/pickles/64tiled_y_train_ir.pkl', 'rb') as f:
                # 370 for ir and 741 for ir
                # y_train = pickle.load(f).reshape(741,)
                y_train = pickle.load(f)

            with open('~/Deep-SAD-PyTorch/pickles/64tiled_y_test_ir.pkl', 'rb') as f:
                # 81 for ir and 259 for ir
                y_test = pickle.load(f)
                # y_test = pickle.load(f).reshape(259,)

        # Standardize data (per feature Z-normalization, i.e. zero-mean and unit variance)

        scaler = StandardScaler().fit(X_train)
        X_train_stand = scaler.transform(X_train)
        X_test_stand = scaler.transform(X_test)

        # Scale to range [0,1]
        minmax_scaler = MinMaxScaler().fit(X_train_stand)
        X_train_scaled = minmax_scaler.transform(X_train_stand)
        X_test_scaled = minmax_scaler.transform(X_test_stand)

        if self.train:
            self.data = torch.tensor(X_train_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.int64)
        else:
            self.data = torch.tensor(X_test_scaled, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.int64)
        self.semi_targets = torch.zeros_like(self.targets)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, semi_target, index)
        """

        sample, target, semi_target = self.data[index], int(self.targets[index]), int(self.semi_targets[index])
        return sample, target, semi_target, index

    def __len__(self):
        return len(self.data)