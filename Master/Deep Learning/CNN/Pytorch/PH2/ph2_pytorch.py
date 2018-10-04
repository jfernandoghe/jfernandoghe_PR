import torch # Tensor Package (for use on GPU)
import os
from torch.utils.data import DataSet
from torch.autograd import Variable # for computational graphs
import torch.nn as nn ## Neural Network package
import torch.nn.functional as F # Non-linearities package
import torch.optim as optim # Optimization package
from torch.utils.data import Dataset, TensorDataset, DataLoader # for dealing with data
import torchvision # for dealing with vision data
import torchvision.transforms as transforms # for modifying vision data to run it through models
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt # for plotting
import numpy as np

roots="/home/jfernandoghe/Documents/x_Datasets/Dataset/PH2/PH2Dataset/"
filename = "/home/jfernandoghe/Documents/x_Datasets/Dataset/PH2/PH2Dataset/PH2_dataset.txt"


# file =open(filename, "r")
# print(file.read())

class ImageData(torch.utils.data.DataSet):
    def __init__(self, root=roots, loader=image_load_func, transform=None):
        self.root = root
        self.files = os.listdir(self.root)
        self.loader = loader
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        return self.transform(self.loader(os.path.join(self.root, self.files[index])))


# for batch in zip(loader_1, ..., loader_8):
#     batch = torch.cat(batch, dim=0)
