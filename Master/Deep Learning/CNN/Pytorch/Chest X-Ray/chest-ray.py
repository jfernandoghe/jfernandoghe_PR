import os
import glob
import h5py
import shutil
import imgaug as aug
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import imgaug.augmenters as iaa
from os import listdir, makedirs, getcwd, remove
from os.path import isfile, join, abspath, exists, isdir, expanduser
from PIL import Image
from pathlib import Path
from skimage.io import imread
from skimage.transform import resize

# Define path to the data directory
data_dir = Path('/home/jfernandoghe/Documents/x_Datasets/Dataset/Chest_ray/')
# Path to train directory (Fancy pathlib...no more os.path!!)
train_dir = data_dir / 'train'
# Path to validation directory
val_dir = data_dir / 'val'
# Path to test directory
test_dir = data_dir / 'test'
# Get the path to the normal and pneumonia sub-directories
normal_cases_dir = train_dir / 'NORMAL'
pneumonia_cases_dir = train_dir / 'PNEUMONIA'
# Get the list of all the images
normal_cases = normal_cases_dir.glob('*.jpeg')
pneumonia_cases = pneumonia_cases_dir.glob('*.jpeg')
# An empty list. We will insert the data into this list in (img_path, label) format
train_data = []
# Go through all the normal cases. The label for these cases will be 0
for img in normal_cases:
    train_data.append((img,0))
# Go through all the pneumonia cases. The label for these cases will be 1
for img in pneumonia_cases:
    train_data.append((img, 1))
# Get a pandas dataframe from the data we have in our list
train_data = pd.DataFrame(train_data, columns=['image', 'label'],index=None)
# Shuffle the data
train_data = train_data.sample(frac=1.).reset_index(drop=True)
# How the dataframe looks like?
train_data.head()