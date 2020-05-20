import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import torch
import argparse
from PIL import Image
from datetime import datetime
from torch.autograd import Variable
from UnwrappedFace import UnwrappedFaceWeightedAverage, UnwrappedFaceWeightedAveragePose
import torchvision
from torchvision.transforms import ToTensor, Compose, Scale, Resize
import matplotlib.image as mpimg
import sys
from utilities import create_video
from tqdm import tqdm


def resizer(file_path):
    img = Image.open(file_path)
    transform_resize = Compose([Resize((256,256))])
    cropped = transform_resize(img)
    cropped.save(file_path, "JPEG")


driver_path = "C:/Users/milja/OneDrive/Desktop/cutframes/"
driver_imgs_total = [driver_path + d for d in sorted(os.listdir(driver_path))][0:]
for img in driver_imgs_total:
    resizer(img)


for i in tqdm(range(1,11)):
    driver_path = "C:/Users/milja/OneDrive/Desktop/cutframes"+str(i)+"/"
    driver_imgs_total = [driver_path + d for d in sorted(os.listdir(driver_path))][0:]
    for img in driver_imgs_total:
        resizer(img)
