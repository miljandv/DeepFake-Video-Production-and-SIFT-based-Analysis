import cv2
import dlib
from imutils import face_utils
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import imageio
import os, sys
from Video_utils import create_video

for i in tqdm(range(1,11)):
    driver_path = "C:/Users/milja/OneDrive/Desktop/cutframes"+str(i)+"/"
    create_video(driver_path,"C:/Users/milja/OneDrive/Desktop/"+str(i)+".avi")

