import matplotlib.image as mpimg
import os
import sys
from PIL import Image
import numpy as np
import cv2
import argparse
from random import seed
from random import randint
seed(1)


def pil_to_cv2(pilimage):
    pil_image = pilimage.convert('RGB')
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image



def append_images(images):
    images = [Image.open(x) for x in images]
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)
    new_im = Image.new('RGB', (total_width, max_height))
    
    x_offset = 0
    for im in images:
      new_im.paste(im, (x_offset,0))
      x_offset += im.size[0]
    #new_im.show()
    #new_im.save('test.jpg')
    return new_im



def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = mpimg.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images



def create_video(image_folder,video_name):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(video_name, 0, 10., (width,height))

    for image in images:
        #print(os.path.join(image_folder, image))
        #img = cv2.imread(os.path.join(image_folder, image))
        #cv2.imshow('image',img)
        #cv2.waitKey(0)
        video.write(cv2.resize(cv2.imread(os.path.join(image_folder, image)),(width,height)))
        print(image)
    cv2.destroyAllWindows()
    video.release()
    return video



def create_comparative_video(video_name,directories,repeat):
    deepfake_sets = [0]*directories.size
    for i in range(directories.size):
        images = [img for img in os.listdir(directories[i]) if img.endswith(".jpg")]
        deepfake_sets[i] = images
    frames = [0]*len(deepfake_sets[0])

    for i in range(len(deepfake_sets[0])):
        frame_parts = [0] * directories.size
        for j in range(directories.size):
            frame_parts[j] = os.path.join(directories[j], deepfake_sets[j][i])
        frames[i] = np.array(pil_to_cv2(append_images(frame_parts)))
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(video_name, 0, 19, (width,height))
    counter = 0
    if repeat:
        counter = 20
    for i in range(counter):
        for image in frames:
            video.write(image)
    cv2.destroyAllWindows()
    video.release()
    return video    







