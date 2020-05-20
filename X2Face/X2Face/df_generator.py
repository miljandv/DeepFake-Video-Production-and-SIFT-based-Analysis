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

cnt = 0
generate_imgs = True

def get_id_based_on_time():
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d_%H_%M_%S")
    return current_time 

t = torch.cuda.get_device_properties(0).total_memory
print(t)


parser = argparse.ArgumentParser(description='Face2Face_generator')
#default_save_folder = './generated/'
parser.add_argument('--save_to', type=str, default='C:/Users/milja/source/repos/DeepFake/X2Face/X2Face/generated', help='Location to save to')
parser.add_argument('--model_name',type=str,default='x2face_model_start.pth', help='Name of the pth model file to be used')
opt = parser.parse_args()
default_save_folder = opt.save_to
default_model_name = opt.model_name






def run_batch(source_images, pose_images, requires_grad=False, volatile=False):
    return model(pose_images, *source_images)

BASE_MODEL = 'C:/Users/milja/source/repos/DeepFake/X2Face/X2Face/release_models/'
state_dict = torch.load(BASE_MODEL + default_model_name)

model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=128)
model.load_state_dict(state_dict['state_dict'])

model = model.cuda()

model = model.eval()

driver_path = "C:/Users/milja/OneDrive/Desktop/cutframes/"
source_path = "D:/ML/Telfor J/CAMERA/Lily_Cole/"
current_time = get_id_based_on_time()
save_folder = current_time

path, dirs, files = next(os.walk(driver_path))
file_count = len(files)
print(file_count)


def load_img(file_path):
    img = Image.open(file_path)
    transform = Compose([Resize((256,256)), ToTensor()])
    transform_resize = Compose([Resize((256,256))])
    cropped = transform_resize(img)
    cropped.save(file_path, "JPEG")
    return Variable(transform(img)).cuda()



    

driver_imgs_total = [driver_path + d for d in sorted(os.listdir(driver_path))][0:]
source_imgs  = [source_path + d for d in sorted(os.listdir(source_path))][0:5]

save_to = default_save_folder +'/'+ save_folder
os.mkdir(save_to)

for img in driver_imgs_total:
    resizer(img)
exit()


if generate_imgs:
    apnd = 0
    if len(driver_imgs_total)%5>0:
        appnd = 1
    for i in range((int)(len(driver_imgs_total)/5) + appnd):
        source_images = []
        driver_imgs = []
        driver_imgs = driver_imgs_total[i*5:i*5+5]
    
        driver_images = None
        for img in driver_imgs:
            if driver_images is None:
                driver_images = load_img(img).unsqueeze(0)
            else:
                driver_images = torch.cat((driver_images, load_img(img).unsqueeze(0)), 0)
        
    
        for img in source_imgs:
            source_images.append(load_img(img).unsqueeze(0).repeat(len(driver_imgs), 1, 1, 1))
            
        # Run the model for each
        result = run_batch(source_images, driver_images)
    
    
        result = result.clamp(min=0, max=1)
        ka = result[1].detach().cpu().data.numpy()
        ka = np.transpose(ka, (2,1,0))
        print(ka.shape)
        ii = Image.fromarray(ka, 'RGB')
        
        img = torchvision.utils.make_grid(result.cpu().data)
        fig_size = plt.rcParams["figure.figsize"]
        fig_size[0] = 24.
        fig_size[1] = 24.
        plt.rcParams["figure.figsize"] = fig_size
        plt.axis('off')
        result_images = img.permute(1,2,0).numpy()
        driving_images = torchvision.utils.make_grid(driver_images.cpu().data).permute(1,2,0).numpy()
        print("The results is: ")
        
    
    
    
        
        
        for i in result.cpu().data:
            print(str(cnt))
            torchvision.utils.save_image(i, filename = save_to+'/'+str(str(cnt).zfill(6))+'.jpg')
            cnt+=1
    
        del result
        del driver_images
        del driving_images
        del source_images
        torch.cuda.empty_cache()
        c = torch.cuda.memory_cached(0)
        a = torch.cuda.memory_allocated(0)
        print(a)
        print(c)

if not generate_imgs:
    save_to = "C:/Users/milja/source/repos/DeepFake/X2Face/X2Face/generated/2020-04-29_06_22_59"

create_video(save_to,"D:/ML/Telfor J/DeepFakes/X2Face/c.avi")
