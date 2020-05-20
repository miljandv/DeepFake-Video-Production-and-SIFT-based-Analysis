import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import os
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
from UnwrappedFace import UnwrappedFaceWeightedAverage, UnwrappedFaceWeightedAveragePose
import torchvision
from torchvision.transforms import ToTensor, Compose, Scale, Resize
import matplotlib.image as mpimg
from utils_3_7 import get_id_based_on_time


parser = argparse.ArgumentParser(description='Face2Face_generator')
#default_save_folder = './generated/'
parser.add_argument('--save_to', type=str, default='./generated/', help='Location to save to')
parser.add_argument('--model_name',type=str,default='x2face_model.pth', help='Name of the pth model file to be used')
opt = parser.parse_args()
default_save_folder = opt.save_to
default_save_folder = "D:/Repos/vision.hu.belgradeinterns/Hmd2Face/X2Face/generated"
default_model_name = opt.model_name
default_model_name  = "modelscopyWeightsmodel_epoch_2368.pth"



def run_batch_CUDA_mem(source_images, driver_images):
    source_images = [torch.stack([img] * len(driver_images)).cuda() for img in source_images]
    driver_images = torch.stack(driver_images).cuda()
    result = model(driver_images, *source_images)
    source_images = [img.detach().cpu() for img in source_images]
    driver_images = driver_images.detach().cpu()
    result = result.detach().cpu()
    result = list(result)
    return result



def run_batch(source_images, pose_images, requires_grad=False, volatile=False):
    return model(pose_images, *source_images)

BASE_MODEL = 'my_train/models/' # Change to your path
state_dict = torch.load(BASE_MODEL + default_model_name)

model = UnwrappedFaceWeightedAverage(output_num_channels=2, input_num_channels=3, inner_nc=128)
model.load_state_dict(state_dict['state_dict'])
print(model)
model = model.cuda()

model = model.eval()

driver_path = "D:/Repos/vision.hu.belgradeinterns/Hmd2Face/X2Face/examples/Taylor_Swift/1.6/HMD_DRIVING/"
source_path = "D:/Repos/vision.hu.belgradeinterns/Hmd2Face/X2Face/examples/Taylor_Swift/1.6/HMD_SOURCE/"

current_time = get_id_based_on_time()
save_folder = current_time

path, dirs, files = next(os.walk(driver_path))
file_count = len(files)
print(file_count)

driver_imgs = [driver_path + d for d in sorted(os.listdir(driver_path))][0:2]
source_imgs  = [source_path + d for d in sorted(os.listdir(source_path))][0:5]
print(driver_imgs)

def load_img(file_path):
    img = Image.open(file_path)
    transform = Compose([Resize((256,256)), ToTensor()])
    return Variable(transform(img)).cuda()

source_images = []
for img in source_imgs:
    source_images.append(load_img(img).unsqueeze(0).repeat(len(driver_imgs), 1, 1, 1))
    
driver_images = None
for img in driver_imgs:
    if driver_images is None:
        driver_images = load_img(img).unsqueeze(0)
    else:
        driver_images = torch.cat((driver_images, load_img(img).unsqueeze(0)), 0)

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
plt.imshow(np.vstack((result_images, driving_images)))



save_to = default_save_folder +'/'+ save_folder
os.mkdir(save_to)
cntt = 0
cnto = 0
plt.savefig(save_to+'/global' + '.jpg')

for i in result.cpu().data:
    print(str(cntt),str(cnto))
    torchvision.utils.save_image(i, filename = save_to+'/'+str(cntt)+str(cnto)+'.jpg')
    if cnto == 9:
        cnto = 0
        cntt+=1 
    else:
        cnto+=1

