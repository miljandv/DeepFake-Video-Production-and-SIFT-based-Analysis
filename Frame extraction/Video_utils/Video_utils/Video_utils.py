import cv2
import dlib
from imutils import face_utils
import numpy as np
from PIL import Image
from tqdm import tqdm
import time
import imageio
import os, sys



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
    print(new_im.size)
    return new_im


class TargetFormat(object):
    GIF = ".gif"
    MP4 = ".mp4"
    AVI = ".avi"



def crop_image(lands,path_to_image):
    data = lands
    mins = np.min(data, axis=0)
    maxs = np.max(data, axis=0)
    im = Image.open(path_to_image)
    imn = np.asarray( im, dtype="int32")
    width, height = im.size
    left = mins[0]
    top = mins[1]
    right = maxs[0]
    bottom = maxs[1]
    im1 = im.crop((left-60, top-74, right+60, bottom+60)) 
    #im1.show() 
    return im1


def PIL_to_CV2():
    pil_image = PIL.Image.open('Image.jpg').convert('RGB') 
    open_cv_image = numpy.array(pil_image) 
    open_cv_image = open_cv_image[:, :, ::-1].copy() 



def get_landmarks(path_to_image):
    p = r"C:/Users/milja/source/repos/DeepFake/Frame extraction/Video_utils/Video_utils/models/shape_predictor_68_face_landmarks.dat"
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
    image = cv2.imread(path_to_image,cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    shape = None
    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        if shape == None:
            return None,False
        shape = face_utils.shape_to_np(shape)
    keypoint_drawer(shape,path_to_image)
    return shape,True


def crop_html_frame(path_to_image):
    im = Image.open(path_to_image)
    im1 = im.crop((280, 52, 606, 377))
    #im1.show() 
    return im1



def frame_extractor(path_to_video,html):
    cap= cv2.VideoCapture(path_to_video)
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        im_name = "C:/Users/milja/OneDrive/Desktop/frames/"+str(i).zfill(6)+".jpg"
        cv2.imwrite(im_name,frame)
        
        if html:
           cropped = crop_html_frame(im_name)
           nm = "C:/Users/milja/OneDrive/Desktop/cutframes8/"+str(i).zfill(6)+".jpg"
           cropped.save(nm, "JPEG")
        elif success==True:
           lands, success = get_landmarks(lands,im_name)
           keypoint_drawer(lands, im_name)
           cropped = crop_frame(im_name)
           #print(cropped.size)
           nm = "C:/Users/milja/OneDrive/Desktop/cutframes4/"+str(i).zfill(6)+".jpg"
           cropped.save(nm, "JPEG")
        i+=1
        print("Image: ",i)
     
    cap.release()
    cv2.destroyAllWindows()
    return None



def create_video(image_folder,video_name,fps):
    images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape
    
    video = cv2.VideoWriter(video_name, 0, fps, (width,height))

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
    video = cv2.VideoWriter(video_name, 0, 10, (width,height))
    counter = 1
    if repeat:
        counter = 20
    for i in range(counter):
        for image in frames:
            video.write(cv2.resize(image,(width,height)))
    cv2.destroyAllWindows()
    video.release()
    return video    



def check_fps(video_path):
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    print(fps)


def keypoint_drawer(keypoints,path_to_image):
    image = cv2.imread(path_to_image,cv2.IMREAD_COLOR)
    for x in keypoints:
        image = cv2.circle(image, (x[0], x[1]), 1, (0,255,0), 3)
    #cv2.imshow( "Display window", image )
    #cv2.waitKey(0)
    #cv2.imwrite( r"C:\BASE\landmarks_paper_2.jpg", image )
    return image


def convertFile(inputpath, targetFormat):
    outputpath = os.path.splitext(inputpath)[0] + targetFormat
    print("converting\r\n\t{0}\r\nto\r\n\t{1}".format(inputpath, outputpath))

    reader = imageio.get_reader(inputpath)
    fps = reader.get_meta_data()['fps']

    writer = imageio.get_writer(outputpath, fps=fps)
    for i,im in enumerate(reader):
        sys.stdout.write("\rframe {0}".format(i))
        sys.stdout.flush()
        writer.append_data(im)
    print("\r\nFinalizing...")
    writer.close()
    print("Done.")

#convertFile("C:\\Users\\milja\\OneDrive\\Desktop\\EwokPr0n.mp4", TargetFormat.GIF)




def main():
    #dirs = "C:/Users/milja/OneDrive/Desktop/cutframes C:/Users/milja/source/repos/DeepFake/X2Face/X2Face/generated/2020-05-07_10_25_06"
    #dirs= dirs.split(' ')
    #folders = np.array(dirs)
    #create_comparative_video(r"C:\Users\milja\OneDrive\Desktop\compare.avi",folders,False)
    #frame_extractor(r"D:\ML\Telfor J\DeepFakes\First order motion model\AG\source_historic3 driving_cam.mp4",True)
    create_video(r"C:\Users\milja\OneDrive\Desktop\cutframes8","C:/Users/milja/OneDrive/Desktop/8.avi",30)
    #check_fps(r"C:\Users\milja\OneDrive\Desktop\AGinput1.mp4")
    #convertFile(r"C:\Users\milja\Downloads\Dragica's reaction to Miljan's existence.mp4",TargetFormat.GIF)



if __name__ == '__main__':
    main()





