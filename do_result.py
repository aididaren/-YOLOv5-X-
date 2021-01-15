import pandas as pd
import os
from cv2 import cv2
import numpy as np

Label_Path="./runs/detect/exp5/labels/"
Image_path="./runs/detect/exp5/"

Image_List=os.listdir(Image_path) 
detect = [ img for img in os.listdir(Image_path) if img[-4:] == '.jpg'] 

with open('./runs/result.txt', 'w') as resFile:  
    for img in detect:
        img_name = img[:-4]
        print(img_name)
        imageurl = Image_path + img
        img = cv2.imread(imageurl)

        label_file = Label_Path + img_name + '.txt'

        if os.path.isfile(label_file):
            with open(label_file, 'r') as labelfile:
                for line in labelfile.readlines():
                    data = line.strip('\n').split(' ')
                    
                    x_1 = float(data[1]) * img.shape[1] - w * img.shape[1]/2 
                    y_1 = float(data[2]) * img.shape[0] - h * img.shape[0]/2 
                    x_2 = float(data[3]) * img.shape[1] + w * img.shape[1]/2 
                    y_2 = float(data[4]) * img.shape[0] + h * img.shape[0]/2 
                    line = '{} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f} \n'.format(img_name, float(data[5]), x_1, y_1, x_2, y_2)
                    resFile.writelines(line)
       
