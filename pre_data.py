import numpy as np
import pandas as pd
import os
from cv2 import cv2

newlabels="./data/Fires/train/labels/"
imagepath="./data/Fires/train/images/"
lablepath="./data/Fires/train/annotations/"
imagelist=os.listdir(imagepath)
lablelist=os.listdir(lablepath)

for i in range(len(imagelist)):
    imageurl=imagepath + imagelist[i]
    
    label_name = imagelist[i][:-4] + '.txt'   
    labelurl=lablepath + label_name
    newlabelurl = newlabels + label_name
    img = cv2.imread(imageurl)
    df=pd.read_table(labelurl,sep=" ",names=["image","class","xmax","ymax","xmin","ymin"])
    
    df["class"]=0  
    df["x"]=(df.xmax+df.xmin)/img.shape[1]/2  
    df["y"]=(df.ymax+df.ymin)/img.shape[0]/2  
    df["w"]= abs((df.xmax-df.xmin)/img.shape[1])  
    df["h"]= abs((df.ymax-df.ymin)/img.shape[0])  

    df.drop(['image', 'xmax','xmin','ymax','ymin'],axis=1, inplace=True)
    df.to_csv(newlabelurl, sep=' ', index=False,header=None)
