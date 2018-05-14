# In[]
import cv2 
import numpy as np   
import os 
from operator import eq
import random
import matplotlib.pyplot as plt 
from skimage import io
import shutil

os.environ["CUDA_VISIBLE_DEVICES"]="0"

LungMaskPath = "D:/[Data]/[Lung_Segmentation]/WholeDataSetMask/Whole" # + Imgs

DataPath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base"

dstPath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base_Cropped"

Folders = ["Imgs", "Masks"] 

MasksType = ["Aortic Knob", "Lt Lower CB", "Pulmonary Conus", "Rt Lower CB", "Rt Upper CB", "DAO" , "Carina" , "LAA"]


FolderTypes = ["train", "validation", "test"]




for folder in FolderTypes :
    
    ImgPath = DataPath + "/" + Folders[0] + "/" + folder
    dstImgPath = dstPath + "/" + Folders[0] + "/" + folder


    if(not os.path.isdir(dstImgPath)):
        os.mkdir(dstImgPath)

    for file in os.listdir(ImgPath) : 
        print(file)
        LungMask = cv2.imread(LungMaskPath + "/" + file, 0)
        _, LungMask = cv2.threshold(LungMask, 127, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(LungMask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        rects = []
        for cnt in contours:
            rects.append(cv2.boundingRect(cnt))

        tcomx = 10
        tcomy = 10 
        bcomx = 10  
        bcomy = 10

        top_x, top_y, bottom_x, bottom_y = 0, 0 ,0, 0

        rects.sort()

        top_x = min([x for (x, y, w, h) in rects])  - tcomx  #26
        top_y = min([y for (x, y, w, h) in rects])  - tcomy  #26
        bottom_x = max([x+w for (x, y, w, h) in rects]) + bcomx  #234
        bottom_y = max([y+h for (x, y, w, h) in rects])  + bcomy #227
        
        #print(top_x, top_y, bottom_x, bottom_y)

        if(top_x <=0 ) : top_x = tcomx
        if(top_y <=0 ) : top_y = tcomy
        
        if(bottom_x >= 1024 ) : bottom_x = 1024 - tcomx
        if(bottom_y >= 1024 ) : bottom_y = 1024 - tcomy
        
        
        
        Img = cv2.imread(ImgPath + "/" + file, 0)
        
        ImgCrop = Img[top_y*2:bottom_y*2, top_x*2:bottom_x*2]
        ImgCrop = cv2.resize(ImgCrop, (1024,1024))
        cv2.imwrite(dstImgPath + "/" + file, ImgCrop)

        for maskType in MasksType:
            maskPath = DataPath + "/" + Folders[1] + "/" + maskType + "/" + folder
            maskdstPath = dstPath + "/" + Folders[1] + "/" + maskType + "/" + folder

            if(not os.path.isdir(dstPath + "/" + Folders[1] + "/" + maskType)):
                os.mkdir(dstPath + "/" + Folders[1] + "/" + maskType)

            if(not os.path.isdir(maskdstPath)):
                os.mkdir(maskdstPath)


            mask = cv2.imread(maskPath + "/" + file, 0)
            ImgCrop = mask[top_y*2:bottom_y*2, top_x*2:bottom_x*2]
            ImgCrop = cv2.resize(ImgCrop, (1024,1024))
            cv2.imwrite(maskdstPath + "/" + file, ImgCrop)
        





