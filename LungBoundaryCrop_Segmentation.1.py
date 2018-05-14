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

LungMaskPath = "D:/[Data]/[Lung_Segmentation]/TempDataSet/Masks/pngfile_post" # + Imgs

DataPath = "D:/[Data]/[Lung_Segmentation]/TempDataSet/Imgs/pngfile"

dstPath = "D:/[Data]/[Lung_Segmentation]/TempDataSet/Imgs_Cropped/pngfile"

#Folders = ["Imgs", "Masks"] 

#MasksType = ["Aortic Knob", "Lt Lower CB", "Pulmonary Conus", "Rt Lower CB", "Rt Upper CB", "DAO" , "Carina" , "LAA"]
MasksType = None

FolderTypes = ["train", "validation", "test"]




#for folder in FolderTypes :
    
    #ImgPath = DataPath + "/" + Folders[0] + "/" + folder
    #dstImgPath = dstPath + "/" + Folders[0] + "/" + folder
ImgPath = DataPath
dstImgPath = dstPath

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

    if(len(rects) == 2):

        outerX_L = min([x for (x, y, w, h) in rects])
        outerX_R = max([x + w for (x, y, w, h) in rects])
        innerX_L = min([x + w for (x, y, w, h) in rects])
        innerX_R = max([x for (x, y, w, h) in rects])

        innerCenter = int((innerX_L + innerX_R)//2)
        outerCenter = int((outerX_L + outerX_R)//2)


        center_X = 0
        if( abs(innerCenter - 512) > abs(outerCenter - 512)) : 
            center_X = outerCenter
        else:
            center_X = innerCenter


        halfWidth = max(abs(outerX_L - center_X), abs(outerX_R - center_X))

        top_y = min([y for (x, y, w, h) in rects])
        bottom_y = max([y+h for (x, y, w, h) in rects])
        top_x = center_X - halfWidth
        bottom_x = center_X + halfWidth 
        #print(halfWidth)
    elif (len(rects) == 1) : 
        x, y, w, h = rects[0]
        center_X = 512 
        halfWidth = max(abs(x - center_X), abs(x + w - center_X))
        top_y = y
        bottom_y = y + h
        top_x = center_X - halfWidth
        bottom_x = center_X + halfWidth 
    else : 
        continue

    top_x = top_x  - tcomx  #26
    top_y = top_y  - tcomy  #26
    bottom_x = bottom_x + bcomx  #234
    bottom_y = bottom_y  + bcomy #227
    
    #print(top_x, top_y, bottom_x, bottom_y)

    if(top_x <=0 ) : top_x = tcomx
    if(top_y <=0 ) : top_y = tcomy
    
    if(bottom_x >= 1024 ) : bottom_x = 1024 - tcomx
    if(bottom_y >= 1024 ) : bottom_y = 1024 - tcomy
    
    
    
    
    
    
    Img = cv2.imread(ImgPath + "/" + file, 0)
    Img = cv2.resize(Img, (2048,2048))
    ImgCrop = Img[top_y*2:bottom_y*2, top_x*2:bottom_x*2]
    ImgCrop = cv2.resize(ImgCrop, (1024,1024))
    cv2.imwrite(dstImgPath + "/" + file, ImgCrop)

    # for maskType in MasksType:
    #     maskPath = DataPath + "/" + Folders[1] + "/" + maskType + "/" + folder
    #     maskdstPath = dstPath + "/" + Folders[1] + "/" + maskType + "/" + folder

    #     if(not os.path.isdir(dstPath + "/" + Folders[1] + "/" + maskType)):
    #         os.mkdir(dstPath + "/" + Folders[1] + "/" + maskType)

    #     if(not os.path.isdir(maskdstPath)):
    #         os.mkdir(maskdstPath)


    #     mask = cv2.imread(maskPath + "/" + file, 0)
    #     ImgCrop = mask[top_y*2:bottom_y*2, top_x*2:bottom_x*2]
    #     ImgCrop = cv2.resize(ImgCrop, (1024,1024))
    #     cv2.imwrite(maskdstPath + "/" + file, ImgCrop)
        





