# In[]

import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 


def expand_Size(srcPath, dstPath):

    img = cv2.imread(srcPath)
    img = np.asarray(img, dtype = "uint8")

    kernel = np.ones((21,21))
    img = cv2.dilate(img, kernel, iterations = 2)  
    cv2.imwrite(dstPath, img)



src = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base/Masks"
dst = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base_Expand_40pixel/Masks"

folders = ["train", "test","validation"]
masks = MasksTypes = ["Aortic Knob", "Lt Lower CB", "Pulmonary Conus", "Rt Lower CB", "Rt Upper CB", "DAO" , "Carina" , "LAA"]


if not os.path.isdir(dst):
    os.mkdir(dst)

for mask in masks : 
    src_ = src + "/" + mask
    dst_ = dst + "/" + mask
    if not os.path.isdir(dst_):
        os.mkdir(dst_)
    for folder in folders : 
        src__ = src_ + "/" + folder
        dst__ = dst_ + "/" + folder
        if not os.path.isdir(dst__):
            os.mkdir(dst__)
        for file in os.listdir(src__):
            expand_Size(src__ + "/" + file, dst__ + "/" + file)
            print(src__ + "/" + file)