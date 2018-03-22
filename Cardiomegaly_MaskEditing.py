# In[]

import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt 


def expand_Size(srcPath, dstPath):

    img = cv2.imread(srcPath)
    img = np.asarray(img, dtype = "uint8")

    kernel = np.ones((11,11))
    img = cv2.dilate(img, kernel, iterations = 2)  
    cv2.imwrite(dstPath, img)



src = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180125_103950"
dst = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180125_103950_Expand_20pixel"

folders = ["train", "test","validation"]
masks = ["Mask_Aortic Knob", 'Mask_Axis', "Mask_Carina","Mask_DAO","Mask_Diaphragm" ,"Mask_LAA",
            "Mask_Lt Lower CB", "Mask_Pulmonary Conus", "Mask_Rt Lower CB", "Mask_Rt Upper CB" ]


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