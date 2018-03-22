# In[]
import cv2 
import numpy as np   
import os 
from operator import eq
import time 


os.environ["CUDA_VISIBLE_DEVICES"]="0"

srcbase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/2k_2k/Imgs"
dstbase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180322_152301_HalfImg"
folders = ["Normal", "Abnormal"]
lowerFolders = ["1_AS", "2_AR", "3_MS", "4_MR", "5_AS+AR", "6_MS_MR"]
srcPath = []
dstPath = []

for folder in folders:   
    if eq(folder, "Normal"):
        srcPath.append(srcbase + "/" + folder)
        dstPath.append(dstbase + "/" + folder)
        if(not os.path.isdir(dstbase + "/" + folder)):
            os.mkdir(dstbase + "/" + folder)
    else:
        if(not os.path.isdir(dstbase + "/" + folder)):
            os.mkdir(dstbase + "/" + folder)
        
        for lowerFolder in lowerFolders:
            srcPath.append(srcbase + "/" + folder + "/" + lowerFolder)
            dstPath.append(dstbase + "/" + folder + "/" + lowerFolder)
            if(not os.path.isdir(dstbase + "/" + folder + "/" + lowerFolder)):
                os.mkdir(dstbase + "/" + folder + "/" + lowerFolder)


def run_Modyfying(srcPaths, dstPaths):
    
    for i, srcPath in enumerate(srcPaths) : 
        for file in os.listdir(srcPath):
            ModyfyingType_2(srcPath, dstPaths[i], file)

def ModyfyingType_1(srcPath, dstPath, filename):
    
    left_Path = dstPath + "/Left"
    Right_Path = dstPath + "/Right"

    if(not os.path.isdir(left_Path)):
        os.mkdir(left_Path)
    if(not os.path.isdir(Right_Path)):
        os.mkdir(Right_Path)

    img = cv2.imread(srcPath + "/" + filename)
    img = np.asarray(img, dtype = "int16")

    height, width, depth = img.shape

    ## left lung 

    left_img = img[:, :int(width/2), :]
    right_img = img[ :, int(width/2):,:]

    cv2.imwrite(Right_Path + "/"+filename, left_img)
    cv2.imwrite(left_Path + "/"+filename, right_img)

    print(srcPath + "/" + filename)


def ModyfyingType_2(srcPath, dstPath, filename):
    
    left_Path = dstPath + "/Left"
    Right_Path = dstPath + "/Right"

    if(not os.path.isdir(left_Path)):
        os.mkdir(left_Path)
    if(not os.path.isdir(Right_Path)):
        os.mkdir(Right_Path)

    img = cv2.imread(srcPath + "/" + filename)
    img = np.asarray(img, dtype = "int16")

    width, height, depth = img.shape

    ## left lung 
    left_img = np.copy(img)
    left_img[ :, int(width/2):,:] = 0
    right_img = np.copy(img)
    right_img[:, :int(width/2), :] = 0

    cv2.imwrite(Right_Path + "/"+filename, left_img)
    cv2.imwrite(left_Path + "/"+filename, right_img)

    print(srcPath + "/" + filename)




run_Modyfying(srcPath, dstPath)
    



