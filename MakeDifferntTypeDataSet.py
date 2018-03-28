# In[]
import cv2 
import numpy as np   
import os 
from operator import eq
import time 


os.environ["CUDA_VISIBLE_DEVICES"]="0"

srcbase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180125_103950/Imgs"
dstbase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180322_152301_HalfImg"
folders = ["train", "validation", "test"]
#lowerFolders = ["1_AS", "2_AR", "3_MS", "4_MR", "5_AS+AR", "6_MS_MR"]
srcPath = []
leftdstPath = []
rightdstPath = []

for folder in folders:   

        srcPath.append(srcbase + "/" + folder)
        leftdstPath.append(dstbase + "/Left/" + folder)
        rightdstPath.append(dstbase + "/Right/" + folder)
        if(not os.path.isdir(dstbase + "/Left/" + folder)):
            os.mkdir(dstbase + "/Left/" + folder)
        if(not os.path.isdir(dstbase + "/Right/" + folder)):
            os.mkdir(dstbase + "/Right/" + folder)


def run_Modyfying(srcPaths, leftdstPaths, rightdstPaths):
    
    for i, srcPath in enumerate(srcPaths) : 
        for file in os.listdir(srcPath):
            ModyfyingType_2(srcPath, leftdstPaths[i], rightdstPaths[i],file)

def ModyfyingType_1(srcPath, leftdstPath, rightdstPath, filename):
    
    left_Path = leftdstPath
    Right_Path = rightdstPath

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


def ModyfyingType_2(srcPath, leftdstPath, rightdstPath, filename):
    
    left_Path = leftdstPath
    Right_Path = rightdstPath

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




run_Modyfying(srcPath, leftdstPath, rightdstPath)
    



