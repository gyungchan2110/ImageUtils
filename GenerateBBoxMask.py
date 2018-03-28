
# In[]
import cv2 
import numpy as np   
import os 
from operator import eq
import random
import matplotlib.pyplot as plt 

os.environ["CUDA_VISIBLE_DEVICES"]="0"

srcbase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180125_103950_Expand_20pixel/Masks/Mask_Rt Upper CB"
dstbase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180125_103950_Expand_20pixel_bbox/Masks/Mask_Rt Upper CB"
folders = ["train", "validation", "test"]
#lowerFolders = ["1_AS", "2_AR", "3_MS", "4_MR", "5_AS+AR", "6_MS_MR"]
srcPath = []
dstPath = []
rightdstPath = []

for folder in folders:   

    srcPath.append(srcbase + "/" + folder)
    dstPath.append(dstbase + "/" + folder)
    #rightdstPath.append(dstbase + "/Right/" + folder)
    if(not os.path.isdir(dstbase + "/" + folder)):
        os.mkdir(dstbase + "/" + folder)
    # if(not os.path.isdir(dstbase + "/Right/" + folder)):
    #     os.mkdir(dstbase + "/Right/" + folder)


def run_Modyfying(srcPaths, dstPaths):
    
    for i, srcPath in enumerate(srcPaths) : 
        for file in os.listdir(srcPath):
            MaskGeneration(srcPath, dstPath[i], file)

def MaskGeneration(srcPath, dstPath, filename):
    
    margin = 10

    mask = cv2.imread(srcPath + "/" + filename)
    mask = np.asarray(mask, dtype = "uint8")
    mask = mask[:,:,0] 
    ret, bin_img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    temp, contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    
    #for cnt in contours:
    x,y, w,h = cv2.boundingRect(contours[0])

    newmask = np.zeros(mask.shape)
    newmask[y:y+h , x-margin:x+w+margin] = 255 

    cv2.imwrite(dstPath + "/" + filename, newmask)
    # fig = plt.figure()
    # fig.set_size_inches(256/256, 1, forward=False)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)
    
    # #newmask = newmask[:,:,0]

    # plt.imshow(mask, cmap = "gray")
    # ax.imshow(newmask, cmap='gray', alpha = 0.15, interpolation = 'nearest')
    # #plt.savefig("D:/Temp/"+filename, dpi = 2048)
    # plt.show()

run_Modyfying(srcPath, dstPath)