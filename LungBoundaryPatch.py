# In[]
import cv2 
import numpy as np   
import os 
from operator import eq
import random
import matplotlib.pyplot as plt 
from skimage import io

os.environ["CUDA_VISIBLE_DEVICES"]="0"

srcbase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180125_103950_Expand_20pixel/Imgs"
classMaskBase_RUCB = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180125_103950_Expand_40pixel/Masks/Mask_Rt Upper CB"
classMaskBase_RLCB = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180125_103950_Expand_40pixel/Masks/Mask_Rt Lower CB"
lungMaskBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180324_LungMaskData/Imgs"
dstBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_LungMaskPatch_3Classes"
bgBase = "D:/Temp/BG"


folders = ["train", "validation", "test"]
#lowerFolders = ["1_AS", "2_AR", "3_MS", "4_MR", "5_AS+AR", "6_MS_MR"]
srcPath = []
dstPath = []
classMaskPath_RUCB = []
classMaskPath_RLCB = []
lungMaskPath = []
bgMaskPath = []

for folder in folders:   

    srcPath.append(srcbase + "/" + folder)
    dstPath.append(dstBase + "/" + folder)
    classMaskPath_RUCB.append(classMaskBase_RUCB + "/" + folder)
    classMaskPath_RLCB.append(classMaskBase_RLCB + "/" + folder)
    lungMaskPath.append(lungMaskBase + "/" + folder)
    bgMaskPath.append(bgBase + "/" + folder)

    
    if(not os.path.isdir(dstBase + "/" + folder)):
        os.mkdir(dstBase + "/" + folder)
    if(not os.path.isdir(classMaskBase_RUCB + "/" + folder)):
        os.mkdir(classMaskBase_RUCB + "/" + folder)
    if(not os.path.isdir(classMaskBase_RLCB + "/" + folder)):
        os.mkdir(classMaskBase_RLCB + "/" + folder)
    if(not os.path.isdir(lungMaskBase + "/" + folder)):
        os.mkdir(lungMaskBase + "/" + folder)
    if(not os.path.isdir(bgBase + "/" + folder)):
        os.mkdir(bgBase + "/" + folder)

def run_Modyfying(srcPaths, dstPaths):
    
    for i, srcPath in enumerate(srcPaths) : 
        for file in os.listdir(srcPath):
            # print(srcPath)
            # print(lungMaskPath[i])
            # print(classMaskPath[i])
            # print(dstPath[i])
            # print(file)
            GetLungBoundaryPatches_2(srcPath,lungMaskPath[i],classMaskPath_RUCB[i],classMaskPath_RLCB[i], dstPath[i], file, (64,64))
        #     break
        # break


def GetLungBoundaryPatches_2(ImgPath, FullLungMaskPath, classMaskPath_RUCB, classMaskPath_RLCB, dstPath, filename, size): 
    
    halfsize = size[0] // 2
    #print(halfsize)
    Image = cv2.imread(ImgPath + "/" + filename)
    Image = np.asarray(Image, dtype = "int16")
    Image = Image[:,:,0]
    fulllLung = cv2.imread(FullLungMaskPath+ "/" + filename)
    fulllLung = np.asarray(fulllLung, dtype = "uint8")
    fulllLung = fulllLung[:,:,0]
    ret, bin_fulllung = cv2.threshold(fulllLung, 127, 255, cv2.THRESH_BINARY)
    temp, contour_fullLung, hierarchy = cv2.findContours(bin_fulllung, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )

    classMask_RUCB = cv2.imread(classMaskPath_RUCB+ "/" + filename)
    classMask_RUCB = np.asarray(classMask_RUCB, dtype = "uint8")
    classMask_RUCB = classMask_RUCB[:,:,0]
    classMask_RUCB = cv2.resize(classMask_RUCB, Image.shape)
    ret, bin_classMask_RUCB = cv2.threshold(classMask_RUCB, 127, 255, cv2.THRESH_BINARY)
    #temp, contour_class_RUCB, hierarchy = cv2.findContours(bin_classMask_RUCB, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )

    classMask_RLCB = cv2.imread(classMaskPath_RLCB+ "/" + filename)
    classMask_RLCB = np.asarray(classMask_RLCB, dtype = "uint8")
    classMask_RLCB = classMask_RLCB[:,:,0]
    classMask_RLCB = cv2.resize(classMask_RLCB, Image.shape)
    ret, bin_classMask_RLCB = cv2.threshold(classMask_RLCB, 127, 255, cv2.THRESH_BINARY)
    #temp, contour_class_RLCB, hierarchy = cv2.findContours(bin_classMask_RLCB, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )

    imgShape = Image.shape

    class_RUCB_cnt = 0
    class_RLCB_cnt = 0 
    class_BG_cnt = 0 

    total_CNT = 30

    classname = ""
    for cnt in contour_fullLung:
        random.shuffle(cnt)
        for pt in cnt:
            #print(pt)
            patch = np.zeros(size)
            step_x = int(size[0] / 2)
            step_y = int(size[1] / 2)
            
            patch[:,:] = Image[pt[0, 0] - step_x : pt[0, 0]+step_x, pt[0, 1] - step_y : pt[0, 1]+step_y]
            if(bin_classMask_RUCB[pt[0, 0], pt[0, 1]] == 255):
                if(class_RUCB_cnt < total_CNT ):
                    classname = "RUCB"
                    class_RUCB_cnt += 1

            elif(bin_classMask_RLCB[pt[0, 0], pt[0, 1]] == 255):
                if(class_RLCB_cnt < total_CNT ):
                    classname = "RLCB"      
                    class_RLCB_cnt += 1   
            else:
                if(class_BG_cnt < total_CNT ):
                    classname = "BG"
                    class_BG_cnt += 1
            
            if(not os.path.isdir(dstPath + "/" + classname )):
                os.mkdir(dstPath + "/" + classname)
            dstFile = dstPath + "/" + classname + "/" + filename[:-4] + "_" + str(pt[0, 0]) + "x" + str(pt[0, 1]) + ".png" 
            cv2.imwrite ( dstFile, patch )
    print(filename)
            
           
def GetLungBoundaryPatches(ImgPath, FullLungMaskPath, classMaskPath,bgMaskPath, dstPath, filename, size): 
    Image = cv2.imread(ImgPath + "/" + filename)
    Image = np.asarray(Image, dtype = "int16")
    Image = Image[:,:,0]
    fulllLung = cv2.imread(FullLungMaskPath+ "/" + filename)
    fulllLung = np.asarray(fulllLung, dtype = "uint8")
    fulllLung = fulllLung[:,:,0]
    classMask = cv2.imread(classMaskPath+ "/" + filename)
    classMask = np.asarray(classMask, dtype = "uint8")
    classMask = classMask[:,:,0]
    classMask = cv2.resize(classMask, fulllLung.shape)
    ret, bin_fulllung = cv2.threshold(fulllLung, 127, 255, cv2.THRESH_BINARY)
    temp, contour_fullLung, hierarchy = cv2.findContours(bin_fulllung, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )
    
    ret, bin_classMask = cv2.threshold(classMask, 127, 255, cv2.THRESH_BINARY)
    temp, contour_class, hierarchy = cv2.findContours(bin_classMask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_NONE )
    
    fig = plt.figure()
    fig.set_size_inches(256/256, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    
    #newmask = newmask[:,:,0]

    plt.imshow(bin_fulllung, cmap = "gray")
    ax.imshow(bin_classMask, cmap='gray', alpha = 0.15, interpolation = 'nearest')
    plt.savefig("D:/Temp/"+filename, dpi = 2048)
    plt.show()


    assert(len(contour_class)==1)

    for cnt in contour_fullLung:
        for pt in cnt:
            #print(pt)
            patch = np.zeros(size)
            step_x = int(size[0] / 2)
            step_y = int(size[1] / 2)
            
            patch[:,:] = Image[pt[0, 0] - step_x : pt[0, 0]+step_x, pt[0, 1] - step_y : pt[0, 1]+step_y]
            if(bin_classMask[pt[0, 0], pt[0, 1]] == 255):
                dstFile = dstPath + "/" + filename[:-4] + "_" + str(pt[0, 0]) + "x" + str(pt[0, 1]) + ".png"
           
            else:
                dstFile = bgMaskPath + "/" + filename[:-4] + "_" + str(pt[0, 0]) + "x" + str(pt[0, 1]) + ".png"
           
            cv2.imwrite ( dstFile, patch )
        #break
        #break   

run_Modyfying(srcPath, dstPath)