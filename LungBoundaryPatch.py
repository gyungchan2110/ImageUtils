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
classMaskBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180125_103950_Expand_40pixel/Masks/Mask_Rt Upper CB"
lungMaskBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180324_LungMaskData/Imgs"
dstBase = "D:/Temp/classpatch"
bgBase = "D:/Temp/BG"


folders = ["train", "validation", "test"]
#lowerFolders = ["1_AS", "2_AR", "3_MS", "4_MR", "5_AS+AR", "6_MS_MR"]
srcPath = []
dstPath = []
classMaskPath = []
lungMaskPath = []
bgMaskPath = []

for folder in folders:   

    srcPath.append(srcbase + "/" + folder)
    dstPath.append(dstBase + "/" + folder)
    classMaskPath.append(classMaskBase + "/" + folder)
    lungMaskPath.append(lungMaskBase + "/" + folder)
    bgMaskPath.append(bgBase + "/" + folder)

    
    if(not os.path.isdir(dstBase + "/" + folder)):
        os.mkdir(dstBase + "/" + folder)
    if(not os.path.isdir(classMaskBase + "/" + folder)):
        os.mkdir(classMaskBase + "/" + folder)
    if(not os.path.isdir(lungMaskBase + "/" + folder)):
        os.mkdir(lungMaskBase + "/" + folder)
    if(not os.path.isdir(bgBase + "/" + folder)):
        os.mkdir(bgBase + "/" + folder)

def run_Modyfying(srcPaths, dstPaths):
    
    for i, srcPath in enumerate(srcPaths) : 
        for file in os.listdir(srcPath):
            print(srcPath)
            print(lungMaskPath[i])
            print(classMaskPath[i])
            print(dstPath[i])
            print(file)
            GetLungBoundaryPatches_2(srcPath,lungMaskPath[i],classMaskPath[i], bgMaskPath[i], dstPath[i], file, (32,32))
            break
        break


def GetLungBoundaryPatches_2(ImgPath, FullLungMaskPath, classMaskPath,bgMaskPath, dstPath, filename, size): 
    
    halfsize = size[0] // 2
    #print(halfsize)
    Image = cv2.imread(ImgPath + "/" + filename)
    Image = np.asarray(Image, dtype = "int16")
    Image = Image[:,:,0]
    fulllLung = cv2.imread(FullLungMaskPath+ "/" + filename)
    fulllLung = np.asarray(fulllLung, dtype = "uint8")
    fulllLung = fulllLung[:,:,0]
    classMask = cv2.imread(classMaskPath+ "/" + filename)
    classMask = np.asarray(classMask, dtype = "uint8")
    classMask = classMask[:,:,0]
    classMask = cv2.resize(classMask, Image.shape)
    ret, bin_classMask = cv2.threshold(classMask, 127, 255, cv2.THRESH_BINARY)
    imgShape = Image.shape

    N_y =  imgShape[0] // size[0]
    N_x =  imgShape[1] // size[1]

    patch = np.zeros(size)
    for i in range(N_y):
        for j in range(N_x):
            start_y = halfsize + i * size[0]
            start_x = halfsize + j * size[1]
            #print(start_y, start_x)
            patch[:,:] = Image[start_y - halfsize : start_y + halfsize , start_x - halfsize: start_x + halfsize]

            if(bin_classMask[start_y, start_x] == 255):
                dstFile = dstPath + "/" + filename[:-4] + "_" + str(start_y) + "x" + str(start_x) + ".png"
                cv2.imwrite ( dstFile, patch )
            #else:
                #dstFile = bgMaskPath + "/" + filename[:-4] + "_" + str(start_y) + "x" + str(start_x) + ".png"
           
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