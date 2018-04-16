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

imgBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180327_151800_2Classes_Original"
srcbase = "D:/[Data]/[Lung_Segmentation]/WholeDataSetMask"

#classMaskBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180125_103950_Expand_40pixel/Masks/Mask_Rt Upper CB"
#lungMaskBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180324_LungMaskData/Imgs"
maskdstBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180327_151800_2Classes_Original_LungMask"
cropmaskdstBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180327_151800_2Classes_Original_LungMask_Cropped"
maskcropmaskdstBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180327_151800_2Classes_Original_LungMask_Cropped_Mask"
dstBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180327_151800_2Classes_Original_Img2Mask_3Channel"


# Img_20180130_175720

# Img_20180130_162001
# Img_20180130_163512
# Img_20180130_164744


lowerFolders = ["Normal", "Abnormal"]
#lowerFolders = ["1_AS", "2_AR", "3_MS", "4_MR", "5_AS+AR", "6_MS_MR"]
srcPaths = []
imgPaths = []
maskdstPaths = []
cropImgsdstPaths = []
maskcropImgsdstPaths = []
dstPath = []



for folder in folders:   

    if(not os.path.isdir(maskdstBase + "/" + folder)):
        os.mkdir(maskdstBase + "/" + folder)

    if(not os.path.isdir(cropmaskdstBase + "/" + folder)):
        os.mkdir(cropmaskdstBase + "/" + folder)

    if(not os.path.isdir(maskcropmaskdstBase + "/" + folder)):
        os.mkdir(maskcropmaskdstBase + "/" + folder)
    if(not os.path.isdir(dstBase + "/" + folder)):
        os.mkdir(dstBase + "/" + folder)

    for lowerFolder in lowerFolders:
        if(not os.path.isdir(maskdstBase + "/" + folder + "/" + lowerFolder)):
            os.mkdir(maskdstBase + "/" + folder + "/" + lowerFolder)
        if(not os.path.isdir(cropmaskdstBase + "/" + folder + "/" + lowerFolder)):
            os.mkdir(cropmaskdstBase + "/" + folder + "/" + lowerFolder)
        if(not os.path.isdir(maskcropmaskdstBase + "/" + folder + "/" + lowerFolder)):
            os.mkdir(maskcropmaskdstBase + "/" + folder + "/" + lowerFolder)
        if(not os.path.isdir(dstBase + "/" + folder + "/" + lowerFolder)):
            os.mkdir(dstBase + "/" + folder + "/" + lowerFolder)


        maskdstPaths.append(maskdstBase + "/" + folder + "/" + lowerFolder)

        cropImgsdstPaths.append(cropmaskdstBase + "/" + folder + "/" + lowerFolder)
        maskcropImgsdstPaths.append(maskcropmaskdstBase + "/" + folder + "/" + lowerFolder)
        dstPath.append(dstBase + "/" + folder + "/" + lowerFolder)

        srcPaths.append(srcbase + "/" + lowerFolder)
        imgPaths.append(imgBase + "/" + folder + "/" + lowerFolder)


def run_Modyfying():
    
    for i, imgPath in enumerate(imgPaths) : 
        for file in os.listdir(imgPath):
            LungBoundaryCrop(imgPath,srcPaths[i], maskdstPaths[i],cropImgsdstPaths[i],maskcropImgsdstPaths[i], file)
            break
        break

def LungBoundaryEnhancement(imgPath, maskPath, dstPath, filename):

    Img = cv2.imread(imgPath + "/" + filename, 0)
    Mask = cv2.imread(maskPath + "/" + filename, 0)
    Img = cv2.resize(Img, (1024,1024))
    Img = np.asarray(Img)
    Mask = np.asarray(Mask)

    Image = np.stack((Img, Img, Mask), -1)

    cv2.imwrite(dstPath + "/" + filename, Image)


def LungBoundaryCrop(imgPath, srcPath, maskdstPath,cropmaskdstPath, maskcropmaskdstPath, filename): 
    
    
    #shutil.copyfile(srcPath + "/" + filename, maskdstPath + "/" + filename)
    
    maskImg = cv2.imread(maskdstPath + "/" + filename, 0)
    maskImg = np.asarray(maskImg, dtype = np.uint8)

    _, maskImg = cv2.threshold(maskImg, 127, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(maskImg, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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

    print((top_x + bottom_x)/2, (top_y + bottom_y)/2)
    center_shift_x = 512 -  (int)((top_x + bottom_x)/2)
    center_shift_y = 512 -  (int)((top_y + bottom_y)/2)


    # maskCrop = maskImg[top_y:bottom_y, top_x:bottom_x]
    # maskCrop = cv2.resize(maskCrop, (1024,1024))
    # cv2.imwrite(maskcropmaskdstPath + "/" + filename, maskCrop)

    Img = cv2.imread(imgPath + "/" + filename)
    Img = np.asarray(Img)
    Img = cv2.resize(Img, (1024,1024))
    # ImgCrop = Img[top_y*2:bottom_y*2, top_x*2:bottom_x*2, :]
    # ImgCrop = cv2.resize(ImgCrop, (1024,1024))
    # cv2.imwrite(cropmaskdstPath + "/" + filename, ImgCrop)
    # print(imgPath + "/" + filename)
    Img_Shifted = np.zeros(Img.shape)
    #Img_Shifted = Img_Shifted * 255
    Img_Shifted[:1024+center_shift_y, center_shift_x:] = Img[-center_shift_y:, :1024-center_shift_x]
    cv2.imwrite("D:/Temp/Shifted.png", Img_Shifted)
    cv2.imwrite("D:/Temp/Original.png", Img)
run_Modyfying()