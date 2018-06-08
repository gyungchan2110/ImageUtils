####### 
## 1. Dicom to PNG

# In[]
import SimpleITK as sitk
import os
from skimage import transform,exposure,io
import numpy as np
import cv2

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def windowing(img, center, width):
    low = center - width/2
    high = center + width/2
    img = (img - low)/width
    img[img<0.]=0.
    img[img>1.]=1.
    return img


def dicom2png(src_Path, dst_Path, imshape):
    #print(src_Path)
    if(not os.path.isdir(src_Path)):
        return 
    if(not os.path.isdir(dst_Path)):
        return 

    for filename in os.listdir(src_Path): 
        if(filename[-4:] != ".dcm"):
            continue  

        filepath = src_Path + "/" + filename
        if(os.path.isdir(filepath)):
            continue 

        print(filepath)
        img = sitk.ReadImage(filepath)
        img = sitk.GetArrayFromImage(img).astype("int16") 
        shape = img.shape
        img = exposure.equalize_hist(img)
        img = img[0,:,:]
        
        if(imshape is not None):
            img = transform.resize(img, imshape)
        #
        #img = windowing(img, 8000, 3000)
        img = img*255
        img = np.asarray(img, dtype = "int16")
        
        img =  np.expand_dims(img, -1)
        
        cv2.imwrite(dst_Path +"/"+ filename[:-4]+"_" + str(shape[1]) + "_" + str(shape[2]) +".png", img)


folder = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_0_Original"
dst = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_1_BasicData"

imshape = (2048,2048)

for lower in os.listdir(folder) :

    if (not os.path.isdir(folder + "/" + lower)):
        continue 

    if( not os.path.isdir(dst + "/" + lower)):
        os.mkdir(dst+ "/" + lower) 


    dicom2png( folder + "/" + lower,  dst+ "/" + lower, imshape)





###################################################################################
## 2. Lung Mask 
# In[]

##################################################################################



#################################################################################

## 3. Lung Mask Crop 
# In[]
import cv2 
import numpy as np   
import os 
from operator import eq
import random
import matplotlib.pyplot as plt 
from skimage import io
import shutil
import csv

os.environ["CUDA_VISIBLE_DEVICES"]="0"



folder = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_1_BasicData/Imgs"
dst = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_2_ImgCropped"
LungMaskPath = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_1_BasicData/Masks"
csvfile = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_2_ImgCropped/metadata.csv"

for lower in os.listdir(folder) : 

    
    if (not os.path.isdir(folder + "/" + lower)):
        continue 

    if( not os.path.isdir(dst + "/" + lower)):
        os.mkdir(dst+ "/" + lower) 

    for file in os.listdir(folder + "/" + lower) : 
        print(file)
        LungMask = cv2.imread(LungMaskPath +  "/" + lower + "/" + file, 0)
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
        
        
        
        Img = cv2.imread(folder +  "/" + lower + "/" + file, 0)
        
        ImgCrop = Img[top_y*2:bottom_y*2, top_x*2:bottom_x*2]
        ImgCrop = cv2.resize(ImgCrop, (1024,1024))
        cv2.imwrite(dst+ "/" + lower +  "/" + file, ImgCrop)

        f = open(csvfile, 'a', encoding = "utf-8", newline='')
        f_writer = csv.writer(f)

        strline = []
        strline.append(file)
        strline.append(str(top_y*2))
        strline.append(str(bottom_y*2))
        strline.append(str(top_x*2))
        strline.append(str(bottom_x*2))

        f_writer.writerow(strline)
        f.close()
####################################################################################################################


###################################################################################
## 4. Line Detection  
# In[]

##################################################################################

###################################################################################
## 5. Mask recur 

# In[]
import os     

os.environ["CUDA_VISIBLE_DEVICES"]="0"

import cv2 
import csv
import numpy as np
from skimage.morphology import skeletonize

Classlist = ["Aortic Knob", "Lt Lower CB", "Pulmonary Conus", "Rt Lower CB", "Rt Upper CB", "DAO" , "Carina" , "LAA"]

filePath = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_3_LineMask"

dstPath = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_4_LineMask_Recur"
csvfile = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_2_ImgCropped/metadata.csv"

for clss in Classlist : 

    clsfilepath = filePath + "/" + clss  

    if(not os.path.isdir(dstPath+"/" + clss) ):
        os.mkdir(dstPath+"/" + clss)

    for file in os.listdir(clsfilepath):
        mask = cv2.imread(clsfilepath + "/" + file, 0)
        mask = np.asarray(mask)
        
        f = open(csvfile, 'r', encoding = "utf-8", newline='')
        f_reader = csv.reader(f)

        xtop = yleft = xbottom = yright = 0

        for row in f_reader:
            if (row[0] == file):
                xtop = int(row[1])
                yleft = int(row[3])
                xbottom = int(row[2])
                yright = int(row[4])
                break
        
        f.close()

        if xtop == 0 or yleft == 0 or xbottom == 0 or yright == 0:
            continue
        #print(file)
        #print(xbottom, xtop, yright, yleft)
        mask_ = cv2.resize(mask, (yright- yleft, xbottom - xtop))

        mask_ = np.asarray(mask_)
        
        #print(mask_.shape)
        mask_2k = np.zeros((2048,2048))

        mask_2k[xtop:xbottom, yleft:yright] = mask_

        units = file.split(".")[0].split("_")
        oy = int(units[1])
        ox = int(units[2])

        mask_orisize = cv2.resize(mask_2k, (ox, oy))
        _, mask_orisize = cv2.threshold(mask_orisize, 127, 255, cv2.THRESH_BINARY)

        element = np.ones((5,5))
        mask_orisize = cv2.erode(mask_orisize,element)
        # eroded_mask = eroded_mask // 255
        # skeleton = skeletonize(eroded_mask)
        # skeleton = skeleton * 255 
        # skeleton = np.asarray(skeleton, dtype = "uint8")
        # element = np.ones((5,5))
        # mask_orisize = cv2.dilate(skeleton,element)
        # mask_orisize = cv2.dilate(mask_orisize,element)

        cv2.imwrite(dstPath+"/" + clss + "/" + units[0] + ".png", mask_orisize)
        print(file)
##################################################################################








###################################################################################
## 6. Overlay 

# In[]

import cv2 
import numpy as np   
import os 
from operator import eq
import random
import matplotlib.pyplot as plt 
import SimpleITK as sitk
from skimage import transform,exposure,io
import csv

# BasePath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base_Expand_40pixel_Cropped_Detected"
# ImgPath = BasePath + '/Imgs/test'
# MaskPaths = BasePath + "/Masks"
# Dstpath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Temp_OverLay"
Classlist = ["Aortic Knob", "Carina"]

os.environ["CUDA_VISIBLE_DEVICES"]= "0"

csvfile = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_2_ImgCropped/metadata.csv"

folder = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_0_Original"
dst = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_5_Overlayed"
MaskPaths = "D:/[Data]/[Cardiomegaly]/4_ChestPA_ToLabel/[]_4_LineMask_Recur"
imshape = (2048,2048)

for lower in os.listdir(folder) :

    if (not os.path.isdir(folder + "/" + lower)):
        continue 

    if( not os.path.isdir(dst + "/" + lower)):
        os.mkdir(dst+ "/" + lower) 

    for filename in os.listdir(folder + "/" + lower): 
        if(filename[-4:] != ".dcm"):
            continue  

        filepath = folder + "/" + lower + "/" + filename
        if(os.path.isdir(filepath)):
            continue 

        print(filepath)
        img = sitk.ReadImage(filepath)
        img = sitk.GetArrayFromImage(img).astype("int16") 
        shape = img.shape
        img = exposure.equalize_hist(img)
        img = img[0,:,:]


        #fig = plt.figure()
        # fig.set_size_inches(256/256, 1, forward=False)
        # ax = plt.Axes(fig, [0., 0., 1., 1.])
        # ax.set_axis_off()
        # fig.add_axes(ax)
        #img = img[:,:,0]

        masks = np.zeros(img.shape, dtype = "uint8")
        for clss in Classlist : 
            Maskpath = MaskPaths + "/" + clss
            if(not os.path.isfile(Maskpath + "/" + filename[:-4] + ".png")):
                continue
            mask = cv2.imread(Maskpath + "/" + filename[:-4] + ".png")
            mask = np.asarray(mask, dtype = "uint8")
            if(masks.shape != mask.shape[:-1]):
                continue
            # element = np.ones((3,3))
            # mask = cv2.erode(mask,element)
            mask = mask // 255
            masks = np.bitwise_or(masks, mask[:,:,0])  

        linewidth = 6 


        f = open(csvfile, 'r', encoding = "utf-8", newline='')
        f_reader = csv.reader(f)

        xtop = yleft = xbottom = yright = 0
        file = filename[:-4] + "_" + str(shape[1]) + "_" + str(shape[2]) + ".png"
        for row in f_reader:
            if (row[0] == file):
                xtop = int(row[1])
                yleft = int(row[3])
                xbottom = int(row[2])
                yright = int(row[4])
                break
        
        f.close()

        if xtop == 0 or yleft == 0 or xbottom == 0 or yright == 0:
            continue

        xtop = xtop * shape[1] // 2048
        xbottom = xbottom * shape[1] // 2048

        yleft = yleft * shape[2] // 2048
        yright = yright * shape[2] // 2048

        
        center = (yleft + yright) //2
        axis_centor = (yleft*3 + yright)//4

        thorax_x = np.zeros(img.shape, dtype = "uint8")
        thorax_x = cv2.rectangle(thorax_x, (center - linewidth //2 ,xtop), (center + linewidth //2 ,xbottom), color = (255,255,255), thickness = -1)
        thorax_x = thorax_x // 255
        masks = np.bitwise_or(masks, thorax_x)  


        axis = np.zeros(img.shape, dtype = "uint8")
        axis = cv2.rectangle(axis, (axis_centor - linewidth //2 ,xtop), (axis_centor + linewidth //2 ,xbottom), color = (255,255,255), thickness = -1)
        axis = axis // 255
        masks = np.bitwise_or(masks, axis)  

        masks = masks * 255

        bbox = np.zeros(img.shape, dtype = "uint8")
        cv2.rectangle(masks, (yleft, xtop), (yright, xbottom), color = (255,255,255), thickness = 2)

        #masks = masks * 255   
        # 

        masks = masks // 255
        alpha = 0.2
        molded =  masks * alpha + img * (1 - alpha)
        # print(masks.shape)
        # print(img.shape)
       # print(molded.shape)
        molded = molded * 255
        molded = np.asarray(molded, dtype = "uint8")
        cv2.imwrite(dst+ "/" + lower + "/"+filename[:-4] +".png", molded)

        # masks = masks * 255
        # plt.imshow(masks, cmap='gray')
        # plt.show()
        # # ax.imshow(masks, cmap='gray', alpha = 0.15, interpolation = 'nearest')
        # #plt.savefig(dst+ "/" + lower + "/"+filename[:-4] +".png", dpi = 2048)
        # plt.show()









##################################################################################
