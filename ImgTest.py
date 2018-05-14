# In[]
import cv2 
import numpy as np   
import os 
from operator import eq
import random
import matplotlib.pyplot as plt 
from skimage import io
import shutil


listBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180201_091700_2Classes"
srcBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Imgs_OriginalData_2k2k_2Classes"
dstBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180327_151800_2Classes_Original"

levels = ["train", "test", "validation"]

classes = ["Normal", "Abnormal"]


for level in levels:
    for classe in classes:
        filelistPath = listBase + "/" + level + "/" + classe
        for file in os.listdir(filelistPath):     
            shutil.copy2(srcBase + "/" + classe + "/" + file, dstBase + "/" + level + "/" + classe + "/" + file)




# In[]

import cv2 
import matplotlib.pyplot as plt 
import csv 
import SimpleITK as sitk
import os
from skimage import transform,exposure,io
import numpy as np

metadata = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/BasicData_MetaFile_Ex.csv"

maskPath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks_OriginalData_OriginalSize"
masks = ["Aortic Knob", "Lt Lower CB", "Pulmonary Conus", "Rt Lower CB", "Rt Upper CB", "DAO" , "Carina" , "LAA", "Axis", "Thorax(x)", "Thorax(y)", "Diaphragm", "Rib(9)", "Rib(10)]

f = open(metadata, 'r')
f_reader = csv.reader(f)


for row in f_reader : 
    translated = row[2]
    orignal = row[0] + "/" + row[1] + "/" + row[1] + ".dcm"

    if( row[0].fine("Abnormal") >= 0) : 
        maskbase = maskPath + "/Abnormal/Total"
    else
        maskbase = maskPath + "/Normal"



    img = sitk.ReadImage(orignal)
    img = sitk.GetArrayFromImage(img).astype("int16") 
    print(img.shape)
    img = exposure.equalize_hist(img)
    img = img[0,:,:]
    #print(np.amax(img), np.amin(img))
    # if(imshape is not None):
    #     img = transform.resize(img, imshape)
    #
    #img = windowing(img, 8000, 3000)
    img = img*255
    original = np.asarray(img, dtype = "int16")
    
    original =  np.expand_dims(original, -1)
    lung_mask = "D:/[Data]/[Lung_Segmentation]/WholeDataSetMask/Whole/" + translated + ".png"
    mask = cv2.imread(lung_mask, 0)

    #original = np.asarray(original, dtype = "uint8")
    twok = np.asarray(twok, dtype = "uint8")
    #m#ask = np.asarray(mask, dtype = "uint8")
    print(twok.shape, original.shape)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

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


    txr = float(top_x) / 1024.
    tyr = float(top_y) / 1024.
    bxr = float(bottom_x) / 1024.
    byr = float(bottom_y) / 1024.

    shape = original.shape  

    otx = int(shape[1] * txr)
    oty = int(shape[0] * tyr)
    obx = int(shape[1] * bxr)
    oby = int(shape[0] * byr)

    y_line = (otx + obx) // 2

    original = np.stack([original[:,:,0],original[:,:,0],original[:,:,0]], axis = -1)
    cv2.rectangle(original, (otx, oty), (obx, oby), color = (0,0,255), thickness = 1)
    cv2.rectangle(original, (y_line - 5, oty), (y_line + 5, oby), color = (255,255,0), thickness = 1)
    cv2.rectangle(original, (y_line//2 - 5, oty), (y_line//2 + 5, oby), color = (255,255,255), thickness = 1)

    for mask in masks : 
        path = maskbase + "/" + translated + ".png"
        mask = cv2.imread(path, 0)
        mask = np.asarray(mask, dtype = "uint8")
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        cv2.polylines(original, contours, isClosed = True, color = (255,255,255), thickness = 1)
    cv2.imwrite("D:/TestTemp/" + translated + ".png", original)
    print(translated)
    break

# In[]
import csv 

csvFile = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks_OriginalData_2k2k/Normal/Thorax(x)/xLevel.csv"
f = open(csvFile, 'r', encoding = "utf-8", newline='')
csvReader = csv.reader(f)

for row in csvReader:
    print(type(row))

f.close()



# In[]

import numpy as np 


Pts = []

Pts.append((37.355891, 127.951228)) # 원주 
Pts.append((37.086152, 127.039604)) # 오산 
Pts.append((37.337880, 126.811520)) # 안산
Pts.append((37.452699, 126.909778)) # 금천구
Pts.append((37.461455, 126.958009)) # 기숙사
Pts.append((37.461455, 126.958009)) # 기숙사2
Pts.append((37.512161, 126.918398)) # 영등포
Pts.append((40.101099, 360-88.230752))

longitude = []
latitude = []

for pt in Pts : 
    longitude.append(pt[1])
    latitude.append(pt[0])

centerPoint = (np.asarray(latitude).mean(), np.asarray(longitude).mean())
print(centerPoint)