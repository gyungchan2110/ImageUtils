# In[]

import SimpleITK as sitk
import os
from skimage import transform,exposure
import numpy as np
import cv2

os.environ["CUDA_VISIBLE_DEVICES"]="0"

def dicom2png(src_Path, dst_Path, imshape = (2048,2048)):

    if(not os.path.isdir(src_Path)):
        return 
    if(not os.path.isdir(dst_Path)):
        return 

    for filename in os.listdir(src_Path): 
        filepath = src_Path + "/" + filename
        if(os.path.isdir(filepath)):
            continue 
        print(filepath)
        img = sitk.ReadImage(filepath)
        img = sitk.GetArrayFromImage(img).astype("int16") 
        img = exposure.equalize_hist(img)
        img = img[0,:,:]
        img = transform.resize(img, imshape)
        #img =  np.expand_dims(img, -1)
        io.imsave(dst_Path +"/"+ filename[:-4]+".png", img)

folders = ["Normal","Abnormal"]

src = "D:/[Data]/[Lung_Segmentation]/Dicom"
dst = "D:/[Data]/[Lung_Segmentation]/PNG"

for folder in folders:

    dicom2png( src + "/" + folder,  dst + "/" + folder)
