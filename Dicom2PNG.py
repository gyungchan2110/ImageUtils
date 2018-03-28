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
        print(img.shape)
        img = exposure.equalize_hist(img)
        img = img[0,:,:]
        print(np.amax(img), np.amin(img))
        if(imshape is not None):
            img = transform.resize(img, imshape)
        #
        #img = windowing(img, 8000, 3000)
        img = img*255
        img = np.asarray(img, dtype = "int16")
        
        img =  np.expand_dims(img, -1)
        print(img.shape)
        cv2.imwrite(dst_Path +"/"+ filename[:-4]+".png", img)

        _ = cv2.imread(dst_Path +"/"+ filename[:-4]+".png")
        print(_.shape)
        break

folders = ["Normal","Abnormal"]

src = "D:/[Data]/[Lung_Segmentation]/Dicom"
dst = "D:/[Data]/[Lung_Segmentation]/[PNG]_1_Basic_Data(2k)/Img_20180319"

for folder in folders:

    dicom2png( src + "/" + folder,  dst + "/" + folder, None)
    break
