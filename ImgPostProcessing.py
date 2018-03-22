""""
Image PostProcessing Script 

Input : Folder 
Output : Folder 



""""


# In[]

import cv2 
import numpy as np
import matplotlib.pyplot as plt 
import os 

def PostProcessing_Ver1(inputImgPath,inputImgPath2):
    img = cv2.imread(inputImgPath)
    img = np.asarray(img, dtype = "uint8")
    img = img[:,:,0]

    # img2 = cv2.imread(inputImgPath2)
    # img2 = np.asarray(img2, dtype = "uint8")
    # img2 = img2[:,:,0]

    # img_ = img | img2

    ret, bin_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    temp, contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE )
    for i, cnt in enumerate(contours):
        cv2.drawContours(bin_img, contours, i, color = (255,255,255), thickness = -1 )   
    kernel = np.ones((5,5),np.uint8)
    bin_img = cv2.dilate(bin_img,kernel,iterations = 5)
    bin_img = cv2.erode(bin_img,kernel,iterations = 7)
    bin_img = cv2.dilate(bin_img,kernel,iterations = 2)
    print(inputImgPath)
    plt.imshow(bin_img, cmap="gray")
    plt.show()

imgFolder = "D:/[Data]/[Lung_Segmentation]/[PNG]_3_Detected_Mask(256)/DetectedMask_20180319_160133"
imgFolder2 = "D:/[Data]/[Lung_Segmentation]/[PNG]_3_Detected_Mask(256)/DetectedMask_20180319_100708"
imgPath = "D:/[Data]/[Lung_Segmentation]/[PNG]_3_Detected_Mask(256)/DetectedMask_20180319_100708/Mask_Img_20180308_110546.png"
for file in os.listdir(imgFolder):
    imgPath = imgFolder + "/" + file
    imgPath_2 = imgFolder2 + "/" + file
    PostProcessing_Ver1(imgPath, imgPath_2)