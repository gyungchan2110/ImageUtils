""""
Mask Refining Script 

Input : Deep Learning Detected Mask Folder 
Input : Correct Masks made by Radiologist 
Output : Refined Mask Folder 




"""



# In[]
import cv2 
import matplotlib.pyplot as plt
from skimage import io, exposure
import numpy as np
from scipy.ndimage.morphology import binary_fill_holes
import os 
import operator

def MakeLungMask(labelPath, originalMaskPath, dstFile):
    labels = []
    kernel = np.ones((5,5),np.uint8)

    for filename in os.listdir(labelPath):
        if(operator.eq(filename[-4:], "tiff")):
            labels.append(labelPath + "/" + filename)

    label = np.zeros((2048,2048))

    for path in labels : 
        img = io.imread(path, as_grey=True)
        ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        img = np.asarray(img, dtype = "uint8")
        label = label + img

    # Step 1. Label Integration(Label map)

    label = np.clip(label, 0, 255)
    label = np.asarray(label, dtype = "uint8")

    # plt.imshow(label, cmap="gray")
    # plt.show()

    # io.imsave("D:/Temp/Step_1.png", label)
    ########################################################################

    # Step 2. Read Original Mask and Binarization 

    mask = cv2.imread(originalMaskPath)
    mask = cv2.resize(mask, (2048,2048))
    ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = np.asarray(mask, dtype = "uint8")
    mask = mask[:,:,0]
    # plt.imshow(mask, cmap="gray")
    # plt.show()

    # io.imsave("D:/Temp/Step_2.png", mask)
    ################################################################

    # Step 3. Overlay original Mask and label map 
    # Black BG and White Lung with white label lines 

    # mask = mask[:,:,0] | label
    # plt.imshow(mask, cmap="gray")
    # plt.show()
    #################################################
    
    # Step 4. Fill up the region between label lines and lung with white 
    # 


    # mask = mask /255
    # mask = binary_fill_holes(mask)
    # mask = mask * 255










    
    # plt.imshow(mask, cmap="gray")
    # plt.show()



    #################################################################

    # Step 5. 
    label_cp = cv2.dilate(label,kernel,iterations = 1)

    label_inv = 255 - label_cp
    mask = mask & label_inv
    mask = np.asarray(mask, dtype = "uint8")
    # plt.imshow(mask, cmap="gray")
    # plt.show()

    # io.imsave("D:/Temp/Step_5.png", mask)
    ##########################################################################

    # Step 6. 


    mask = cv2.erode(mask,kernel,iterations = 1)
    temp, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_TC89_L1 )
    areas = []
    for cnt in contours:
        areas.append(cv2.contourArea(cnt))

        #del(areas[len(areas)-1])
    areas = np.array(areas)
   # print(areas)
    maxindex = 0
    secondmaxindex = 0

    max = 0
    secondmax = 0
    for i, area in enumerate(areas):
        if(area > max):
            secondmax = max
            secondmaxindex = maxindex

            max = area
            maxindex = i

        if area < max and area > secondmax:
            secondmax = area
            secondmaxindex = i

    for i, cnt in enumerate(contours):
        if (i is not maxindex) and (i is not secondmaxindex) :
            cv2.drawContours(mask, contours, i, color = (0,0,0), thickness = -1 )

    
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 3)
    #io.imsave("D:/Temp/Step_6.png", mask)
    #mask = np.resize(mask, (1024,1024))
#############################################################################################################

    # Step 7. Overlay original Mask and label map 
    # Black BG and White Lung with white label lines 

    mask = mask | label
    # plt.imshow(mask, cmap="gray")
    # plt.show()

    # io.imsave("D:/Temp/Step_7.png", mask)

############################################################################################################

    # mask_inv = 255 - mask
    # plt.imshow(mask_inv, cmap="gray")
    # plt.show()

    # io.imsave("D:/Temp/Step_8.png", mask_inv)

############################################################################################################
    temp, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL , cv2.CHAIN_APPROX_SIMPLE  )
    mask_inv = np.zeros(mask.shape)
    print(len(contours))
    for i, cnt in enumerate(contours):
        cv2.drawContours(mask, contours, i, color = (255,255,255), thickness = -1 )    
        #rect = cv2.boundingRect(cnt)
        #mask_inv[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = 255 - mask[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        #print(rect)
        #mask = cv2.rectangle(mask, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255,255,255), thickness = 1)
    mask = np.asarray(mask, dtype='uint8')
    # plt.imshow(mask, cmap="gray")
    # plt.show()

    

############################################################################################################
    mask = cv2.dilate(mask,kernel,iterations = 3)
    
    #io.imsave("D:/Temp/Step_9.png", mask)
    mask = mask /255
    mask = binary_fill_holes(mask)
    mask = mask *255
    mask = np.asarray(mask, dtype = "uint8")
    mask = cv2.erode(mask,kernel,iterations = 5)
    mask = cv2.dilate(mask,kernel,iterations = 2)
    # plt.imshow(mask, cmap="gray")
    # plt.show()
    io.imsave(dstFile, mask)

    print(dstFile)



labels = ["D:/[Data]/[Lung_Segmentation]/overlay_레이블/Label_20180315/Normal", "D:/[Data]/[Lung_Segmentation]/overlay_레이블/Label_20180315/Abnormal"]
maskPath = ["D:/[Data]/[Lung_Segmentation]/[PNG]_3_Detected_Mask(256)/DetectedMask_20180308_113204", "D:/[Data]/[Lung_Segmentation]/[PNG]_3_Detected_Mask(256)/DetectedMask_20180308_112943_Abnormal"]
dst = ["D:/[Data]/[Lung_Segmentation]/GeneratedMask/Mask_20180315/Normal", "D:/[Data]/[Lung_Segmentation]/GeneratedMask/Mask_20180315/Abnormal"]


for i, label in enumerate(labels): 
    for folder in os.listdir(label) : 
        if(not os.path.isdir(label + "/" + folder)):
            continue 
        print(folder)
        MakeLungMask(label + "/" + folder, maskPath[i] + "/Mask" + folder + ".png", dst[i] + "/" + folder + ".png")
        #break
    #break 