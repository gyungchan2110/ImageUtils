
# In[]

import cv2
import numpy as np
from skimage.morphology import skeletonize
import math
import os 
import random


filepath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base_Expand_20pixel_Cropped/Masks/DAO/test"
detected = "R:/Segmentation/MaskRCNN_LOGS/logs/Test_20180430_160039/Mask"


filelist = os.listdir(filepath) 
filename = random.choice(filelist)



#filename = "Img_20180131_145652.png"
print( filename)
filepath = filepath + "/" + filename

detected = detected + "/" + filename


detected_img = cv2.imread(detected, 0)
detected_img = np.asarray(detected_img)
detected_img = cv2.resize(detected_img, (1024,1024))
# size = np.size(img)
# skel = np.zeros(img.shape,np.uint8)
 
ret,detected_img = cv2.threshold(detected_img,127,255,0)
element = np.ones((5,5))
# done = False
# print(element)

eroded = cv2.erode(detected_img,element)
eroded = cv2.erode(eroded,element)
eroded = cv2.erode(eroded,element)
eroded = cv2.erode(eroded,element)
eroded = eroded // 255 
skeleton_Detected = skeletonize(eroded)
# skeleton_Detected = skeleton_Detected * 255
# cv2.imwrite("D:/Temp/Temp.png.png",skeleton)

Orig_img = cv2.imread(filepath, 0)
Orig_img = np.asarray(Orig_img)
Orig_img = cv2.resize(Orig_img, (1024,1024))
# size = np.size(img)
# skel = np.zeros(img.shape,np.uint8)
 
ret,Orig_img = cv2.threshold(Orig_img,127,255,0)
element = np.ones((3,3))
# done = False
# print(element)

eroded = cv2.erode(Orig_img,element)
#eroded = cv2.erode(eroded,element)
eroded = eroded // 255
skeleton2 = skeletonize(eroded)

sk = skeleton_Detected + skeleton2
sk[sk > 1] = 1
sk = sk * 255
####################################################################
sk = np.asarray(sk, dtype = "uint8")
out = cv2.distanceTransform(sk,cv2.DIST_L2, 5, cv2.CV_8U)
print(out.max())
out = out * 255

skeleton_Detected = skeleton_Detected * 255
arrays = np.where(skeleton_Detected>1)

skeleton2 = skeleton2 * 255
arrays_2 = np.where(skeleton2>1)

print(len(arrays[0]), len(arrays_2[0]))




arrays = np.where(skeleton_Detected>1)
arrays = np.asarray(arrays)


arrays_2 = np.where(skeleton2>1)
arrays_2 = np.asarray(arrays_2)


length_1 = len(skeleton_Detected[skeleton_Detected>1])
length_2 = len(skeleton2[skeleton2>1])
length = min(length_1, length_2)

start = 0
if(length_1 < length_2):
    arr = arrays
    length = length_1

    diss = []
    for i in range(length_2):
        pt1 = (arr[1][0], arr[0][0])
        pt2 = (arrays_2[1][0 + i], arrays_2[0][0 + i])
        dis = math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0])  +   (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        diss.append(dis)
    diss = np.asarray(diss)
    start = diss.argmin()

    distance = []
    for i in range(length): 
        
        if(start + i >= length_2):
            break

        pt1 = (arr[1][0 + i], arr[0][0 + i])
        pt2 = (arrays_2[1][start + i], arrays_2[0][start + i])
        dis = math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0])  +   (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        distance.append(dis)
    distance = np.asarray(distance)
    print("Method 1 : ", start)
    print(distance.min(), distance.max(), distance.mean(),distance.std())


else:
    arr = arrays_2
    length = length_2

    diss = []
    for i in range(length_2):
        pt1 = (arr[1][0], arr[0][0])
        pt2 = (arrays[1][0 + i], arrays[0][0 + i])
        dis = math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0])  +   (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        diss.append(dis)
    diss = np.asarray(diss)
    start = diss.argmin()

    distance = []
    for i in range(length): 
        
        if(start + i >= length_1):
            break

        pt1 = (arr[1][0 + i], arr[0][0 + i])
        pt2 = (arrays[1][start + i], arrays[0][start + i])
        dis = math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0])  +   (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
        distance.append(dis)
    distance = np.asarray(distance)
    print("Method 1 : ")
    print(distance.min(), distance.max(), distance.mean(), distance.std())

dif = np.array([length_1 - length, length_2 - length], dtype = "int")
start = dif // 2
end = start + length

#print(start, end)

distance = []
for i in range(length): 
    


    pt1 = (arrays[1][start[0] + i], arrays[0][start[0] + i])
    pt2 = (arrays_2[1][start[1] + i], arrays_2[0][start[1] + i])
    dis = math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0])  +   (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    distance.append(dis)
    # if i % 10 == 0:
    #     #pass
    #     cv2.line(skeleton_, pt1, pt2, thickness = 1, color = (255,255,255))
#print(distance)
distance = np.asarray(distance)
print("Method 2 : ")
print(distance.min(), distance.max(), distance.mean(), distance.std())


cv2.imwrite("D:/Temp/Temp.png2.png",sk)


# In[]

from skimage.morphology import skeletonize
import cv2
import numpy as np
import math


filepath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base_Expand_20pixel/Masks/Aortic Knob/train/Img_20180130_160333.png"
filepath_2 = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base_Expand_20pixel/Masks/Aortic Knob/train/Img_20180130_160447.png"

Mask = cv2.imread(filepath, 0)
ret, bin_mask = cv2.threshold(Mask, 127, 255, cv2.THRESH_BINARY)
bin_mask = bin_mask // 255 
skeleton = skeletonize(bin_mask)

mask = np.ones(Mask.shape, dtype = "uint8")
mask = mask * 255

Mask_2 = cv2.imread(filepath_2, 0)
ret, bin_mask_2 = cv2.threshold(Mask_2, 127, 255, cv2.THRESH_BINARY)
bin_mask_2 = bin_mask_2 // 255 
skeleton_2 = skeletonize(bin_mask_2)

skeleton_ = skeleton_2 + skeleton

skeleton_[skeleton_ > 1] = 1
skeleton_ = skeleton_ * 255
skeleton_ = np.asarray(skeleton_, dtype = "uint8")


skeleton = skeleton * 255
skeleton = np.asarray(skeleton, dtype = "uint8")

length_1 = len(skeleton[skeleton>1])



skeleton_2 = skeleton_2 * 255
skeleton_2 = np.asarray(skeleton_2, dtype = "uint8")
img2_fg = cv2.bitwise_and(skeleton,skeleton_2,mask =mask)

length_2 = len(skeleton_2[skeleton_2>1])


arrays = np.where(skeleton>1)
arrays = np.asarray(arrays)


arrays_2 = np.where(skeleton_2>1)
arrays_2 = np.asarray(arrays_2)

length = min(length_1, length_2)

dif = np.array([length_1 - length, length_2 - length], dtype = "int")
start = dif // 2
end = start + length

print(start, end)

distance = []
for i in range(length): 
    


    pt1 = (arrays[1][start[0] + i], arrays[0][start[0] + i])
    pt2 = (arrays_2[1][start[1] + i], arrays_2[0][start[1] + i])
    dis = math.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0])  +   (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    distance.append(dis)
    if i % 10 == 0:
        #pass
        cv2.line(skeleton_, pt1, pt2, thickness = 1, color = (255,255,255))
print(distance)
distance = np.asarray(distance)
print(distance.min(), distance.max(), distance.mean())
cv2.imwrite("D:/Temp/test_AA.png", skeleton_)




