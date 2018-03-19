""""
Image Preprocesing Script 

Input : Folder 
Output : Folder 

Gaussian Bluring & Shapening 


""""



# In[]

import cv2 
import numpy as np
import matplotlib.pyplot as plt 

def GaussianBlur(srcPath, dstPath, fileName, kernal):
    img = cv2.imread(srcPath + "/" + fileName)
    img = cv2.GaussianBlur(img, (kernal, kernal), 0)
    img = cv2.GaussianBlur(img, (kernal, kernal), 0)
    cv2.imwrite(dstPath +"/" + fileName, img)
    # plt.imshow(img)
    # plt.show()
    


def Shapening(srcPath, dstPath, fileName, kernal):
    img = cv2.imread(srcPath + "/" + fileName)
    kernal = np.zeros((3,3))
    if kernal == 1:
        kernal = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    else:
        kernal = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])

    img = cv2.filter2D(img, -1, kernal)
    cv2.imwrite(dstPath +"/" + fileName, img)
    plt.imshow(img)
    plt.show()

Basic = "D:/[Data]/[Lung_Segmentation]/[PNG]_2_Generated_Data(2k)"
src = Basic + "/GeneratedData_20180316_094000_Basic/Img"
dst = Basic + "/GeneratedData_20180316_140100_Blur_k3_e1/Img"

Folders = ["train", "validation", "test"]

kernals = [3,5,1,2]

for Folder in Folders : 
    for file in os.listdir(src + "/" + Folder):
        imgFile = src + "/" + Folder 
        dstFile = dst + "/" + Folder 
        GaussianBlur(imgFile, dstFile, file, 5)
