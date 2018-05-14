# In[]
import cv2 
import numpy as np   
import os 
from operator import eq
import random
import matplotlib.pyplot as plt 

BasePath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base_Expand_40pixel_Cropped_Detected"
ImgPath = BasePath + '/Imgs/test'
MaskPaths = BasePath + "/Masks"
Dstpath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Temp_OverLay"
Classlist = ["Aortic Knob", "Lt Lower CB", "Pulmonary Conus", "Rt Lower CB", "Rt Upper CB", "DAO" , "Carina" , "LAA"]

os.environ["CUDA_VISIBLE_DEVICES"]= "0"

for file in os.listdir(ImgPath): 

    img = cv2.imread(ImgPath + "/" + file)
    img = np.asarray(img, dtype = "uint8")
 

    fig = plt.figure()
    fig.set_size_inches(256/256, 1, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    img = img[:,:,0]

    plt.imshow(img, cmap = "gray")
    masks = np.zeros(img.shape, dtype = "uint8")
    for clss in Classlist : 
        Maskpath = MaskPaths + "/" + clss + "/test"
        if(not os.path.isfile(Maskpath + "/" + file)):
            continue
        mask = cv2.imread(Maskpath + "/" + file)
        mask = np.asarray(mask, dtype = "uint8")
        mask = mask // 255
        masks = np.bitwise_or(masks, mask[:,:,0])  

    linewidth = 10 

    thorax_x = np.zeros(img.shape, dtype = "uint8")
    thorax_x = cv2.rectangle(thorax_x, (512 - linewidth //2 ,0), (512 + linewidth //2 ,1024), color = (255,255,255), thickness = -1)
    thorax_x = thorax_x // 255
    masks = np.bitwise_or(masks, thorax_x)  


    axis = np.zeros(img.shape, dtype = "uint8")
    axis = cv2.rectangle(axis, (256 - linewidth //2 ,0), (256 + linewidth //2 ,1024), color = (255,255,255), thickness = -1)
    axis = axis // 255
    masks = np.bitwise_or(masks, axis)  

    masks = masks * 255    
    ax.imshow(masks, cmap='gray', alpha = 0.15, interpolation = 'nearest')
    plt.savefig(Dstpath + "/"+file, dpi = 2048)
    plt.show()

