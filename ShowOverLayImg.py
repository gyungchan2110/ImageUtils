# In[]
import cv2 
import numpy as np   
import os 
from operator import eq
import random
import matplotlib.pyplot as plt 

BasePath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180125_103950_Expand_20pixel"
ImgPath = BasePath + '/Imgs/test'
MaskPath = BasePath + "/Mask_Rt Upper CB/test"

filenames  = next(os.walk(ImgPath))[2]

filename = random.choice(filenames)

image = cv2.imread(ImgPath + "/" + filename)
mask = cv2.imread(MaskPath + "/" + filename)

image = np.asarray(image)
mask = np.asarray(mask)

        
fig = plt.figure()
fig.set_size_inches(256/256, 1, forward=False)
ax = plt.Axes(fig, [0., 0., 1., 1.])
ax.set_axis_off()
fig.add_axes(ax)
image = image[:,:,0]
mask = mask[:,:,0]

plt.imshow(image, cmap = "gray")
ax.imshow(mask, cmap='gray', alpha = 0.15, interpolation = 'nearest')
plt.savefig("D:/Temp/"+filename, dpi = 2048)
plt.show()