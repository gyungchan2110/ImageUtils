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