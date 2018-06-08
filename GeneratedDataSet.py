
# In[]
import os
import cv2 
import numpy as np
import shutil 

ImgSrcBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Imgs_OriginalData_2k2k/Imgs"
MaskSrcBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Masks_OriginalData_2k2k"

ImgDstBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base/Imgs"

MaskDstBase = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base/Masks"


Classes = ["1_AS", "2_AR", "3_MS", "4_MR", "5_AS+AR", "6_MS_MR" ]
MasksTypes = ["Aortic Knob", "Lt Lower CB", "Pulmonary Conus", "Rt Lower CB", "Rt Upper CB", "DAO" , "Carina" , "LAA"]
FolderTypes = ["train", "validation", "test"]
ratio = np.array((6.,2.,2.))

ImgPathes = []
MaskPathes = []
ImgDsts = []
MaskDsts = []

ImgPathes.append(ImgSrcBase + "/" + "Normal")
MaskPathes.append(MaskSrcBase + "/" + "Normal")
ImgDsts.append(ImgDstBase )
MaskDsts.append(MaskDstBase )

for cl in Classes : 
    ImgPathes.append(ImgSrcBase + "/Abnormal/" + cl)
    MaskPathes.append(MaskSrcBase + "/Abnormal/" + cl)
    ImgDsts.append(ImgDstBase )
    MaskDsts.append(MaskDstBase )


for i, ImgSrcPath in enumerate(ImgPathes):
    
    ImgDstPath = ImgDsts[i]
    MaskSrcPath = MaskPathes[i]
    MaskDstPath = MaskDsts[i]

    files = os.listdir(ImgSrcPath) 

    filecount = len(files)

    ratio = ratio / ratio.sum()

    Counts = ratio * filecount 
    Counts = np.asarray(Counts, dtype="int16")



    for i, file in enumerate(files) :
        print(file)
        if( i < Counts[0]): 
            folder = FolderTypes[0] 
        elif(i >= Counts[0] and i < Counts[0] + Counts[1]) : 
            folder = FolderTypes[1] 
        else: 
            folder = FolderTypes[2]

        if(not os.path.isdir(ImgDstPath + "/" + folder)):
            os.mkdir(ImgDstPath + "/" + folder)

        shutil.copy(ImgSrcPath + "/" + file, ImgDstPath + "/" + folder)


        for masktype in MasksTypes: 

            if(not os.path.isdir(MaskDstPath + "/" + masktype)):
                os.mkdir(MaskDstPath + "/" + masktype)


            if(not os.path.isdir(MaskDstPath + "/" + masktype + "/" + folder)):
                os.mkdir(MaskDstPath + "/" + masktype + "/" + folder)

            shutil.copy( MaskSrcPath + "/" + masktype + "/" + file, MaskDstPath + "/" + masktype + "/" + folder + "/" + file)
        
# In[]

import shutil 
import os  

Imgpath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base/Imgs"
DstPath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_2_Generated_Data(2k)/Generated_Data_20180410_191400_Seg_Base/Masks/Thorax(x)"
Source = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/Temporal"

folders = ["test", "validation", "train"]

for folder in folders:
    
    if not os.path.isdir(DstPath + "/" + folder):
        os.mkdir(DstPath + "/" + folder)

    for file in os.listdir(Imgpath + "/" + folder):
        shutil.copy2(Source + "/" + file, DstPath + "/" + folder + "/" + file)