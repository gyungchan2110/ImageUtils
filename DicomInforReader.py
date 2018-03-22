# In[]

import SimpleITK as sitk

import sys, os
import csv
from operator import eq

metaFile = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/BasicData_MetaFile.csv"
metaFile_ex = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[PNG]_1_Basic_Data(2k)/BasicData_MetaFile_Ex.csv"
#filePath = "D:/[Data]/[Cardiomegaly]/1_ChestPA_Labeled_Baeksongyi/[DCM]_0_Original/Normal_20180118_1020/Normal_20180118_201/1000000/1000000.dcm"

saveFolder = "D:/Temp"


f = open(metaFile, 'r', encoding='utf-8', newline='')
reader = csv.reader(f)

f_ex = open(metaFile_ex, 'w', encoding='utf-8', newline='')
writer = csv.writer(f_ex)

currentFolder = ""
for line in reader:
    datas = line
    originalFile = datas[0] + "/" + datas[1] + "/" + datas[1] + ".dcm"
    img = sitk.ReadImage(originalFile)


    PID = img.GetMetaData("0010|0020")

    datas.append(str(PID))

    writer.writerow(datas)
    
    if(not eq(currentFolder, datas[0])):
        currentFolder = datas[0]

    folderlevels = currentFolder.split("/")
    folder = folderlevels[len(folderlevels)-1]
    saveFile = saveFolder + "/" + folder + ".csv"

    f_save = open(saveFile, 'a', encoding='utf-8', newline='')
    f_writer = csv.writer(f_save)
    f_writer.writerow(datas)
    f_save.close()

    print(originalFile)
    
f.close()
f_ex.close()
