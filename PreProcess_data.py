# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 11:06:03 2021

@author: valerio.lupperger
"""
import pickle
import glob
import os
import pandas as pd
from utils.dermAI_utils_server import extract_Features

#load input file
# f = open('./input_new.pckl', 'rb')
# input_data = pickle.load(f)
# f.close()
input_data= pd.read_excel(io=('./dermAI_data/sample_Data_AK_SCC_x.xlsx'),engine='openpyxl')
#data path
path = "/lustre/groups/aih/aap"
#path = "G:/dermAI_data"
#save path
out_path= "/home/icb/valerio.lupperger/dermAI/dermAI_data/CCL22"
#out_path= "D:/dermAI_data/KimiaNet"
patch_width=512
out_file= list()
img_files = glob.glob(path + "/**/*Patient*.czi", recursive = True)
#input_data= input_data[::-1]
for idx, row in input_data.iterrows():
    print(idx)
    patientID=row['Folder'].split('_')[1]+'_'+row['Folder'].split('_')[2]
    case_files= [s for s in img_files if 'Patient_'+patientID in s]
    
    if len(case_files)<1:#no images for the case
        continue
    #out_file.append([row['Class'],row['Subclass'],row['Disease'],'Patient_'+patientID])
    if os.path.isfile(out_path+'/h5_files/Patient_'+patientID+'.h5'):
        continue
    #
    model=extract_Features(case_files, patch_width, out_path, patientID,mode='CCL')
                           
out_df= pd.DataFrame(out_file,columns=['Class','Subclass','Disease','Folder'])    
out_df.to_excel(out_path+'/sample_Data.xlsx')                    
