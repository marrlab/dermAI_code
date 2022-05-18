# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 11:57:41 2021

@author: valerio.lupperger
"""
import gc
import pandas as pd
import os
from PIL import Image
import h5py
import slideio
import numpy as np
from torchvision import models, transforms
import torch
from utils.resnet_custom import resnet50_baseline
import torch.nn as nn
import utils.ResNet as ResNet
from utils.ccl import CCL

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%%
class fully_connected(nn.Module):
	"""docstring for BottleNeck"""
	def __init__(self, model, num_ftrs, num_classes):
		super(fully_connected, self).__init__()
		self.model = model
		self.fc_4 = nn.Linear(num_ftrs,num_classes)

	def forward(self, x):
		x = self.model(x)
		x = torch.flatten(x, 1)
		out_1 = x
		out_3 = self.fc_4(x)
		return  out_1


def extract_Features(case_files, patch_width, out_path, patientID,mode):
    if mode=='imageNet':
        model = resnet50_baseline(pretrained=True)
    elif mode=='KimiaNet':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.features = nn.Sequential(model.features , nn.AdaptiveAvgPool2d(output_size= (1,1)))

        model = fully_connected(model.features, 1024,30)
        model.load_state_dict(torch.load('./utils/KimiaNetPyTorchWeights.pth'),strict=False)
    elif mode=='CCL':
        backbone = ResNet.resnet50
        model = CCL(backbone, 128, 65536, mlp=True, two_branch=True, normlinear=True).cuda()
        pretext_model = torch.load(r'./utils/best_ckpt.pth')
        model.load_state_dict(pretext_model, strict=True)
        model.encoder_q.fc = nn.Identity()
        model.encoder_q.instDis = nn.Identity()
        model.encoder_q.groupDis = nn.Identity()
    model = model.to(device)
    model.eval()
    transform = transforms.Compose([            #[1]
            transforms.Resize(256),                    #[2]
            #transforms.CenterCrop(224),                #[3]
            transforms.ToTensor(),                     #[4]
            transforms.Normalize(                      #[5]
                    mean=[0.485, 0.456, 0.406],                #[6]
                    std=[0.229, 0.224, 0.225]                  #[7]
                    )])
    #create folders
    #os.makedirs(os.path.join(out_path, 'pt_files'), exist_ok=True)
    os.makedirs(os.path.join(out_path, 'h5_files'), exist_ok=True)
    os.makedirs(os.path.join('/lustre/groups/aih/aap/img_files'), exist_ok=True)
    os.makedirs(os.path.join('/lustre/groups/aih/aap/img_files/Patient_'+patientID), exist_ok=True)
    coords = pd.DataFrame({'file attachment' : [],'scene' : [], 'x' : [], 'y' : []}) 
    feats= np.empty([0,2048])
    for file in case_files:
            
        
        attachment= file.split('_15')[-1][0:-4]
        slide = slideio.Slide(file,driver="CZI")
    
        for scn in range(0,slide.num_scenes):                
            scene=slide.get_scene(scn)
            if scene.size[0]<2048 or scene.size[1]<2048:
                continue
            #img=scene.read_block((int(np.floor(np.mod(scene.size[0],2048)/2)),int(np.floor(np.mod(scene.size[1],2048)/2)),int(np.floor(scene.size[0]/2048))*2048,int(np.floor(scene.size[1]/2048))*2048),(int(np.floor(scene.size[0]/2048))*128,int(np.floor(scene.size[1]/2048))*128))

            #im= Image.fromarray(img)
            #im.save('/lustre/groups/aih/aap/img_files/Patient_'+patientID+'/Patient_'+patientID+'_file'+attachment+'_scene_'+str(scn)+'.tif')
            print('/lustre/groups/aih/aap/img_files/Patient_'+patientID+'/Patient_'+patientID+'_file'+attachment+'_scene_'+str(scn)+'.tif')            
            #del im
            #gc.collect()
            img=scene.read_block((int(np.floor(np.mod(scene.size[0],2048)/2)),int(np.floor(np.mod(scene.size[1],2048)/2)),int(np.floor(scene.size[0]/2048))*2048,int(np.floor(scene.size[1]/2048))*2048),(int(np.floor(scene.size[0]/2048))*2048,int(np.floor(scene.size[1]/2048))*2048))
            
            for x in range(0,img.shape[0],patch_width):
                for y in range(0,img.shape[1],patch_width):                    
                    patch = img[x:x+patch_width,y:y+patch_width,:]
                    if ((sum(sum(np.logical_and(patch[:,:,0]>200,patch[:,:,1]>200,patch[:,:,2]>200)))) +(sum(sum(np.logical_and(patch[:,:,0]==0,patch[:,:,1]==0,patch[:,:,2]==0)))))/(patch_width*patch_width) < 0.7:
                        im = Image.fromarray(patch)                                
                        img_t = transform(im)
                        batch_t = torch.unsqueeze(img_t, 0)
                        batch_t = batch_t.to(device)
                        features = model(batch_t)			
                        features = features.cpu().detach().numpy()
                        feats = np.append(feats,features,0)
                        coords=coords.append(pd.DataFrame(data=[[attachment,scn,x,y]], columns=['file attachment','scene','x','y']),ignore_index=True)

    # Write data to HDF5 and pt
    print('features size: ', feats.shape)
    print('coordinates size: ', coords.shape)    
    file= h5py.File(out_path+'/h5_files/Patient_'+patientID+'.h5', 'w')
    file.create_dataset("features", data=feats)
    file.close()
    coords.to_hdf(out_path+'/h5_files/Patient_'+patientID+'.h5', "coords")
   # features = torch.from_numpy(feats)
    #torch.save(features, os.path.join(out_path, 'pt_files', 'Patient_'+patientID+'.pt'))                 
    return model                     

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([1, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    mpl_table = ax.table(cellText=data.round(2).values, bbox=bbox, colLabels=data.columns,rowLabels=data.index.values, **kwargs)
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
    return ax.get_figure(), ax                            
#%%
def plot_results(class_data, datasets, target, result_path, model, input_dim):
    import os
    from utl import Cell_Net_truncated_multiclass
    from utl import show_pred_mtrx
    from main_multi import generate_batch
    from main_multi import parse_args
    from keras.models import Model
    import pandas as pd
    import numpy as np
    import pickle
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    
    classes= class_data[target].unique().tolist()
    args= parse_args()
    args.useGated=True
    input_dim = (1024,)
    output_dim=len(classes)
    model = Cell_Net_truncated_multiclass.cell_net(input_dim, output_dim , args, useMulGpu=False)
    acc=[]
    #f.close()
    out=pd.DataFrame({'Folder' : [],'gt' : [], 'pred' : [], 'acc' : []})
    for idx, dataset in enumerate(datasets):
        testdata=dataset.get('test')
        model.load_weights(result_path+'Batch_size_1_fold_' + str(idx)+ '_best.hd5')
        inter_output_model = Model(model.input, model.get_layer(index = 5).output )
        acc1=[]
        for test_bag in testdata:
            
    #test_bag=['D:/DermAI_test/pt_files/Patient_'+Patient_ID+'.pt']
            test_set1 = generate_batch([test_bag],class_data,target)
            batch= test_set1[0]
            img_batch =[]
            for i in batch[0]:          
                exp_img = np.expand_dims(i, 0)
                img_batch.append(exp_img)
            res = model.predict_on_batch(x=np.concatenate(img_batch))
            re = model.test_on_batch(x=np.concatenate(img_batch),y=batch[1])
            acc1.append(re[1])
            out = out.append(pd.DataFrame(data=[[test_bag.split('/')[-1][:-3],batch[1][0],res[0], np.argmax(res),re[1]]], columns=['Folder','gt','pred_all','pred','acc']),ignore_index=True)
            #attention=np.zeros((len(batch[0]),1))
    #model.load_weights('D:/Nextcloud/PostDoc/scripts/Atten_Deep_MIL_keras_dermAI/Saved_model/Class_pred_best.hd5')
            #attention = inter_output_model.predict(x=np.concatenate(img_batch))
        acc.append(np.mean(acc1))
    conf_mat=confusion_matrix(out['gt'],out['pred'])
    class_coversion=pd.DataFrame({'art_lbl':list(range(0,len(classes))),'true_lbl':classes,'size_tot':np.sum(conf_mat,axis=1)})
    show_pred_mtrx.show_pred_mtrx(conf_mat, class_coversion,path_save=result_path+'/conf_mat_new.png')
    df = pd.DataFrame(classification_report(out['gt'], 
                                        out['pred'], digits=2,target_names=classes,
                                        output_dict=True)).T
    df.round(2)
    df['support']=df['support'].astype(int)
    fig,ax = render_mpl_table(df, header_columns=0, col_width=2.0)
    fig.savefig(result_path+'/result_table.png',bbox_inches='tight')
    

