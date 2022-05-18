# -*- coding: utf-8 -*-
"""
Created on Mon Jan 31 10:35:41 2022

@author: valerio.lupperger
"""
def plot_results(class_data, datasets, target, result_path, model, input_dim,outID):
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
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
    

    classes= class_data[target].unique().tolist()
    #args= parse_args()
    #args.useGated=True
    #input_dim = (1024,)
    output_dim=len(classes)
    #model = Cell_Net_truncated_multiclass.cell_net(input_dim, output_dim , args, useMulGpu=False)
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
            #coords=pd.read_hdf(data_path+'/h5_files/'+ID+'.h5', 'coords')
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
    show_pred_mtrx.show_pred_mtrx(conf_mat, class_coversion,path_save=result_path+'/conf_mat_'+outID+'.png')
    df = pd.DataFrame(classification_report(out['gt'], 
                                        out['pred'], digits=2,target_names=classes,
                                        output_dict=True)).T
    df.round(2)
    df['support']=df['support'].astype(int)
    fig,ax = render_mpl_table(df, header_columns=0, col_width=2.0)
    fig.savefig(result_path+'/result_table_'+outID+'.png',bbox_inches='tight')
    f = open('./input_new.pckl', 'rb')
    output = pickle.load(f)
    f.close()
    result  = pd.merge( output,out, on=["Folder"])
    result.to_excel(result_path+'/result_'+outID+'.xlsx')
    
    #ax.set_xticklabels(classes, rotation=45, ha='right', rotation_mode='anchor')
    #disp.plot(cmap=plt.cm.Greys,ax=ax,xticks_rotation=45)
    #plt.savefig(result_path+'/conf_mat.png',bbox_inches='tight', dpi=300)
    #result = pd.merge(out, class_data, on="Folder")
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
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

