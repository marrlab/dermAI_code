import numpy as np
import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold

def load_dataset(dataset_path, n_folds, rand_state, num_classes, class_data, target, classes):
    """
    Parameters
    --------------------
    :param dataset_path:
    :param n_folds:
    :return: list
        List contains split datasets for K-Fold cross-validation
    """
    all_path=[]
    # load datapath from path
    
    #classes= class_data['Subclass'].unique().tolist()
    for i in range(num_classes):
        class_files = class_data.loc[class_data[target]==(classes[i])]['Folder'].tolist()
        all_path = all_path + [dataset_path+'/h5_files/'+item+'.h5' for item in class_files]
        #all_path = all_path + glob.glob(dataset_path+'/'+str(i)+'/img*')
    #pos_path = glob.glob(dataset_path+'/1/img*')
    #neg_path = glob.glob(dataset_path+'/0/img*')

    #pos_num = len(pos_path)
    #neg_num = len(neg_path)

    #all_path = pos_path + neg_path
    all_path=list(map(lambda st: str.replace(st, "\\", "/"), all_path))
    #print(len(all_path))
    #num_bag = len(all_path)
    kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rand_state)
    datasets = []
    for train_idx, test_idx in kf.split(all_path,class_data[target]):
        dataset = {}
        dataset['train'] = [all_path[ibag] for ibag in train_idx]
        dataset['test'] = [all_path[ibag] for ibag in test_idx]
        datasets.append(dataset)
    return datasets