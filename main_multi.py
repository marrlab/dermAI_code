#!/usr/bin/env python
'''
This is a re-implementation of the following paper:
"Attention-based Deep Multiple Instance Learning"
I got very similar results but some data augmentation techniques not used here
https://128.84.21.199/pdf/1802.04712.pdf
*---- Jiawen Yao--------------*
'''


import numpy as np
import random
import pandas as pd
import time
import os
from utl.dermAI_plot_results import plot_results
from utl import Cell_Net_truncated_multiclass
from random import shuffle
import argparse
from keras.models import Model
from utl.dataset import load_dataset
#from utl.data_aug_op import random_flip_img, random_rotate_img
import scipy.misc as sci
import tensorflow as tf
from keras import backend as K
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
#import matplotlib.pyplot as plt
import torch
import pickle
import wandb
import h5py
from wandb.keras import WandbCallback

def parse_args():
    """Parse input arguments.
    Parameters
    -------------------
    No parameters.
    Returns
    -------------------
    args: argparser.Namespace class object
        An argparse.Namespace class object contains experimental hyper-parameters.
    """
    parser = argparse.ArgumentParser(description='Train a Attention-based Deep MIL')
    parser.add_argument('--lr', dest='init_lr',
                        help='initial learning rate',
                        default=1e-4, type=float)
    parser.add_argument('--decay', dest='weight_decay',
                        help='weight decay',
                        default=0.0005, type=float)
    parser.add_argument('--momentum', dest='momentum',
                        help='momentum',
                        default=0.9, type=float)
    parser.add_argument('--epoch', dest='max_epoch',
                        help='number of epoch to train',
                        default=100, type=int)
    parser.add_argument('--useGated', dest='useGated',
                        help='use Gated Attention',
                        default=False, type=int)

    # if len(sys.argv) == 1:
    #     parser.print_help()
    #     sys.exit(1)

    args = parser.parse_args()
    return args

def generate_batch(path,class_data,target):
    bags = []
    classes= class_data[target].unique().tolist()
    for each_path in path:
        name_img = []
        img = []
        #print(each_path)
        #img_vectors = torch.load(each_path)
        f= h5py.File(each_path, 'r')
        img_vectors=np.array(f.get("features"))
        f.close()
        coords=pd.read_hdf(each_path, 'coords')
        num_ins = len(img_vectors)
        label = class_data.loc[class_data['Folder']==(each_path.split('/')[-1][:-3])][target].tolist()
        #curr_label=([np.array([classes.index(label[0])])]*num_ins)
        curr_label=([classes.index(label[0])]*num_ins)
        for i in range(0,num_ins):
            img_vector = img_vectors[i]#.cpu().detach().numpy()
            img.append(img_vector)
            name_img.append(each_path.split('/')[-1][:-3])
        bags.append((img, curr_label, coords, name_img))

    return bags

def  Get_train_valid_Path(Train_set, train_percentage=0.8):
    """
    Get path from training set
    :param Train_set:
    :param train_percentage:
    :return:
    """
    
    indexes = np.arange(len(Train_set))
    random.shuffle(indexes)

    num_train = int(train_percentage*len(Train_set))
    train_index, test_index = np.asarray(indexes[:num_train]), np.asarray(indexes[num_train:])

    Model_Train = [Train_set[i] for i in train_index]
    Model_Val = [Train_set[j] for j in test_index]

    return Model_Train, Model_Val


def test_eval(model, test_set, patch_size):
    """Evaluate on testing set.
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    test_set : list
        A list of testing set contains all training bags features and labels.
    Returns
    -----------------
    test_loss : float
        Mean loss of evaluating on testing set.
    test_acc : float
        Mean accuracy of evaluating on testing set.
    """
    num_test_batch = len(test_set)
    test_loss = np.zeros((num_test_batch, 1), dtype=float)
    test_acc = np.zeros((num_test_batch, 1), dtype=float)
    for ibatch, batch in enumerate(test_set):
        img_data = batch[0]         
        img_batch = []
        for i in img_data:
            img = i           
            exp_img = np.expand_dims(img, 0)
            img_batch.append(exp_img)
        result = model.predict_on_batch(x=np.concatenate(img_batch))
        #test_loss[ibatch] = result[0]
        test_loss[ibatch] = batch[1][0]
        #test_acc[ibatch] = result[1]
        test_acc[ibatch] = np.argmax(result)==batch[1][0]
    return (test_loss), (test_acc)
    

def train_eval(model, train_set, irun, ifold, patch_size, identifier):
    """Evaluate on training set. Use Keras fit_generator
    Parameters
    -----------------
    model : keras.engine.training.Model object
        The training mi-Cell-Net model.
    train_set : list
        A list of training set contains all training bags features and labels.
    Returns
    -----------------
    model_name: saved lowest val_loss model's name
    """
    batch_size = 1
    model_train_set, model_val_set = Get_train_valid_Path(train_set, train_percentage=0.9)

    from utl.DataGenerator import DataGenerator
    train_gen = DataGenerator(batch_size=batch_size, shuffle=False, patch_size=patch_size).generate(model_train_set)
    val_gen = DataGenerator(batch_size=batch_size, shuffle=False, patch_size=patch_size).generate(model_val_set)
    if not os.path.exists("Saved_model_"+identifier):
        os.makedirs("Saved_model_"+identifier)
    model_name = "Saved_model_"+identifier+"/" + "Batch_size_" + str(batch_size) + "_fold_" + str(ifold)+ "_best.hd5"

    checkpoint_fixed_name = ModelCheckpoint(model_name,
                                            monitor='val_loss', verbose=1, save_best_only=True,
                                            save_weights_only=True, mode='auto')

    EarlyStop = EarlyStopping(monitor='val_loss', patience=20)

    callbacks = [checkpoint_fixed_name, EarlyStop,WandbCallback()]

    history = model.fit(train_gen, steps_per_epoch=len(model_train_set)//batch_size,
                                             epochs=args.max_epoch, validation_data=val_gen,
                                            validation_steps=len(model_val_set)//batch_size, callbacks=callbacks)

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    train_acc = history.history['bag_accuracy']
    val_acc = history.history['val_bag_accuracy']

    return model_name


def model_training(input_dim, output_dim, dataset, irun, ifold, class_data, patch_size,target, identifier):

    train_bags = dataset['train']
    test_bags = dataset['test']

    # convert bag to batch
    train_set = generate_batch(train_bags, class_data,target)
    test_set = generate_batch(test_bags, class_data,target)

    model = Cell_Net_truncated_multiclass.cell_net(input_dim,output_dim, args, useMulGpu=False)
   # preTrained="Saved_model_600_two_classes/Batch_size_1_fold_0_best.hd5"
   # model.load_weights(preTrained)
    # train model
    t1 = time.time()
    # for epoch in range(args.max_epoch):
    model_name = train_eval(model, train_set, irun, ifold, patch_size, identifier)

    print("load saved model weights")
    model.load_weights(model_name)

    test_loss, test_acc = test_eval(model, test_set, patch_size)
    if sum(test_acc==0) + sum(test_acc==1) != len(test_acc):
        print(f'wrong test_acc:{test_acc}')
    t2 = time.time()
    #

    print ('run time:', (t2 - t1) / 60.0, 'min')
    print ('test_acc={:.3f}'.format(np.mean(test_acc)))

    return np.mean(test_acc), model



if __name__ == "__main__":
    
    identifier='AK_SCC_low_res70'
    wandb.init(project=identifier, entity="vlupperger")

    args = parse_args()
    wandb.config = {
        "learning_rate": args.init_lr,
        "epochs": args.max_epoch,
        "batch_size": 1
    }

    print ('Called with args:')
    print (args)
    patch_size = 2048
    input_dim = (patch_size,)
    random.seed(37)
    run = 1
    n_folds = 5
    
    target='Disease'
    
    data_path = './dermAI_data/CCL'
    class_data= pd.read_excel(io=(data_path+'/sample_Data_'+identifier+'.xlsx'),engine='openpyxl')

    acc = np.empty((run, n_folds), dtype=object)
    classes= class_data[target].unique().tolist()
    num_classes = len(classes)
    irun=0
    datasets = load_dataset(dataset_path=data_path, n_folds=n_folds, rand_state=irun, num_classes=num_classes, class_data=class_data, target=target, classes=classes)
    for ifold in range(0,n_folds):
        print ('run=', irun, '  fold=', ifold)
        acc[irun][ifold],model = (model_training(input_dim, num_classes, datasets[ifold], irun, ifold, class_data, patch_size,target, identifier))
    f = open('Saved_model_'+identifier+'/datasets.pckl', 'wb')
    pickle.dump(datasets, f)
    f.close()
    result_path="Saved_model_"+identifier+"/"
    plot_results(class_data, datasets, target, result_path, model, input_dim,'mil_ccl')
    with open(result_path+'acc_mil_ccl.txt', 'w') as f:
        f.write('Test accuracy = '+str(np.mean(acc))+ ' +- ' + str(np.std(acc)))

    print ('mi-net mean accuracy = ', np.mean(acc))
    print ('std = ', np.std(acc))
