
import tensorflow as tf

#from keras.utils import multi_gpu_model
from keras.models import Model
from keras.optimizers import SGD,Adam
from keras.regularizers import l2
from keras.layers import Input, Dense, Layer, Dropout, Conv2D, MaxPooling2D, Flatten, multiply
from .metrics_multi import bag_accuracy, bag_loss
from .custom_layers import Mil_Attention, Last_Sigmoid

def cell_net(input_dim, output_dim, args, useMulGpu=False):

    lr = args.init_lr
    weight_decay = args.init_lr
    momentum = args.momentum

    data_input = Input(shape=input_dim, dtype='float32', name='input')

    fc1 = Dense(2048, activation='relu',kernel_regularizer=l2(weight_decay), name='fc1')(data_input)
    fc1 = Dropout(0.5)(fc1)
    fc2 = Dense(1024, activation='relu', kernel_regularizer=l2(weight_decay), name='fc2')(fc1)
    fc2 = Dropout(0.5)(fc2)

  #  fp = Feature_pooling(output_dim=1, kernel_regularizer=l2(0.0005), pooling_mode='max',
#                          name='fp')(fc2)

    alpha = Mil_Attention(L_dim=128, output_dim=1, kernel_regularizer=l2(weight_decay), name='alpha', use_gated=args.useGated)(fc2)
    x_mul = multiply([alpha, fc2])

    out = Last_Sigmoid(output_dim=output_dim, name='FC1_sigmoid')(x_mul)
    #
    model = Model(inputs=[data_input], outputs=[out])

    #model.summary()

    if useMulGpu == True:
        #parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=bag_loss, metrics=[bag_accuracy])
    else:
        model.compile(optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999), loss=bag_loss, metrics=[bag_accuracy])
        parallel_model = model

    return parallel_model



