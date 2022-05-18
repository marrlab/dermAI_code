from keras import backend as K
import tensorflow as tf


def bag_accuracy(y_true, y_pred):
    """Compute accuracy of one bag.
    Parameters
    ---------------------
    y_true : Tensor (N x 1)
        GroundTruth of bag.
    y_pred : Tensor (1 X 1)
        Prediction score of bag.
    Return
    ---------------------
    acc : Tensor (1 x 1)
        Accuracy of bag label prediction.
    """
    y_true = tf.cast(K.mean(y_true, axis=0, keepdims=False),tf.int64)
    y_pred = (K.mean(y_pred, axis=0, keepdims=False))
    #print(y_pred.shape)
    
    #y_true=tf.one_hot(tf.cast((y_true),tf.int32), depth=int(y_pred.shape[1]))
    #print(y_true.shape)
    acc = K.equal(K.argmax((y_pred)), (y_true))
    return acc


def bag_loss(y_true, y_pred):
    """Compute binary crossentropy loss of predicting bag loss.
    Parameters
    ---------------------
    y_true : Tensor (N x 1)
        GroundTruth of bag.
    y_pred : Tensor (1 X 1)
        Prediction score of bag.
    Return
    ---------------------
    acc : Tensor (1 x 1)
        Binary Crossentropy loss of predicting bag label.
    """
    y_true = K.mean(y_true, axis=0, keepdims=False)
    y_pred = K.mean(y_pred, axis=0, keepdims=False)
    #loss = K.mean(K.binary_crossentropy(y_true, y_pred))
    #loss = K.mean(K.categorical_crossentropy(y_true, y_pred), axis=-1)
    y_true=tf.one_hot(tf.cast(K.mean(y_true),tf.int32), depth=int(y_pred.shape[0]))
    cce = tf.keras.losses.CategoricalCrossentropy()
    loss = cce(y_true, y_pred)
    #loss = K.mean(scce(y_true, y_pred))
    return loss