
from keras.layers import *
from keras.layers import BatchNormalization
from keras import initializers, regularizers, constraints, optimizers
from keras.models import load_model, Model
import keras.backend as K


def AtzoriNet(input_shape, classes, n_pool='average', n_dropout=0., n_l2=0.0005, n_init='glorot_normal', batch_norm=False):

    if n_init == 'glorot_normal':
        kernel_init = initializers.glorot_normal(seed=0)
    elif n_init == 'glorot_uniform':
        kernel_init = initializers.glorot_uniform(seed=0)
    elif n_init == 'he_normal':
        kernel_init = initializers.he_normal(seed=0)
    elif n_init == 'he_uniform':
        kernel_init = initializers.he_uniform(seed=0)
    elif n_init == 'normal':
        kernel_init = initializers.normal(seed=0)
    elif n_init == 'uniform':
        kernel_init = initializers.uniform(seed=0)
    # kernel_init = n_init
    kernel_regl = regularizers.l2(n_l2)

    ## Block 0 [Input]
    X_input = Input(input_shape, name='b0_input')
    X = X_input
    if batch_norm:
        X = BatchNormalization()(X)
    
    ## Block 1 [Pad -> Conv -> ReLU -> Dropout]
    X = ZeroPadding2D((0,4))(X)
    X = Conv2D(32, (1, 10), padding='valid', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b1_conv2d_32_1x10')(X)
    X = Activation('relu', name='b1_relu')(X)
    X = Dropout(n_dropout, name='b1_dropout')(X)
    
    ## Block 2 [Pad -> Conv -> ReLU -> -> Dropout -> Pool]
    X = ZeroPadding2D((1,1))(X)
    X = Conv2D(32, (3, 3), padding='valid', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b2_conv2d_32_3x3')(X)
    X = Activation('relu', name='b2_relu')(X)
    X = Dropout(n_dropout, name='b2_dropout')(X)
    if n_pool == 'max':
        X = MaxPooling2D((3,3), strides = (3,3), name='b2_pool')(X)
    else:
        X = AveragePooling2D((3,3), strides = (3,3), name='b2_pool')(X)
    
    ## Block 3 [Pad -> Conv -> ReLU -> Dropout -> Pool]
    X = ZeroPadding2D((2,2))(X)
    X = Conv2D(64, (5, 5), padding='valid', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b3_conv2d_64_5x5')(X)
    X = Activation('relu', name='b3_relu')(X)
    X = Dropout(n_dropout, name='b3_dropout')(X)
    if n_pool == 'max':
        X = MaxPooling2D((3,3), strides = (3,3), name='b3_pool')(X)
    else:
        X = AveragePooling2D((3,3), strides = (3,3), name='b3_pool')(X)
    
    ## Block 4 [Pad -> Conv -> ReLU -> Dropout]
    X = ZeroPadding2D((2,0))(X)
    X = Conv2D(64, (5, 1), padding='valid', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b4_conv2d_64_5x1')(X)
    X = Activation('relu', name='b4_relu')(X)
    X = Dropout(n_dropout, name='b4_dropout')(X)
    
    ## Block 5 [Pad -> Conv -> Softmax]
    X = Conv2D(classes, (1, 1), padding='same', kernel_regularizer=kernel_regl, kernel_initializer=kernel_init, name='b5_conv2d_{}_1x1'.format(classes))(X)
    X = Activation('softmax', name='b5_soft')(X)
    X = Reshape((-1,), name='b5_reshape')(X)
    
    model = Model(inputs = X_input, outputs = X, name='AtzoriNet')

    return model

def getNetwork(network):

    model = AtzoriNet
    return model


