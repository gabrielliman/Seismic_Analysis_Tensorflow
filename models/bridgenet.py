from tensorflow import keras
from keras.models import *
from keras.layers import *

def input_block(num_dim=2, X_channels=1, pool_size_all=2):
    if num_dim == 3:
        top_inputs = Input(shape=(None, None, None, X_channels))
        pool_size = (pool_size_all, pool_size_all, pool_size_all)
    elif num_dim == 2:
        top_inputs = Input(shape=(None, None, X_channels))
        pool_size = (pool_size_all, pool_size_all)
    elif num_dim == 1:
        top_inputs = Input(shape=(None, X_channels))
        pool_size = pool_size_all
    return top_inputs, pool_size

def conv2act_block(num_dim=2, kernel_size=3, use_BN=False, kernels_now=16, act_hide=1, drop_rate=0, conv2act_repeat=1, dilation_rate=1, input_layer=None):
    for repeat in range(conv2act_repeat + 1):
        if num_dim == 3:
            output_layer = Conv3D(filters=kernels_now, kernel_size=kernel_size, padding='same', dilation_rate=(dilation_rate, dilation_rate, dilation_rate))(input_layer)
        elif num_dim == 2:
            output_layer = Conv2D(filters=kernels_now, kernel_size=kernel_size, padding='same', dilation_rate=(dilation_rate, dilation_rate))(input_layer)
        elif num_dim == 1:
            output_layer = Conv1D(filters=kernels_now, kernel_size=kernel_size, padding='same', dilation_rate=dilation_rate)(input_layer)


        if drop_rate > 0:
            output_layer = Dropout(rate=drop_rate)(output_layer)


        if use_BN:
            output_layer = BatchNormalization()(output_layer)


        if act_hide == 1:
            output_layer = Activation('relu')(output_layer)
        elif act_hide == 2:
            output_layer = LeakyReLU(alpha=0.3)(output_layer)
        elif act_hide == 3:
            output_layer = PReLU(alpha_initializer='zeros')(output_layer)
        elif act_hide == 4:
            output_layer = ELU(alpha=1.0)(output_layer)
        elif act_hide == 5:
            output_layer = ThresholdedReLU(theta=1.0)(output_layer)


        input_layer = output_layer

    return output_layer


def pooling_block(num_dim=2, pool_size=None, pool_way=1, input_layer=None):
    if num_dim == 3:
        if pool_way == 1:
            output_layer = MaxPooling3D(pool_size=pool_size)(input_layer)
        elif pool_way == 2:
            output_layer = AveragePooling3D(pool_size=pool_size)(input_layer)

    elif num_dim == 2:
        if pool_way == 1:
            output_layer = MaxPooling2D(pool_size=pool_size)(input_layer)
        elif pool_way == 2:
            output_layer = AveragePooling2D(pool_size=pool_size)(input_layer)

    elif num_dim == 1:
        if pool_way == 1:
            output_layer = MaxPooling1D(pool_size=pool_size)(input_layer)
        elif pool_way == 2:
            output_layer = AveragePooling1D(pool_size=pool_size)(input_layer)

    return output_layer


def up_concatenate_block(num_dim=2, up_sampling_size=None, up_layer=None, concatenate_layer=None):
    # ##########################################################################
    if num_dim == 3:
        output_layer = concatenate([UpSampling3D(size=up_sampling_size)(up_layer), concatenate_layer], axis=-1)
    
    elif num_dim == 2:
        output_layer = concatenate([UpSampling2D(size=up_sampling_size)(up_layer), concatenate_layer], axis=-1)
    
    elif num_dim == 1:
        output_layer = concatenate([UpSampling1D(size=up_sampling_size)(up_layer), concatenate_layer], axis=-1)

    return output_layer


def output_block(num_dim=2, Y_channels=1, act_last=0, input_layer=None):
    # ##########################################################################
    if num_dim == 3:
        output_layer = Conv3D(filters=Y_channels, kernel_size=1)(input_layer)

    elif num_dim == 2:
        output_layer = Conv2D(filters=Y_channels, kernel_size=1)(input_layer)
        
    elif num_dim == 1:
        output_layer = Conv1D(filters=Y_channels, kernel_size=1)(input_layer)

    # ##########################################################################
    if act_last == 1:
        output_layer = Activation('sigmoid')(output_layer)
    elif act_last == 2:
        # [-1 1]
        output_layer = Activation('tanh')(output_layer)
    elif act_last == 3:
        # 多分类
        output_layer = Activation('softmax')(output_layer)

    return output_layer

def BridgeNet_1(num_dim=2, X_channels=1, Y_channels=1, kernel_size=3, pool_size_all=2, use_BN=False, kernels_all=[16,16],#None,
                 act_hide=1, act_last=0, pool_way=1, drop_rate=0, conv2act_repeat=1, dilation_rate=1):
    # ##########################################################################
    top_inputs, pool_size = input_block(num_dim, X_channels, pool_size_all)

    up_sampling_size = pool_size

    ####################################################################################################################
    conv01 = conv2act_block(num_dim, kernel_size, use_BN, kernels_all[0], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=top_inputs)
    down01 = pooling_block(num_dim, pool_size, pool_way, input_layer=conv01)
    ####################################################################################################################
    ####################################################################################################################
    ####################################################################################################################
    conv02 = conv2act_block(num_dim, kernel_size, use_BN, kernels_all[1], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=down01)
    up1 = up_concatenate_block(num_dim, up_sampling_size, up_layer=conv02, concatenate_layer=conv01)
    ####################################################################################################################
    conv03 = conv2act_block(num_dim, kernel_size, use_BN, kernels_all[0], act_hide, drop_rate, conv2act_repeat, dilation_rate, input_layer=up1)

    conv04 = output_block(num_dim, Y_channels, act_last, input_layer=conv03)

    model = Model(inputs=[top_inputs], outputs=[conv04])

    return model