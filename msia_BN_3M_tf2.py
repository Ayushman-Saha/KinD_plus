import tensorflow as tf
from keras.src.layers import Activation
from tensorflow.keras import layers
from tensorflow import keras


def illu_attention_3_M(input_feature, input_i, name):
    """
    Illumination attention mechanism that applies sigmoid-activated
    convolution as attention weights.
    """
    kernel_size = 3

    # Apply convolution to input_i
    concat = layers.Conv2D(
        filters=1,
        kernel_size=kernel_size,
        strides=1,
        padding="same",
        activation=None,
        use_bias=False,
        name=f'{name}_conv'
    )(input_i)

    # Apply sigmoid activation
    concat = Activation('sigmoid', name=f"{name}_sigmoid")(concat)

    # Element-wise multiplication (attention)
    return input_feature * concat


def pool_upsampling_3_M(input_feature, level, training, name):
    """
    Pool-upsampling module with different pooling levels.
    """
    num_filters = input_feature.shape[-1]

    if level == 1:
        # Direct convolution without pooling
        pu_conv = layers.Conv2D(
            num_filters,
            kernel_size=3,
            strides=1,
            padding='same',
            name=f'{name}_pu_conv'
        )(input_feature)
        pu_conv = layers.BatchNormalization(name=f'{name}_bn')(pu_conv, training=training)
        conv_up = layers.ReLU(name=f'{name}_relu')(pu_conv)

    elif level == 2:
        # 2x2 max pooling followed by upsampling
        pu_net = layers.MaxPooling2D(
            pool_size=2,
            strides=2,
            padding='same',
            name=f'{name}_pu_net'
        )(input_feature)
        pu_conv = layers.Conv2D(
            num_filters,
            kernel_size=3,
            strides=1,
            padding='same',
            name=f'{name}_pu_conv'
        )(pu_net)
        pu_conv = layers.BatchNormalization(name=f'{name}_bn')(pu_conv, training=training)
        pu_conv = layers.ReLU(name=f'{name}_relu')(pu_conv)
        conv_up = layers.Conv2DTranspose(
            num_filters,
            kernel_size=2,
            strides=2,
            padding='same',
            name=f'{name}_conv_up'
        )(pu_conv)

    elif level == 4:
        # 4x4 max pooling followed by 2-stage upsampling
        pu_net = layers.MaxPooling2D(
            pool_size=4,
            strides=4,
            padding='same',
            name=f'{name}_pu_net'
        )(input_feature)
        pu_conv = layers.Conv2D(
            num_filters,
            kernel_size=1,
            strides=1,
            padding='same',
            name=f'{name}_pu_conv'
        )(pu_net)
        pu_conv = layers.BatchNormalization(name=f'{name}_bn')(pu_conv, training=training)
        pu_conv = layers.ReLU(name=f'{name}_relu')(pu_conv)
        conv_up_1 = layers.Conv2DTranspose(
            num_filters,
            kernel_size=2,
            strides=2,
            padding='same',
            name=f'{name}_conv_up_1'
        )(pu_conv)
        conv_up = layers.Conv2DTranspose(
            num_filters,
            kernel_size=2,
            strides=2,
            padding='same',
            name=f'{name}_conv_up'
        )(conv_up_1)
    else:
        raise ValueError(f"Unsupported level: {level}. Use 1, 2, or 4.")

    return conv_up


def Multi_Scale_Module_3_M(input_feature, training, name):
    """
    Multi-scale module that combines features at different scales.
    """
    # Generate features at different scales
    Scale_1 = pool_upsampling_3_M(input_feature, 1, training, name=f'{name}_pu1')
    Scale_2 = pool_upsampling_3_M(input_feature, 2, training, name=f'{name}_pu2')
    Scale_4 = pool_upsampling_3_M(input_feature, 4, training, name=f'{name}_pu4')

    # Concatenate all scales
    # res = tf.concat([input_feature, Scale_1, Scale_2, Scale_4], axis=-1)
    res = layers.Concatenate(axis=-1)([input_feature, Scale_1, Scale_2, Scale_4])

    # Project back to original number of channels
    multi_scale_feature = layers.Conv2D(
        input_feature.shape[-1],
        kernel_size=1,
        strides=1,
        padding='same',
        name=f'{name}_multi_scale_feature'
    )(res)

    return multi_scale_feature


def msia_3_M(input_feature, input_i, name, training):
    """
    Multi-Scale Illumination Attention module.
    Combines spatial attention with multi-scale feature extraction.
    """
    # Apply illumination-based spatial attention
    spatial_attention_feature = illu_attention_3_M(input_feature, input_i, name)

    # Apply multi-scale module
    msia_feature = Multi_Scale_Module_3_M(spatial_attention_feature, training, name)

    return msia_feature
