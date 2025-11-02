import tensorflow as tf
import keras
from tensorflow.keras import layers
from msia_BN_3M_tf2 import msia_3_M
from tensorflow.keras.layers import Activation


def lrelu(x):
    """Leaky ReLU activation function"""
    return tf.nn.leaky_relu(x, alpha=0.2)

def restoration_loss(y_true, y_pred):
    # SSIM loss per channel
    ssim_1 = tf.image.ssim(y_pred[:, :, :, 0:1], y_true[:, :, :, 0:1], max_val=1.0)
    ssim_2 = tf.image.ssim(y_pred[:, :, :, 1:2], y_true[:, :, :, 1:2], max_val=1.0)
    ssim_3 = tf.image.ssim(y_pred[:, :, :, 2:3], y_true[:, :, :, 2:3], max_val=1.0)
    loss_ssim = 1 - (ssim_1 + ssim_2 + ssim_3) / 3.0

    # Take mean across batch for SSIM
    ssim_mean = (tf.reduce_mean(ssim_1) + tf.reduce_mean(ssim_2) + tf.reduce_mean(ssim_3)) / 3.0
    loss_ssim = 1 - ssim_mean

    # MSE loss
    loss_mse = tf.reduce_mean(tf.square(y_pred - y_true))

    return loss_mse + loss_ssim


def upsample_and_concat(x1, x2, output_channels, in_channels, scope_name):
    """
    Upsample x1 and concatenate with x2.

    Args:
        x1: Lower resolution feature map to upsample
        x2: Higher resolution feature map to concatenate
        output_channels: Number of output channels after upsampling
        in_channels: Number of input channels from x1
        scope_name: Name for the operation

    Returns:
        Concatenated feature map with 2*output_channels
    """
    pool_size = 2

    # Use functional Conv2DTranspose
    deconv = layers.Conv2DTranspose(
        filters=output_channels,
        kernel_size=pool_size,
        strides=pool_size,
        padding='same',
        name=f'{scope_name}_deconv'
    )(x1)

    # Concatenate along channel axis
    deconv_output = layers.Concatenate(axis=-1)([deconv, x2])

    return deconv_output


def DecomNet(input_tensor, name='DecomNet'):
    """
    Decomposition Network: Decomposes input into Reflectance (R) and Illumination (L).
    Uses prefixed layer names to avoid duplicate name conflicts when called twice.
    """
    # Use a unique prefix for each model instance
    prefix = name + '_'

    # Encoder path
    conv1 = layers.Conv2D(32, 3, padding='same', activation=lrelu, name=prefix+'g_conv1_1')(input_tensor)
    pool1 = layers.MaxPooling2D(2, strides=2, padding='same', name=prefix+'g_pool1')(conv1)

    conv2 = layers.Conv2D(64, 3, padding='same', activation=lrelu, name=prefix+'g_conv2_1')(pool1)
    pool2 = layers.MaxPooling2D(2, strides=2, padding='same', name=prefix+'g_pool2')(conv2)

    conv3 = layers.Conv2D(128, 3, padding='same', activation=lrelu, name=prefix+'g_conv3_1')(pool2)

    # Decoder path
    up8 = upsample_and_concat(conv3, conv2, 64, 128, prefix+'g_up_1')
    conv8 = layers.Conv2D(64, 3, padding='same', activation=lrelu, name=prefix+'g_conv8_1')(up8)

    up9 = upsample_and_concat(conv8, conv1, 32, 64, prefix+'g_up_2')
    conv9 = layers.Conv2D(32, 3, padding='same', activation=lrelu, name=prefix+'g_conv9_1')(up9)

    # Reflectance output (1x1 conv)
    conv10 = layers.Conv2D(3, 1, padding='same', activation=None, name=prefix+'g_conv10')(conv9)
    R_out = Activation('sigmoid',name =prefix+'sigmoid')(conv10)

    # Illumination branch
    l_conv2 = layers.Conv2D(32, 3, padding='same', activation=lrelu, name=prefix+'l_conv1_2')(conv1)
    l_conv3 = layers.Concatenate(axis=-1)([l_conv2, conv9])
    l_conv4 = layers.Conv2D(1, 1, padding='same', activation=None, name=prefix+'l_conv1_4')(l_conv3)
    L_out =  Activation('sigmoid',name =prefix+'sigmoid1')(l_conv4)

    return R_out, L_out


def Restoration_net(input_r, input_i, training=True):
    """
    Restoration Network: Removes noise from reflectance using MSIA modules.

    Args:
        input_r: Input reflectance map
        input_i: Input illumination map
        training: Boolean flag for training mode

    Returns:
        out: Denoised reflectance map
    """
    # Stage 1
    conv1 = layers.Conv2D(32, 3, padding='same', activation=lrelu, name='de_conv1_1')(input_r)
    conv1 = layers.Conv2D(64, 3, padding='same', activation=lrelu, name='de_conv1_2')(conv1)
    msia_1 = msia_3_M(conv1, input_i, name='de_conv1', training=training)

    # Stage 2
    conv2 = layers.Conv2D(128, 3, padding='same', activation=lrelu, name='de_conv2_1')(msia_1)
    conv2 = layers.Conv2D(256, 3, padding='same', activation=lrelu, name='de_conv2_2')(conv2)
    msia_2 = msia_3_M(conv2, input_i, name='de_conv2', training=training)

    # Stage 3
    conv3 = layers.Conv2D(512, 3, padding='same', activation=lrelu, name='de_conv3_1')(msia_2)
    conv3 = layers.Conv2D(256, 3, padding='same', activation=lrelu, name='de_conv3_2')(conv3)
    msia_3 = msia_3_M(conv3, input_i, name='de_conv3', training=training)

    # Stage 4
    conv4 = layers.Conv2D(128, 3, padding='same', activation=lrelu, name='de_conv4_1')(msia_3)
    conv4 = layers.Conv2D(64, 3, padding='same', activation=lrelu, name='de_conv4_2')(conv4)
    msia_4 = msia_3_M(conv4, input_i, name='de_conv4', training=training)

    # Stage 5 - Output
    conv5 = layers.Conv2D(32, 3, padding='same', activation=lrelu, name='de_conv5_1')(msia_4)
    conv10 = layers.Conv2D(3, 3, padding='same', activation=None, name='de_conv10')(conv5)
    out = Activation('sigmoid', name='sigmoid')(conv10)

    return out


def Illumination_adjust_net(input_i, input_ratio):
    """
    Illumination Adjustment Network: Enhances illumination based on adjustment ratio.

    Args:
        input_i: Input illumination map
        input_ratio: Desired illumination adjustment ratio

    Returns:
        L_enhance: Enhanced illumination map
    """
    # Concatenate illumination and ratio
    input_all = layers.Concatenate(axis=-1)([input_i, input_ratio])

    # Processing layers
    conv1 = layers.Conv2D(32, 3, padding='same', activation=tf.nn.leaky_relu, name='conv_1')(input_all)
    conv2 = layers.Conv2D(32, 3, padding='same', activation=tf.nn.leaky_relu, name='conv_2')(conv1)
    conv3 = layers.Conv2D(32, 3, padding='same', activation=tf.nn.leaky_relu, name='conv_3')(conv2)
    conv4 = layers.Conv2D(1, 3, padding='same', activation=tf.nn.leaky_relu, name='conv_4')(conv3)

    L_enhance = Activation('sigmoid',name='sigmoid')(conv4)

    return L_enhance