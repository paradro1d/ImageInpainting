from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.applications import vgg16

VALID_LOSS = 1
HOLE_LOSS = 6
PERCEPTUAL_LOSS = 0.05
STYLE_LOSS = 120
TOTAL_VARIATION_REGULARIZATION = 0.1
INPUT_SHAPE = (256, 256, 3)

vgg_loaded = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE, pooling='max')
vgg_loaded.trainable = False
layers = ['block1_pool', 'block2_pool', 'block3_pool']
outputs = [vgg_loaded.get_layer(layer).output for layer in layers]
vgg = keras.Model([vgg_loaded.input], outputs)


def Gaussian_kernel(channels, kernel_size, sigma):
    ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
    xx, yy = tf.meshgrid(ax, ax)
    kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel = kernel / tf.reduce_sum(kernel)
    kernel = tf.tile(kernel[..., tf.newaxis], [1, 1, channels])
    return kernel

gau_ker = Gaussian_kernel(3, 11, 0.93)[..., tf.newaxis]

def gaussian_blur(img):
    return tf.nn.depthwise_conv2d(img, gau_ker, [1, 1, 1, 1],
            padding='SAME', data_format='NHWC')


def hole_loss(true, predicted, mask, coefficient=HOLE_LOSS):
    true = gaussian_blur(true)
    predicted = gaussian_blur(true)
    n = tf.cast(tf.size(true), tf.float32)
    return coefficient*tf.norm((1 - mask)*(predicted - true), ord=1)/n

def valid_loss(true, predicted, mask, coefficient=VALID_LOSS):
    n = tf.cast(tf.size(true), tf.float32)
    return coefficient*tf.norm((mask*predicted - true), ord=1)/n

def perceptual_loss(true, predicted, mask, coefficient=PERCEPTUAL_LOSS):
    output = 0
    feature_maps_predicted = vgg(predicted)
    feature_maps_true = vgg(true)
    feature_maps_comp = vgg(true*mask + (1 - mask)*predicted)
    for maps in zip(feature_maps_predicted, feature_maps_true, feature_maps_comp):
        n = tf.cast(tf.size(maps[0]), tf.float32)
        output += tf.norm(maps[0] - maps[1], ord=1)/n
        output += tf.norm(maps[2] - maps[1], ord=1)/n
    return coefficient*output

def autocorr_matrix(x):
    return tf.einsum('njim, njik->nmk', x, x)

def style_loss(true, predicted, mask, coefficient=STYLE_LOSS):
    output = 0
    feature_maps_predicted = vgg(predicted)
    feature_maps_true = vgg(true)
    feature_maps_comp = vgg(true*mask + (1 - mask)*predicted)
    for maps in zip(feature_maps_predicted, feature_maps_true, feature_maps_comp):
        n = tf.cast(tf.size(maps[0]), tf.float32)
        for mp in maps:
            mp = tf.einsum('njim, njik->nmk', mp, mp)
        k = tf.cast(tf.shape(maps[0])[-1], tf.float32)
        output += tf.norm(maps[0] - maps[1], ord=1)/(n*k)
        output += tf.norm(maps[2] - maps[1], ord=1)/(n*k)
    return output*coefficient

def total_variation_loss(predicted, coefficient=TOTAL_VARIATION_REGULARIZATION):
    n = tf.cast(tf.size(predicted), tf.float32)
    return tf.reduce_sum(tf.image.total_variation(predicted))/n
