from tensorflow import keras
import tensorflow as tf


class PartialConvolution(keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=2):
        super(PartialConvolution, self).__init__()
        self.picture_convolution = keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False)
        self.mask_convolution = keras.layers.Conv2D(1, kernel_size, strides=strides, padding='same', kernel_initializer = keras.initializers.Ones(), use_bias=False)
        self.mask_convolution.trainable = False
        self.bias = self.add_weight('bias', shape=[filters], trainable=True, dtype=tf.float32)
        self.kernel_size = kernel_size

    def call(self, inputs):
        [input_tensor, mask_tensor] = inputs
        inp_conved = self.picture_convolution(input_tensor*mask_tensor)
        mask_conved = self.mask_convolution(mask_tensor)
        out_picture = tf.math.divide_no_nan(inp_conved, mask_conved)*self.kernel_size*self.kernel_size
        out_picture = out_picture + self.bias
        out_mask = tf.math.sign(mask_conved)
        return [out_picture, out_mask]

class MaskApplication(keras.layers.Layer):
    def call(self, inputs):
        [image, mask] = inputs
        return image*mask
