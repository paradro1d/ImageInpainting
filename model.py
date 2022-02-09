from losses import valid_loss, hole_loss, perceptual_loss, style_loss, total_variation_loss
import tensorflow as tf
from partial_convolution import PartialConvolution as PConv
from partial_convolution import MaskApplication

def create_model():
    input_image = keras.Input(shape=(256, 256, 3))
    input_mask = keras.Input(shape=(256, 256, 1))
    input_masked = MaskApplication()([input_image, input_mask])
    pconv1_image, pconv1_mask = PConv(64, 7)([input_masked, input_mask])
    relu1 = Activation('relu')(pconv1_image)
    pconv2_image, pconv2_mask = PConv(128, 5)([relu1, pconv1_mask])
    bn2 = BatchNormalization()(pconv2_image)
    relu2 = Activation('relu')(bn2)
    pconv3_image, pconv3_mask = PConv(256, 3)([relu2, pconv2_mask])
    bn3 = BatchNormalization()(pconv3_image)
    relu3 = Activation('relu')(bn3)
    pconv4_image, pconv4_mask = PConv(512, 3)([relu3, pconv3_mask])
    bn4 = BatchNormalization()(pconv4_image)
    relu4 = Activation('relu')(bn4)
    pconv5_image, pconv5_mask = PConv(512, 3)([relu4, pconv4_mask])
    bn5 = BatchNormalization()(pconv5_image)
    relu5 = Activation('relu')(bn5)
    pconv6_image, _ = PConv(512, 3)([relu5, pconv5_mask])
    bn6 = BatchNormalization()(pconv6_image)
    relu6 = Activation('relu')(bn6)
    decode_image_upsamp0 = UpSampling(512)(relu6)
    decode_image_upsamp0 = Conv2D(512, 3, strides=1, padding='same')(decode_image_upsamp0)
    bn_up0 = BatchNormalization()(decode_image_upsamp0)
    relu_up0 = LeakyReLU(0.2)(bn_up0)
    concat0 = Concatenate()([relu_up0, relu5])
    concat0 = Conv2D(512, 3, padding='same', strides=1)(concat0)
    concat0 = BatchNormalization()(concat0)
    concat0 = LeakyReLU(0.2)(concat0)
    decode_image_upsamp1 = UpSampling(512)(concat0)
    decode_image_upsamp1 = Conv2D(512, 3, strides=1, padding='same')(decode_image_upsamp1)
    bn_up1 = BatchNormalization()(decode_image_upsamp1)
    relu_up1 = LeakyReLU(0.2)(bn_up1)
    concat1 = Concatenate()([relu_up1, relu4])
    concat1 = Conv2D(256, 3, padding='same', strides=1)(concat1)
    concat1 = BatchNormalization()(concat1)
    concat1 = LeakyReLU(0.2)(concat1)
    decode_image_upsamp2 = UpSampling(256)(concat1)
    decode_image_upsamp2 = Conv2D(256, 3, strides=1, padding='same')(decode_image_upsamp2)
    bn_up2 = BatchNormalization()(decode_image_upsamp2)
    relu_up2 = LeakyReLU(0.2)(bn_up2)
    concat2 = Concatenate()([relu_up2, relu3])
    concat2 = Conv2D(128, 3, padding='same', strides=1)(concat2)
    concat2 = BatchNormalization()(concat2)
    concat2 = LeakyReLU(0.2)(concat2)
    decode_image_upsamp3 = UpSampling(128)(concat2)
    decode_image_upsamp3 = Conv2D(128, 3, strides=1, padding='same')(decode_image_upsamp3)
    bn_up3 = BatchNormalization()(decode_image_upsamp3)
    relu_up3 = LeakyReLU(0.2)(bn_up3)
    concat3 = Concatenate()([relu_up3, relu2])
    concat3 = Conv2D(64, 3, padding='same', strides=1)(concat3)
    concat3 = BatchNormalization()(concat3)
    concat3 = LeakyReLU(0.2)(concat3)
    decode_image_upsamp4 = UpSampling(64)(concat3)
    decode_image_upsamp4 = Conv2D(3, 64, strides=1, padding='same')(decode_image_upsamp4)
    bn_up4 = BatchNormalization()(decode_image_upsamp4)
    relu_up4 = LeakyReLU(0.2)(bn_up4)
    concat4 = Concatenate()([relu_up4, relu1])
    concat4 = Conv2D(32, 3, padding='same', strides=1)(concat4)
    concat4 = BatchNormalization()(concat4)
    concat4 = LeakyReLU(0.2)(concat4)
    decode_image_upsamp5 = UpSampling(32)(concat4)
    concat5 = Concatenate()([decode_image_upsamp5, input_masked])
    output_image = Conv2D(3, 3, strides=1, dtype='float32', padding='same')(concat5)
    model = keras.Model(inputs=[input_image, input_mask], outputs=output_image)
    return model


class pconvmodel(keras.Model):
    def __init__(self):
        super(pconvmodel, self).__init__()
        self.model = create_model()

    def compile(self, optimizer):
        super(pconvmodel, self).compile()
        self.optimizer = optimizer

    def train_step(self, batch):
        images = batch[0]
        masks = batch[1]
        with tf.GradientTape() as tape:
            images_predicted = self.model((images, masks))
            hl = hole_loss(images, images_predicted, masks)
            vl = valid_loss(images, images_predicted, masks)
            pl = perceptual_loss(images, images_predicted, masks)
            sl = style_loss(images, images_predicted, masks)
            tl = total_variation_loss(images_predicted)
            loss = hl + vl + pl + sl + tl
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return {'Total loss':loss, 'Hole': hl, 'Valid': vl, 'Perceptual': pl, 'Style':sl, 'Total variation':tl}

