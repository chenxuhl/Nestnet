# -*- coding:utf-8 -*-
# Author : Ray
# Data : 2019/7/23 4:53 PM

from keras.models import *
from keras.layers import *
from keras.optimizers import *

IMAGE_SIZE = 512


def unet(input_size=(IMAGE_SIZE, IMAGE_SIZE, 3), num_class=2):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    conv1 = Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = LeakyReLU(alpha=0.3)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    conv2 = Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = LeakyReLU(alpha=0.3)(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    conv3 = Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = LeakyReLU(alpha=0.3)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    conv4 = Conv2D(512, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = LeakyReLU(alpha=0.3)(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation=None, padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = LeakyReLU(alpha=0.3)(conv5)
    conv5 = Conv2D(1024, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv5)
    conv5 = LeakyReLU(alpha=0.3)(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation=None, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    up6 = LeakyReLU(alpha=0.3)(up6)
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation=None, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = LeakyReLU(alpha=0.3)(conv6)
    conv6 = Conv2D(512, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = LeakyReLU(alpha=0.3)(conv6)

    up7 = Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    up7 = LeakyReLU(alpha=0.3)(up7)
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = LeakyReLU(alpha=0.3)(conv7)
    conv7 = Conv2D(256, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = LeakyReLU(alpha=0.3)(conv7)

    up8 = Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    up8 = LeakyReLU(alpha=0.3)(up8)
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)
    conv8 = Conv2D(128, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv8)
    conv8 = LeakyReLU(alpha=0.3)(conv8)

    up9 = Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    up9 = LeakyReLU(alpha=0.3)(up9)
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = LeakyReLU(alpha=0.3)(conv9)
    conv9 = Conv2D(64, 3, activation=None, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = LeakyReLU(alpha=0.3)(conv9)

    if num_class == 2:
        conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
        loss_function = 'binary_crossentropy'
    else:
        conv10 = Conv2D(num_class, 1, activation='softmax')(conv9)
        loss_function = 'categorical_crossentropy'

    # conv10 = Conv2D(3,1,activation='softamx')(conv9)

    model = Model(input=inputs, output=conv10)

    return model, loss_function


def standard_unit(input_tensor, nb_filter, dropout_rate, kernel_size=3):
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation='relu', kernel_initializer='he_normal',
               padding='same')(input_tensor)
    x = Dropout(dropout_rate)(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation='relu', kernel_initializer='he_normal',
               padding='same')(x)
    x = Dropout(dropout_rate)(x)

    return x


def nestnet(input_size=(IMAGE_SIZE, IMAGE_SIZE, 3), num_class=2, dropout_rate=0.5, deep_supervision=False):
    img_input = Input(input_size)
    nb_filter = [32, 64, 128, 256, 512]

    conv1_1 = standard_unit(img_input, nb_filter=nb_filter[0], dropout_rate=dropout_rate)
    pool1 = MaxPooling2D((2, 2), strides=(2, 2))(conv1_1)

    conv2_1 = standard_unit(pool1, nb_filter=nb_filter[1], dropout_rate=dropout_rate)
    pool2 = MaxPooling2D((2, 2), strides=(2, 2))(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1])
    conv1_2 = standard_unit(conv1_2, nb_filter=nb_filter[0], dropout_rate=dropout_rate)

    conv3_1 = standard_unit(pool2, nb_filter=nb_filter[2], dropout_rate=dropout_rate)
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1])
    conv2_2 = standard_unit(conv2_2, nb_filter=nb_filter[1], dropout_rate=dropout_rate)

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2])
    conv1_3 = standard_unit(conv1_3, nb_filter=nb_filter[0], dropout_rate=dropout_rate)

    conv4_1 = standard_unit(pool3, nb_filter=nb_filter[3], dropout_rate=dropout_rate)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2))(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1])
    conv3_2 = standard_unit(conv3_2, nb_filter=nb_filter[2], dropout_rate=dropout_rate)

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2])
    conv2_3 = standard_unit(conv2_3, nb_filter=nb_filter[1], dropout_rate=dropout_rate)

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3])
    conv1_4 = standard_unit(conv1_4, nb_filter=nb_filter[0], dropout_rate=dropout_rate)

    conv5_1 = standard_unit(pool4, nb_filter=nb_filter[4], dropout_rate=dropout_rate)

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1])
    conv4_2 = standard_unit(conv4_2, nb_filter=nb_filter[3], dropout_rate=dropout_rate)

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2])
    conv3_3 = standard_unit(conv3_3, nb_filter=nb_filter[2], dropout_rate=dropout_rate)

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3])
    conv2_4 = standard_unit(conv2_4, nb_filter=nb_filter[1], dropout_rate=dropout_rate)

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4])
    conv1_5 = standard_unit(conv1_5, nb_filter=nb_filter[0], dropout_rate=dropout_rate)

    if num_class == 1:
        nestnet_output_1 = Conv2D(num_class, (1, 1), activation='sigmoid', kernel_initializer='he_normal',
                                  padding='same')(conv1_2)
        nestnet_output_2 = Conv2D(num_class, (1, 1), activation='sigmoid', kernel_initializer='he_normal',
                                  padding='same')(conv1_3)
        nestnet_output_3 = Conv2D(num_class, (1, 1), activation='sigmoid', kernel_initializer='he_normal',
                                  padding='same')(conv1_4)
        nestnet_output_4 = Conv2D(num_class, (1, 1), activation='sigmoid', kernel_initializer='he_normal',
                                  padding='same')(conv1_5)
        loss_function = 'binary_crossentropy'

    else:
        nestnet_output_1 = Conv2D(num_class, (1, 1), activation='softmax', kernel_initializer='he_normal',
                                  padding='same')(conv1_2)
        nestnet_output_2 = Conv2D(num_class, (1, 1), activation='softmax', kernel_initializer='he_normal',
                                  padding='same')(conv1_3)
        nestnet_output_3 = Conv2D(num_class, (1, 1), activation='softmax', kernel_initializer='he_normal',
                                  padding='same')(conv1_4)
        nestnet_output_4 = Conv2D(num_class, (1, 1), activation='softmax', kernel_initializer='he_normal',
                                  padding='same')(conv1_5)
        loss_function = 'categorical_crossentropy'

    if deep_supervision:
        model = Model(input=img_input, output=[nestnet_output_1,
                                               nestnet_output_2,
                                               nestnet_output_3,
                                               nestnet_output_4])
    else:
        model = Model(input=img_input, output=[nestnet_output_4])

    return model, loss_function

