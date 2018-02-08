#!/usr/bin/env python3

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, concatenate, merge, Add, Lambda
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, ZeroPadding2D,Activation
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf



if __name__ == "__main__":
    input_shape = (1, 584, 565)
    K.set_image_data_format("channels_first")

    data_input = Input(shape=input_shape, name="data_input", dtype="float32")

    # Convolution Layer 1
    conv1_1 = Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv1_1", padding="SAME")(data_input)
    conv1_2 = Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv1_2", padding="SAME")(conv1_1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool1', padding="SAME")(conv1_2)

    # Convolution Layer 2
    conv2_1 = Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv2_1", padding="SAME")(max_pool1)
    conv2_2 = Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv2_2", padding="SAME")(conv2_1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool2',padding="SAME")(conv2_2)


    # Convolution Layer3
    conv3_1 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_1", padding="SAME")(max_pool2)
    conv3_2 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_2", padding="SAME")(conv3_1)
    conv3_3 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_3", padding="SAME")(conv3_2)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name='max_pool3', padding="SAME")(conv3_3)


    # Convolution Layer4
    conv4_1 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_1", padding="SAME")(max_pool3)
    conv4_2 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_2", padding="SAME")(conv4_1)
    conv4_3 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_3", padding="SAME")(conv4_2)

    #
    conv1_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv1_2_16", padding="SAME")(conv1_2)
    conv2_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv2_2_16", padding="SAME")(conv2_2)
    conv3_3_16 = Conv2D(16, kernel_size=(3, 3), name="conv3_2_16", padding="SAME")(conv3_3)
    conv4_3_16 = Conv2D(16, kernel_size=(3, 3), name="conv4_3_16", padding="SAME")(conv4_3)

    # Deconvolution Layer1
    side_multi2_up = Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="side_multi2_up")(conv2_2_16)
    upside_multi2 = Cropping2D(cropping=((0, 0), (0, 1)), name="upside_multi2")(side_multi2_up)

    # Decovolution Layer2
    side_multi3_up = Conv2DTranspose(16, kernel_size=(8, 8), strides=(4, 4), padding="VALID", name="side_multi3_up")(conv3_3_16)
    upside_multi3 = Cropping2D(cropping=((2, 2), (3, 4)), name="upside_multi3")(side_multi3_up)

    # Deconvolution Layer3
    side_multi4_up = Conv2DTranspose(16, kernel_size=(16, 16), strides=(8, 8), padding="VALID", name="side_multi4_up")(conv4_3_16)
    upside_multi4 = Cropping2D(cropping=((4, 4), (5, 6)), name="upside_multi4")(side_multi4_up)

    # Specialized Layer
    concat_upscore = concatenate([conv1_2_16,upside_multi2,upside_multi3,upside_multi4], name="concat-upscore",axis=1)
    upscore_fuse = Conv2D(3, kernel_size=(1, 1), name="upscore_fuse")(concat_upscore)



    model = Model(inputs=[data_input], outputs=[upscore_fuse])
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])


    print(model.get_config())
    print(model.summary())
    model.load_weights('3_class.h5')

