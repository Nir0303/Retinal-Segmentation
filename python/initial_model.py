#!/usr/bin/env python3

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, concatenate, merge, Add, Lambda
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, ZeroPadding2D
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf



if __name__ == "__main__":
    input_shape = (1, 584, 565)
    K.set_image_data_format("channels_first")

    data_input = Input(shape=input_shape, name="data_input", dtype="float32")

    #Convolution Layer 1
    conv1_1 = Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv1_1" , padding= "SAME")(data_input)
    conv1_2 = Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv1_2", padding= "SAME")(conv1_1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool1', padding="SAME")(conv1_2)
    max_pool1 = ZeroPadding2D((146, 141))(max_pool1)

    #Convolution Layer 2
    conv2_1 = Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv2_1", padding="SAME")(max_pool1)
    conv2_2 = Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv2_2", padding="SAME")(conv2_1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2) , padding="SAME")(conv2_2)
    max_pool2 = ZeroPadding2D((146, 141))(max_pool2)

    #Convolution Layer3
    conv3_1 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_1",padding="SAME")(max_pool2)
    conv3_2 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_2", padding="SAME")(conv3_1)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME")(conv3_2)
    max_pool3 = ZeroPadding2D((146, 141))(max_pool3)

    #Convolution Layer4
    conv4_1 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_1", padding="SAME")(max_pool3)
    conv4_2 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_2", padding="SAME")(conv4_1)
    conv4_3 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_3", padding="SAME")(conv4_2)

    #
    conv1_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv1_2_16", padding="SAME")(conv1_2)
    conv2_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv2_2_16" ,padding="SAME")(conv2_2)
    conv3_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv3_2_16", padding="SAME")(conv3_2)
    conv4_3_16 = Conv2D(16, kernel_size=(3, 3), name="conv4_3_16", padding="SAME")(conv4_3)

    #Deconvolution Layer1
    upsample2_ = Conv2DTranspose(16, kernel_size=(4, 4),strides=(2, 2), padding="SAME",name="upsample2_")(conv2_2_16)
    crop2 = Cropping2D(cropping = ((292, 292),(282,283)),name="crop2")(upsample2_)

    #Decovolution Layer2
    upsample4_ = Conv2DTranspose(16, kernel_size=(8, 8), strides=(4, 4),padding="SAME", name="upsample4_")(conv3_2_16)
    crop4 = Cropping2D(cropping=((876, 876), (847, 848)),name="crop4")(upsample4_)

    #Deconvolution Layer3
    upsample8_ = Conv2DTranspose(16, kernel_size=(16, 16), strides=(8, 8),padding="VALID", name="upsample8_")(conv4_3_16)
    crop8 = Cropping2D(cropping=((2048, 2048), (1981, 1982)),name="crop8")(upsample8_)

    #Specialized Layer
    concat_layer = concatenate([conv1_2_16,crop2, crop4, crop8], name="concat_layer")
    weighting_av = Conv2D(3, kernel_size=(1, 1), name="weighting_av")(concat_layer)

    model = Model(inputs=[data_input], outputs=[concat_layer,weighting_av])
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    plot_model(model, to_file='model.png')

    print(model.get_config())
    print(model.summary())

    model.load_weights('3_class.h5')
