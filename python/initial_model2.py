#!/usr/bin/env python3

from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout, Input, concatenate, merge, Add, Lambda
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, ZeroPadding2D,Activation
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
import os
import prepare_image

def transform_conv_weight(W):
    # for non FC layers, do this because Keras does convolution vs Caffe correlation
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j], 2)
    W = np.transpose(W,(3, 2, 1, 0))
    return W

def transform_fc_weight(W):
    return W.T


if __name__ == "__main__":
    input_shape = (3, 584, 565)
    K.set_image_data_format("channels_first")


    data_input = Input(shape=input_shape, name="data_input", dtype="float32")

    train_images = prepare_image.load_images(data_type="train", image_type="image")
    train_labels = prepare_image.load_images(data_type="train", image_type="label")
    test_images = prepare_image.load_images(data_type="test", image_type="image")
    test_labels = prepare_image.load_images(data_type="test", image_type="label")

    print(train_images.shape,train_images.shape[0])

    # load weights
    CAFFE_WEIGHTS_DIR = os.path.join(os.getcwd(), "model")

    W_conv1_1 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv1_1.npy")))
    b_conv1_1 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv1_1.npy"))
    W_conv1_2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv1_2.npy")))
    b_conv1_2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv1_2.npy"))

    W_conv2_1 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv2_1.npy")))
    b_conv2_1 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv2_1.npy"))
    W_conv2_2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv2_2.npy")))
    b_conv2_2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv2_2.npy"))

    W_conv3_1 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv3_1.npy")))
    b_conv3_1 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv3_1.npy"))
    W_conv3_2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv3_2.npy")))
    b_conv3_2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv3_2.npy"))
    W_conv3_3 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv3_3.npy")))
    b_conv3_3 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv3_3.npy"))

    W_conv4_1 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv4_1.npy")))
    b_conv4_1 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv4_1.npy"))
    W_conv4_2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv4_2.npy")))
    b_conv4_2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv4_2.npy"))
    W_conv4_3 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv4_3.npy")))
    b_conv4_3 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv4_3.npy"))

    W_conv1_2_16 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv1_2_16.npy")))
    b_conv1_2_16 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv1_2_16.npy"))
    W_conv2_2_16 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv2_2_16.npy")))
    b_conv2_2_16 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv2_2_16.npy"))
    W_conv3_3_16 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv3_3_16.npy")))
    b_conv3_3_16 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv3_3_16.npy"))
    W_conv4_3_16 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_conv4_3_16.npy")))
    b_conv4_3_16 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_conv4_3_16.npy"))

    W_upsample2 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_upsample2_.npy")))
    b_upsample2 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_upsample2_.npy"))
    W_upsample4 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_upsample4_.npy")))
    b_upsample4 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_upsample4_.npy"))
    W_upsample8 = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_upsample8_.npy")))
    b_upsample8 = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_upsample8_.npy"))

    W_weighting_av = transform_conv_weight(np.load(os.path.join(CAFFE_WEIGHTS_DIR, "W_new-score-weighting_av.npy")))
    b_weighting_Av = np.load(os.path.join(CAFFE_WEIGHTS_DIR, "b_new-score-weighting_av.npy"))

    # Convolution Layer 1
    conv1_1 = Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv1_1", padding="SAME"
                    ,weights = (W_conv1_1, b_conv1_1))(data_input)
    conv1_2 = Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv1_2", padding="SAME"
                     ,weights=(W_conv1_2, b_conv1_2))(conv1_1)
    max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool1', padding="SAME")(conv1_2)

    # Convolution Layer 2
    conv2_1 = Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv2_1", padding="SAME"
                     , weights=(W_conv2_1, b_conv2_1))(max_pool1)
    conv2_2 = Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv2_2", padding="SAME"
                     , weights=(W_conv2_2, b_conv2_2))(conv2_1)
    max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool2',padding="SAME")(conv2_2)


    # Convolution Layer3
    conv3_1 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_1", padding="SAME"
                     , weights=(W_conv3_1, b_conv3_1))(max_pool2)
    conv3_2 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_2", padding="SAME"
                     , weights=(W_conv3_2, b_conv3_2))(conv3_1)
    conv3_3 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_3", padding="SAME"
                     , weights=(W_conv3_3, b_conv3_3))(conv3_2)
    max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2),name='max_pool3', padding="SAME")(conv3_3)


    # Convolution Layer4
    conv4_1 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_1", padding="SAME"
                     , weights=(W_conv4_1, b_conv4_1))(max_pool3)
    conv4_2 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_2", padding="SAME"
                     , weights=(W_conv4_2, b_conv4_2))(conv4_1)
    conv4_3 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_3", padding="SAME"
                     , weights=(W_conv4_3, b_conv4_3))(conv4_2)

    #
    conv1_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv1_2_16", padding="SAME"
                        , weights=(W_conv1_2_16, b_conv1_2_16))(conv1_2)
    conv2_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv2_2_16", padding="SAME"
                        , weights=(W_conv2_2_16, b_conv2_2_16))(conv2_2)
    conv3_3_16 = Conv2D(16, kernel_size=(3, 3), name="conv3_2_16", padding="SAME"
                        , weights=(W_conv3_3_16, b_conv3_3_16))(conv3_3)
    conv4_3_16 = Conv2D(16, kernel_size=(3, 3), name="conv4_3_16", padding="SAME"
                        , weights=(W_conv4_3_16, b_conv4_3_16))(conv4_3)

    # Deconvolution Layer1
    side_multi2_up = Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2), padding="SAME", name="side_multi2_up"
                                     , weights=(W_upsample2, b_upsample2))(conv2_2_16)
    upside_multi2 = Cropping2D(cropping=((0, 0), (0, 1)), name="upside_multi2")(side_multi2_up)

    # Decovolution Layer2
    side_multi3_up = Conv2DTranspose(16, kernel_size=(8, 8), strides=(4, 4), padding="VALID", name="side_multi3_up"
                                     , weights=(W_upsample4, b_upsample4))(conv3_3_16)
    upside_multi3 = Cropping2D(cropping=((2, 2), (3, 4)), name="upside_multi3")(side_multi3_up)

    # Deconvolution Layer3
    side_multi4_up = Conv2DTranspose(16, kernel_size=(16, 16), strides=(8, 8), padding="VALID", name="side_multi4_up"
                                     , weights=(W_upsample8, b_upsample8))(conv4_3_16)
    upside_multi4 = Cropping2D(cropping=((4, 4), (5, 6)), name="upside_multi4")(side_multi4_up)

    # Specialized Layer
    concat_upscore = concatenate([conv1_2_16,upside_multi2,upside_multi3,upside_multi4], name="concat-upscore",axis=1)
    upscore_fuse = Conv2D(3, kernel_size=(1, 1), name="upscore_fuse"
                          , weights=(W_weighting_av, b_weighting_Av))(concat_upscore)

    #output = Lambda(output_lambda,output_shape=output_of_lambda)(upscore_fuse)

    model = Model(inputs=[data_input], outputs=[upscore_fuse])
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_images,train_labels,batch_size=10,epochs=1)
    t = model.predict(test_images,batch_size=10)
    print(type(t))
    #print(model.get_config())
    print(model.summary())
    #model.load_weights('model/3_class.h5')

    plot_model(model, "model.png")
    K.clear_session()
