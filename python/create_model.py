#!/usr/bin/env python3

import os
import argparse
import numpy as np
import tensorflow as tf
# import caffe
import json
import prepare_image
import utility
import keras

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Flatten, Dropout, Input, concatenate, merge, Add, Lambda
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, ZeroPadding2D, Activation
from keras.layers import MaxPooling2D
from keras import backend as K
import keras.backend.tensorflow_backend as tfb
from keras.metrics import binary_accuracy
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD,Adam

K.set_image_data_format("channels_first")
# caffe.set_mode_cpu()
cur_dir = os.getcwd()
MODEL_PROTO = os.path.join(cur_dir, 'model', 'train.prototxt')
# MODEL_WEIGHTS = os.path.join(cur_dir, 'model', 'train_start.caffemodel')
# MODEL_WEIGHTS = os.path.join(cur_dir, 'model', '_iter_20000.caffemodel')

mapping = {
    'conv1_1': 'conv1_1',
    'conv1_2': 'conv1_2',
    'conv2_1': 'conv2_1',
    'conv2_2': 'conv2_2',
    'conv3_1': 'conv3_1',
    'conv3_2': 'conv3_2',
    'conv3_3': 'conv3_3',
    'conv4_1': 'conv4_1',
    'conv4_2': 'conv4_2',
    'conv4_3': 'conv4_3',
    'conv1_2_16': 'conv1_2_16',
    'conv2_2_16': 'conv2_2_16',
    'conv3_3_16': 'conv3_3_16',
    'conv4_3_16': 'conv4_3_16',
    'upsample2_': 'side_multi2_up',
    'upsample4_': 'side_multi3_up',
    'upsample8_': 'side_multi4_up',
    'new-score-weighting_av': 'upscore_fuse'
}


def sigmoid_cross_entropy_with_logits(target , output):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output)
    return tf.reduce_mean(loss,axis=-1)


def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor
    and a target tensor. POS_WEIGHT is used as a multiplier
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)


def parse_args():
    """
        function for argument parsing
        :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", "-c", help="Cache data wherever possible", action='store_true')
    parser.add_argument("--log_level", "-l", help="Set loglevel for debugging and analyis",
                         default="INFO")
    args = parser.parse_args()
    return args


class RetinaModel(object):
    def __init__(self):
        self.model = None
        self.input = None
        self.output = None

    def create_model(self):
        if args.cache and os.path.exists("cache/model.json"):
            with open("cache/model.json") as f:
                self.model = model_from_json(json.dumps(json.load(f)))
            return
            
        input_shape =(3, 584, 565)

        data_input = Input(shape=input_shape, name="data_input")
        conv1_1 = Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv1_1",
                          padding="SAME")(data_input)
        conv1_2 = Conv2D(64, kernel_size=(3, 3), activation='relu', name="conv1_2",
                          padding="SAME")(conv1_1)
        max_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool1',
                                  padding="SAME")(conv1_2)

        # Convolution Layer 2
        conv2_1 = Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv2_1",
                          padding="SAME")(max_pool1)
        conv2_2 = Conv2D(128, kernel_size=(3, 3), activation='relu', name="conv2_2",
                          padding="SAME")(conv2_1)
        max_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool2',
                                  padding="SAME")(conv2_2)

        # Convolution Layer3
        conv3_1 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_1",
                          padding="SAME")(max_pool2)
        conv3_2 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_2",
                          padding="SAME")(conv3_1)
        conv3_3 = Conv2D(256, kernel_size=(3, 3), activation='relu', name="conv3_3",
                          padding="SAME")(conv3_2)
        max_pool3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='max_pool3',
                                  padding="SAME")(conv3_3)

        # Convolution Layer4
        conv4_1 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_1",
                          padding="SAME")(max_pool3)
        conv4_2 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_2",
                          padding="SAME")(conv4_1)
        conv4_3 = Conv2D(512, kernel_size=(3, 3), activation='relu', name="conv4_3",
                          padding="SAME")(conv4_2)

        #
        conv1_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv1_2_16",
                             padding="SAME")(conv1_2)
        conv2_2_16 = Conv2D(16, kernel_size=(3, 3), name="conv2_2_16",
                             padding="SAME")(conv2_2)
        conv3_3_16 = Conv2D(16, kernel_size=(3, 3), name="conv3_3_16",
                             padding="SAME")(conv3_3)
        conv4_3_16 = Conv2D(16, kernel_size=(3, 3), name="conv4_3_16",
                             padding="SAME")(conv4_3)

        # Deconvolution Layer1
        side_multi2_up = Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2),
                                          padding="SAME", name="side_multi2_up")(conv2_2_16)
        upside_multi2 = Cropping2D(cropping=((0, 0),(0, 1)), name="upside_multi2")(side_multi2_up)

        # Decovolution Layer2
        side_multi3_up = Conv2DTranspose(16, kernel_size=(8, 8), strides=(4, 4),
                                          padding="VALID", name="side_multi3_up")(conv3_3_16)
        upside_multi3 = Cropping2D(cropping=((2, 2),(3, 4)), name="upside_multi3")(side_multi3_up)

        # Deconvolution Layer3
        side_multi4_up = Conv2DTranspose(16, kernel_size=(16, 16), strides=(8, 8),
                                          padding="VALID", name="side_multi4_up")(conv4_3_16)
        upside_multi4 = Cropping2D(cropping=((4, 4),(5, 6)), name="upside_multi4")(side_multi4_up)

        # Specialized Layer
        concat_upscore = concatenate([conv1_2_16, upside_multi2, upside_multi3, upside_multi4],
                                      name="concat-upscore", axis=1)
        upscore_fuse = Conv2D(3, kernel_size=(1, 1),activation='sigmoid', name="upscore_fuse")(concat_upscore)

        softmax_layer = Activation('softmax')(upscore_fuse)

        self.model = Model(inputs=[data_input], outputs=[softmax_layer])

        if args.cache:
            with open("cache/model.json", 'w') as json_file:
                json_model = self.model.to_json()
                json_file.write(json_model)

    def set_weights(self):
        if args.cache and os.path.exists("cache/keras_10000_model_weights.h5"):
            # self.model.load_weights("cache/model_weights.h5")
            self.model.load_weights("cache/keras_10000_model_weights.h5")
            return
        # net = caffe.Net(MODEL_PROTO, MODEL_WEIGHTS, caffe.TEST)
        for k, v in net.params.items():
            w = np.transpose(v[0].data, (2, 3, 1, 0))
            self.model.get_layer(name=mapping[k]).set_weights([w, v[1].data])
        if args.cache:
            self.model.save_weights(os.path.join('cache', 'model_weights.h5'))

    def get_data(self):
        if args.cache and os.path.exists('cache/image'):
            self.train_images = np.load('cache/image/train_images.npy')
            self.train_labels = np.load('cache/image/train_labels.npy')
            self.test_images = np.load('cache/image/test_images.npy')
            self.test_labels = np.load('cache/image/test_labels.npy')
            return

        self.train_images = prepare_image.load_images(data_type="train", image_type="image")
        self.train_labels = prepare_image.load_images(data_type="train", image_type="label")
        self.test_images = prepare_image.load_images(data_type="test", image_type="image")
        self.test_labels = prepare_image.load_images(data_type="test", image_type="label")
        if args.cache:
            utility.create_directory('cache/image')
            np.save('cache/image/train_images.npy', self.train_images)
            np.save('cache/image/train_labels.npy', self.train_labels)
            np.save('cache/image/test_images.npy', self.test_images)
            np.save('cache/image/test_labels.npy', self.test_labels)

    def run(self):
        print(self.train_images.shape)
        sgd = SGD(lr=1e-8, decay=1e-6, momentum=0.9, nesterov=True)
        """
        weight_save_callback = keras.callback.ModelCheckpoint('/model/weights.hdf5', monitor='val_loss',
                                                verbose=0, save_best_only=True, mode='auto')
        self.model.compile(optimizer=sgd, loss=sigmoid_cross_entropy_with_logits,
                           metrics=['accuracy'], callbacks=[weight_save_callback])
        """
        self.model.compile(optimizer=sgd, loss=sigmoid_cross_entropy_with_logits,
                            metrics=['accuracy'])
        self.model.fit(self.train_images, self.train_labels, batch_size=10, epochs=100)
        test_predict = self.model.predict(self.test_images, batch_size=10)
        print(test_predict[0])
        # np.save('cache/test_predict.npy', test_predict)
        # self.model.save_weights(os.path.join('cache', 'keras_15000_model_weights.h5'))


if __name__ == '__main__':
    args = parse_args()
    rm = RetinaModel()
    rm.create_model()
    rm.set_weights()
    rm.get_data()
    # plot_model(rm.model,"model.png")
    # rm.run()
    K.clear_session()
