#!/usr/bin/env python3

import os
import argparse
import numpy as np
import tensorflow as tf
# import caffe
import json
import prepare_image
import utility

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Flatten, Dropout, Input, concatenate, merge, Add, Lambda
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, ZeroPadding2D, Activation
from keras.layers import MaxPooling2D
from keras import backend as K
from keras.metrics import binary_accuracy
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences

K.set_image_data_format("channels_first")
# caffe.set_mode_cpu()
cur_dir = os.getcwd()
MODEL_PROTO = os.path.join(cur_dir, 'model', 'train.prototxt')
MODEL_WEIGHTS = os.path.join(cur_dir, 'model', 'train_start.caffemodel')

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
        input_shape = (3, 584, 565)

        data_input = Input(shape=input_shape, name="data_input", dtype="float32")
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
        upscore_fuse = Conv2D(3, kernel_size=(1, 1), name="upscore_fuse")(concat_upscore)

        self.model = Model(inputs=[data_input], outputs=[upscore_fuse])

        if args.cache:
            with open("cache/model.json", 'w') as json_file:
                json_model = self.model.to_json()
                json_file.write(json_model)

    def set_weights(self):
        if args.cache and os.path.exists("cache/model_weights.h5"):
            self.model.load_weights("cache/model_weights.h5")
            return
        # net = caffe.Net(MODEL_PROTO, MODEL_WEIGHTS, caffe.TEST)
        for k, v in net.params.items():
            w = np.transpose(v[0].data, (2, 3, 1, 0))
            self.model.get_layer(name=mapping[k]).set_weights([w, v[1].data])
        if args.cache:
            self.model.save_weights(os.path.join('cache', 'model_weights.h5'))

    def validation_data(self):
        train_images = np.load('cache/image/train_images.npy')
        train_labels = np.load('cache/image/train_labels.npy')

        for i in range(10):
            for image,label in zip(train_images,train_labels):
                yield image.reshape(1, *image.shape),label.reshape(1, *image.shape)

    def get_data(self):
        if args.cache and os.path.exists('cache/image'):
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
        validation_data = self.validation_data()

        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy',
                           metrics=['accuracy'],)
        self.model.fit_generator(validation_data, steps_per_epoch=2, epochs=1, workers=3)
        test_predict = self.model.predict(self.test_images, batch_size=10)
        test_accuracy = binary_accuracy(self.test_labels, test_predict)
        print(test_accuracy)
        np.save('cache/test_predict.npy', test_predict)


if __name__ == '__main__':
    args = parse_args()
    rm = RetinaModel()
    rm.create_model()
    rm.set_weights()
    rm.get_data()
    rm.run()
    K.clear_session()
