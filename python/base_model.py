#!/usr/bin/env python3
import sys
import os
import numpy as np
import tensorflow as tf
import json
import prepare_image
import utility
import h5py
import keras

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Flatten, Dropout, Input, concatenate, merge, Add, Dropout
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, ZeroPadding2D, Activation
from keras.layers import MaxPooling2D, UpSampling2D, Permute
from keras import backend as K
from keras.activations import softmax
import keras.backend.tensorflow_backend as tfb
from keras.utils import plot_model
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD,Adam
from time import time
from generate_metrics import *

K.set_image_data_format("channels_first")
cur_dir = os.getcwd()
pylon5 = os.environ["SCRATCH"] if os.environ.get("SCRATCH", None) else "."
pylon5_cache = os.path.join(pylon5, 'cache')

def image_accuracy(y_true, y_pred):
    with tf.name_scope("ImageAccuracy"):
        X_sigmoid = tf.nn.sigmoid(y_true, name="Sigmoid")
        X_softmax = tf.nn.softmax(X_sigmoid, axis=1, name="Softmax")
        verify = tf.cast(tf.equal(tf.argmax(X_softmax, axis=1),
                                  tf.argmax(y_pred, axis=1), name="Compare"),
                         dtype=tf.float32, name="Cast")
        accuracy = tf.reduce_mean(verify, name="Accuracy")
        return accuracy


def sigmoid_cross_entropy_with_logits(target, output):
    with tf.name_scope("SigmoidCrossEntropyLoss"):
        # print(target.get_shape())
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output, name="SigmoidCrossEntropy")
        return tf.reduce_mean(loss, axis=1, name="LossMean")

def softmax_cross_entropy_with_logits(target, output):
    with tf.name_scope ("SigmoidCrossEntropyLoss"):
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,
                                                          logits=output,
                                                          dim=1
                                                          )
        return tf.reduce_mean(loss, axis=-1)


class BaseModel(object):
    def __init__(self, classification=3, dataset="big", reload="False", activation='relu', cache=True):
        self.model = None
        self.input = None
        self.output = None
        self._classification = classification
        self.dataset = dataset
        self.reload = reload
        self.cache = cache
        self.activation = activation if activation != 'leaky_relu' else tf.nn.leaky_relu

    def create_model(self):
        return Exception("Not Implemented Error")

    def set_weights(self):
        return Exception("Not Implemented Error")

    def _write_hdf5(self, name, data):
        output_file = os.path.join(self.cache_image, name+'.h5')
        with h5py.File(output_file, "w") as f:
            f.create_dataset('image', data=data, dtype=data.dtype)

    @staticmethod
    def _load_hdf5(input_file):
        with h5py.File(input_file, "r") as f:  # "with" close the file after its nested commands
            return f["image"][()]

    def get_data(self):
        self.cache_image = os.path.join(pylon5_cache, self.dataset, 'image')
        if self.reload:
            utility.remove_directory(self.cache_image)
        if self.cache and os.path.exists(self.cache_image):
            # print(self.cache_image);exit()
            self.train_images = self._load_hdf5(os.path.join(self.cache_image, 'train_images.h5'))
            self.train_labels = self._load_hdf5(os.path.join(self.cache_image, 'train_labels.h5'))
            self.test_images = self._load_hdf5(os.path.join(self.cache_image, 'test_images.h5'))
            self.test_labels = self._load_hdf5(os.path.join(self.cache_image, 'test_labels.h5'))
            return

        self.train_images = prepare_image.load_images(data_type="train", image_type="image",
                                                      classification=self._classification,
                                                      dataset = self.dataset)
        self.train_labels = prepare_image.load_images(data_type="train", image_type="label",
                                                      classification=self._classification,
                                                      dataset=self.dataset)
        self.test_images = prepare_image.load_images(data_type="test", image_type="image",
                                                     classification=self._classification,
                                                     dataset="small")
        self.test_labels = prepare_image.load_images(data_type="test", image_type="label",
                                                     classification=self._classification,
                                                     dataset="small")
        return
        if self.cache and not os.path.exists(self.cache_image):
            utility.create_directory(self.cache_image)
            self._write_hdf5('train_images', self.train_images)
            self._write_hdf5('train_labels', self.train_labels)
            self._write_hdf5('test_images', self.test_images)
            self._write_hdf5('test_labels', self.test_labels)

    def evaluate(self, type="test"):
        if type == "test":
            x_data = self.test_images
            test_data = self.test_labels
        else:
            x_data = self.train_images
            test_data = self.train_labels
        self.predict(data=x_data)
        tf_accuracy(test_data=test_data, predict_data=self.test_predict)
        np_f1score(test_data=test_data, predict_data=self.test_predict)

    def run(self):
        self.fit()
        self.set_weights()
        print("Evaluate results for train data")
        self.evaluate(type="train")
        # self.evaluate(type="test")



    def fit(self):
        return Exception("Not Implemented Error")

    def predict(self, data):
        return Exception("Not Implemented Error")


if __name__ == '__main__':

    args = parse_args()
    rm = RetinaModel(classification=args.classification, dataset=args.dataset,
                     reload=args.reload, activation=args.activation, cache=args.cache)
    rm.create_model()
    rm.set_weights()
    rm.get_data()
    print(rm.test_labels.shape)
    print(rm.train_images.shape)
    # print(rm.model.layers[1].get_weights())
    # print(rm.model.layers[1].output_shape)
    # plot_model(rm.model,"model.png")
    # rm.run()
    # rm.predict()
    K.clear_session()
