#!/usr/bin/env python3
import sys
import os
import argparse
import numpy as np
import tensorflow as tf
import json
import prepare_image
import utility
import h5py
import keras

from keras.models import Sequential, Model, model_from_json
from keras.layers import Dense, Flatten, Dropout, Input, concatenate, merge, Add, Lambda
from keras.layers import Conv2D, Conv2DTranspose, Cropping2D, ZeroPadding2D, Activation
from keras.layers import MaxPooling2D, UpSampling2D
from keras import backend as K
import keras.backend.tensorflow_backend as tfb
from keras.utils import plot_model,Sequence
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import SGD,Adam

K.set_image_data_format("channels_first")
cur_dir = os.getcwd()


def sigmoid_cross_entropy_with_logits(target, output):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
                                                   logits=output)
    return tf.reduce_mean(loss,axis=-1)


def softmax_cross_entropy_with_logits(target, output):
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=target,
                                                      logits=output)
    return tf.reduce_mean(loss,axis=-1)


def parse_args():
    """
        function for argument parsing
        :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", "-c", help="Cache data wherever possible", action='store_true')
    parser.add_argument("--classification", "-t", help="Cache data wherever possible",
                        default=4, type=int)
    parser.add_argument("--dataset", "-d", help="dataset small or big",
                         default="big", choices=["small", "big"], type=str)
    parser.add_argument("--reload", "-r", help="reload data", action='store_true')
    parser.add_argument("--log_level", "-l", help="Set loglevel for debugging and analysis",
                         default="INFO")
    args = parser.parse_args()
    return args


class DriveSequence(Sequence):
    def __init__(self, image_train,label_train, batch_size):
        self.batch_size = batch_size
        self.x, self.y = image_train, label_train

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        return self.x[idx * self.batch_size:(idx + 1) * self.batch_size, ...], \
               self.x[idx * self.batch_size:(idx + 1) * self.batch_size, ...]



class RetinaModel(object):
    def __init__(self, classification=3, dataset="big", reload="False"):
        self.model = None
        self.input = None
        self.output = None
        self._classification = classification
        self.dataset = dataset
        self.reload = reload

    def create_model(self):
        input_shape = (3, 565, 565)

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
        side_multi2_up = UpSampling2D(size=(2, 2), name="side_multi2_up")(conv2_2_16)

        upside_multi2 = Cropping2D(cropping=((0, 1),(0, 1)), name="upside_multi2")(side_multi2_up)

        #Decovolution Layer2
        side_multi3_up = UpSampling2D(size=(4, 4), name="side_multi3_up")(conv3_3_16)
        upside_multi3 = Cropping2D(cropping=((1, 2),(1, 2)), name="upside_multi3")(side_multi3_up)

        # Deconvolution Layer3
        side_multi4_up = UpSampling2D(size=(8, 8), name="side_multi4_up")(conv4_3_16)
        upside_multi4 = Cropping2D(cropping=((1, 2),(1, 2)), name="upside_multi4")(side_multi4_up)

        # Specialized Layer
        concat_upscore = concatenate([conv1_2_16, upside_multi2, upside_multi3, upside_multi4],
                                      name="concat-upscore", axis=1)
        upscore_fuse = Conv2D(self._classification, kernel_size=(1, 1), name="upscore_fuse")(concat_upscore)

        self.model = Model(inputs=[data_input], outputs=[upscore_fuse])


    def set_weights(self):
        if args.cache and os.path.exists("cache/keras_crop_model_weights.h5"):
            # self.model.load_weights("cache/keras_crop_model_weights.h5")
            with open("cache/3_class_model.json") as f:
                model_3class = model_from_json(json.dumps(json.load(f)))
            model_3class.load_weights("cache/keras_crop_model_weights.h5")

            for layer, layer3 in zip(self.model.layers, model_3class.layers):
                if(layer.name != 'upscore_fuse' and self._classification !=3) or self._classification == 3:
                    layer.set_weights(layer3.get_weights())

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
        if args.cache and os.path.exists(self.cache_image):
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
                                                     dataset=self.dataset)
        self.test_labels = prepare_image.load_images(data_type="test", image_type="label",
                                                     classification=self._classification,
                                                     dataset=self.dataset)

        if args.cache and not os.path.exists(self.cache_image):
            utility.create_directory(self.cache_image)
            self._write_hdf5('train_images', self.train_images)
            self._write_hdf5('train_labels', self.train_labels)
            self._write_hdf5('test_images', self.test_images)
            self._write_hdf5('test_labels', self.test_labels)

    def run(self):
        print(self.train_images.shape)
        sgd = SGD(lr=1e-6, decay=1e-4, momentum=0.9, nesterov=True)
        weight_save_callback = keras.callbacks.ModelCheckpoint('/cache/checkpoint_weights.hdf5', monitor='val_loss',
                                                verbose=0, save_best_only=True, mode='auto')
        tb_callback = keras.callbacks.TensorBoard(log_dir='./Graph', histogram_freq=0,
                                     write_graph=True, write_images=True)
        tb_callback.set_model(self.model)
        weight_save_callback.set_model(self.model)
        self.model.compile(optimizer=sgd, loss=sigmoid_cross_entropy_with_logits,
                            metrics=['accuracy'])

        # self.model.fit(self.train_images, self.train_labels, batch_size=5, epochs=300)
        sequence = DriveSequence(self.train_images, self.train_labels, batch_size=5)
        self.model.fit_generator(sequence, epochs=2, steps_per_epoch=(int(len(self.train_images)/5)), workers=2)
        self.model.save_weights(os.path.join('cache', 'keras_crop_model_weights_2class.h5'))

    def predict(self):
        test_predict = self.model.predict(self.test_images, batch_size=10)
        print(test_predict[0])
        print(test_predict.shape)
        np.save('cache/test_predict2.npy', test_predict)


if __name__ == '__main__':
    pylon5 = os.environ["SCRATCH"] if os.environ.get("SCRATCH", None) else "."
    pylon5_cache = os.path.join(pylon5, 'cache')
    args = parse_args()
    rm = RetinaModel(classification=args.classification, dataset=args.dataset,
                     reload=args.reload)
    rm.create_model()
    rm.set_weights()
    rm.get_data()
    print(rm.test_labels.shape)
    print(rm.train_images.shape)
    # plot_model(rm.model,"model.png")
    rm.run()
    rm.predict()
    # print(rm.model.summary())
    K.clear_session()
