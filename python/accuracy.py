import numpy as np
from PIL import Image
from scipy.special import expit
from PIL import ImageDraw
import prepare_image

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def accuracy():

    predict_labels = np.load("cache/test_predict2_class_4_1.npy")

    # predict_labels = softmax(expit(predict_labels))
    test_labels = prepare_image.load_images(data_type="test", image_type="label", classification=4,
                                             dataset="small")
    accuracy_sum = 0
    for image_index in range(len(test_labels)):
        predict_label = predict_labels[image_index]
        predict_label = softmax(expit(predict_label)).transpose(1, 2, 0)
        test_label = test_labels[image_index].transpose(1, 2, 0)
        verify = np.argmax(test_label, axis=-1) == np.argmax(predict_label, axis=-1)
        #print(verify.sum())
        accuracy_sum += verify.sum()/(565*565)
        #print(accuracy_sum)
    print("Accuracy of Model is {}".format(accuracy_sum*100/11))

accuracy()

def image_accuracy(y_true, y_pred):
    y_pred = tf.nn.softmax(tf.sigmoid(y_pred, 'predict_sigmoid'), axis=0).transpose(1, 2, 0)
    y_true = y_true.transpose(1, 2, 0)
    accuracy_mask = tf.cast(K.equal(y_pred, y_true), 'int32')
    accuracy = tf.sum(accuracy_mask) / 319225
    return accuracy
