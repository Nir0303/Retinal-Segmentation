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

    predict_labels = np.load("cache/test_predict2_class_4.npy")

    # predict_labels = softmax(expit(predict_labels))
    test_labels = prepare_image.load_images(data_type="test", image_type="label", classification=4,
                                             dataset="small")
    accuracy_sum = 0
    for image_index in range(len(test_labels)):
        predict_label = predict_labels[image_index]
        predict_label = softmax(expit(predict_label)).transpose(1, 2, 0)
        test_label = test_labels[image_index].transpose(1, 2, 0)
        verify = np.argmax(test_label, axis=-1) == np.argmax(predict_label, axis=-1)
        accuracy_sum += verify.sum()/(565*565)
    print("Accuracy of Model is {}".format(accuracy_sum*100/11))

accuracy()