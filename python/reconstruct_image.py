import numpy as np
from PIL import Image
from scipy.special import expit
from PIL import ImageDraw


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def reconstruct_image_vec(classification='4_class', data=None):
    if not data:
        return
    predict_label = np.load(data)
    for image_index in range(len(predict_label)):
        image_data = predict_label[image_index]
        image_data = softmax(expit(image_data)).transpose(1, 2, 0)
        image_data_max = np.max(image_data, axis=2)
        background = image_data_max == image_data[..., -1]
        if classification == '4_class':
            a = image_data_max == image_data[..., 0]
            o = image_data_max == image_data[..., 1]
            v = image_data_max == image_data[..., 2]
            image_r = np.where(np.stack((a, o, v), axis=2), 255, 0)
            image = Image.fromarray(np.uint8(image_r), mode='RGB') 
        elif classification == '2_class':
            optic_nerve = image_data_max == image_data[..., 0]
            image_r = np.where(optic_nerve, 255, 0).reshape(565, 565)
            image = Image.fromarray(np.uint8(image_r), mode='L')
        elif classification == '3_class':
            a = image_data_max == image_data[..., 0]
            v = image_data_max == image_data[..., 1]
            z = np.zeros(shape=a.shape)
            image_r = np.where(np.stack((a, z, v), axis=2), 255, 0)
            image = Image.fromarray(np.uint8(image_r), mode='RGB')
    # image.show()
        image.save('data/predict/dev/label_{}.png'.format( image_index))


# reconstruct_image_vec(classification='2_class', data="cache/test_predict2_class.npy")
reconstruct_image_vec(classification='4_class', data="cache/test_predict2_class_4_relu.npy")
# reconstruct_image_vec(classification='3_class', data="cache/test_predict2_class_3.npy")