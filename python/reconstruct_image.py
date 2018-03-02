import numpy as np
from PIL import Image


def build_image():
    test_label = np.load("cache/test_predict.npy")
    for i in range(len(test_label)):
        image_data = np.array(test_label[0], dtype=np.int8).transpose(2,1,0)
        image = Image.fromarray(image_data, 'RGB')
        image.save('data/predict/label_{}.png'.format(i))

build_image()